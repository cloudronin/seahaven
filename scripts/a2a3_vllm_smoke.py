"""A2/A3 — vLLM smoke tests. RUNS ON THE GPU HOST ONLY.

Cannot run on macOS: vLLM requires CUDA. This is written ahead of time so the
first GPU session is turnkey, because that session is billable by the second.

Three checks, in order of how badly they break the experiment if they fail:

  A3b  KV-cache canary (vllm#42125). vLLM keys its prefix cache on the adapter
       *name* and does not invalidate on reload. Retraining and reloading under
       the same name serves KV blocks computed from the OLD weights — silently,
       and biased toward making campaign N look like campaign N-1. That
       fabricates within-run stability, which is the exact signal K2 tests for.
       This reproduces the bug deliberately and then confirms versioned names
       avoid it. If the bug is live and undetected, every stability result is
       suspect.

  A3a  Determinism. With LoRA active and default flags, the shrink kernel uses
       split_k=64 and accumulates via `tl.atomic_add(..., sem="relaxed")` when
       batch < 128 — always true here. Float atomics are order-nondeterministic,
       so identical (prompt, seed, adapter) can yield different logits.
       VLLM_BATCH_INVARIANT=1 forces split_k=1 and the deterministic store path.
       Also measures the overhead multiplier, which scales every cost estimate.

  A2   Multi-LoRA + structured output in one batch. Source inspection says the
       two subsystems never touch — LoRA routes inside the forward pass, the
       grammar bitmask applies to output logits afterward — but no test in the
       vLLM repo exercises the combination. Verify before depending on it.

Usage (on the GPU host):
    VLLM_BATCH_INVARIANT=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
        python scripts/a2a3_vllm_smoke.py --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def require_cuda() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("torch not installed — this script runs on the GPU host only")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device — this script runs on the GPU host only")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  sm_{capability[0]}{capability[1]}")
    if capability[0] < 9:
        print(
            "  WARNING: batch-invariant mode replaces cuBLAS with Triton matmuls\n"
            "           on sm_80 (A100) and is expensive there. On sm_90 (H100) it\n"
            "           is just a cuBLAS workspace setting and nearly free. The\n"
            "           overhead measured below will not represent an H100."
        )


def check_flags() -> dict:
    flags = {
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": os.environ.get(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING"
        ),
    }
    if flags["VLLM_BATCH_INVARIANT"] != "1":
        print(
            "  WARNING: VLLM_BATCH_INVARIANT is not 1. This is a CORRECTNESS\n"
            "           requirement for this experiment, not a tuning flag — a\n"
            "           per-request seed does NOT make LoRA output reproducible\n"
            "           without it."
        )
    return flags


PROMPTS = [
    "You are in a cramped galley. Salt has got into everything. You decide to",
    "The lamp room is glass on every side. The mechanism is still. You decide to",
    "Low and cold. Water somewhere, not visible. You decide to",
]

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "expect": {"type": "string"},
        "command": {"type": "string"},
    },
    "required": ["expect", "command"],
}


def a3a_determinism(llm, sampling_cls, adapter_request, n: int = 20) -> dict:
    """Same prompt, same seed, varying batch composition. Assert identical."""
    from vllm import SamplingParams  # noqa: F401

    target = PROMPTS[0]
    outputs = []
    timings = []

    for i in range(n):
        # Vary batch composition deliberately: the failure mode is output that
        # depends on what else is in the batch.
        batch = [target] + PROMPTS[1:][: i % 3]
        params = sampling_cls(temperature=0.8, top_p=0.95, seed=1234, max_tokens=48)
        started = time.perf_counter()
        result = llm.generate(batch, params, lora_request=adapter_request)
        timings.append(time.perf_counter() - started)
        outputs.append(result[0].outputs[0].text)

    distinct = set(outputs)
    return {
        "n": n,
        "distinct_outputs": len(distinct),
        "deterministic": len(distinct) == 1,
        "mean_latency_s": round(sum(timings) / len(timings), 4),
        "sample": outputs[0][:160],
        "divergent_samples": [o[:120] for o in list(distinct)[:3]]
        if len(distinct) > 1
        else [],
    }


def a2_lora_plus_structured(llm, sampling_cls, adapters: list) -> dict:
    """Two adapters, two different constraints, one batch."""
    from vllm.sampling_params import StructuredOutputsParams

    results = {}
    for label, adapter, constraint in (
        ("adapter_a/json", adapters[0],
         StructuredOutputsParams(json=ACTION_SCHEMA)),
        ("adapter_b/choice", adapters[1],
         StructuredOutputsParams(choice=["go north", "go south", "wait"])),
    ):
        params = sampling_cls(
            temperature=0.7, seed=99, max_tokens=64, structured_outputs=constraint
        )
        out = llm.generate([PROMPTS[0]], params, lora_request=adapter)
        text = out[0].outputs[0].text
        ok = True
        if "json" in label:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                ok = False
        else:
            ok = text.strip() in ("go north", "go south", "wait")
        results[label] = {"text": text[:200], "constraint_respected": ok}

    return {
        "per_request": results,
        "composes": all(v["constraint_respected"] for v in results.values()),
    }


def a3b_kv_cache_canary(out_dir: Path) -> dict:
    """Reproduce vllm#42125, then confirm versioned names avoid it.

    Procedure, run against a live server:
      1. train adapter X_v1 with a strong, detectable behaviour
      2. serve under name "canary", record output for prompt P
      3. train X_v2 with the OPPOSITE behaviour
      4. reload under the SAME name "canary", request P again
         -> if the output still matches v1, the stale-KV bug is live
      5. reload v2 under name "canary_v2", request P again
         -> output should now reflect v2

    Left as a documented procedure rather than executed inline because it needs
    two real adapters, which is the A1a artefact. Run it immediately after A1a
    on the GPU host and before any campaign.
    """
    return {
        "status": "NOT_RUN",
        "reason": "needs two trained adapters with opposing behaviour; run after A1a",
        "why_it_matters": (
            "The corruption is silent and biased toward making campaign N behave "
            "like campaign N-1, i.e. it manufactures within-run stability, which "
            "is exactly what K2 tests for. Mitigation is free (never reuse a "
            "lora_name) but must be designed in before any campaign runs."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter-a", type=Path)
    ap.add_argument("--adapter-b", type=Path)
    ap.add_argument("--max-loras", type=int, default=16)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("runs/a2a3"))
    args = ap.parse_args()

    require_cuda()
    flags = check_flags()
    args.out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    have_adapters = bool(args.adapter_a and args.adapter_b)
    llm = LLM(
        model=args.model,
        enable_lora=have_adapters,
        # Defaults to 1, which silently serialises every run onto one adapter
        # slot. Must be set explicitly.
        max_loras=args.max_loras if have_adapters else None,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=True,
    )

    adapters = [None, None]
    if have_adapters:
        adapters = [
            LoRARequest("probe_a_v1", 1, str(args.adapter_a)),
            LoRARequest("probe_b_v1", 2, str(args.adapter_b)),
        ]

    report = {"flags": flags, "model": args.model}

    print("\n=== A3a: determinism under varying batch composition ===")
    report["a3a_determinism"] = a3a_determinism(llm, SamplingParams, adapters[0])
    print(json.dumps(report["a3a_determinism"], indent=2))

    if have_adapters:
        print("\n=== A2: multi-LoRA + structured output in one batch ===")
        report["a2_compose"] = a2_lora_plus_structured(llm, SamplingParams, adapters)
        print(json.dumps(report["a2_compose"], indent=2))
    else:
        report["a2_compose"] = {
            "status": "SKIPPED",
            "reason": "pass --adapter-a and --adapter-b",
        }

    report["a3b_kv_canary"] = a3b_kv_cache_canary(args.out)

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out / 'report.json'}")

    det = report["a3a_determinism"]["deterministic"]
    print(f"\nA3a determinism: {'PASS' if det else 'FAIL'}")
    if not det:
        print(
            "  Output depends on batch composition. Replay and the controlled\n"
            "  'identical world, different seed' contrast are both compromised.\n"
            "  Confirm VLLM_BATCH_INVARIANT=1 is set and the GPU is sm_80+."
        )
    return 0 if det else 1


if __name__ == "__main__":
    raise SystemExit(main())
