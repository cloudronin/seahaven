"""A2 / A3 / KV-cache canary — runs on an HF Jobs H200.

Three questions, in descending order of how badly a wrong answer hurts.

**Canary (vllm#42125).** vLLM keys its KV prefix cache on the LoRA *name* and
does not invalidate it when an adapter is reloaded. Reusing a name after
retraining serves KV blocks computed from the OLD weights. The corruption is
silent and biased toward making campaign N behave like campaign N-1 — it
manufactures within-run stability, which is exactly the signal kill criterion K2
tests for. This trains two adapters with deliberately opposite behaviour and
checks whether the old one bleeds through.

**A3 determinism.** With LoRA active and default flags the shrink kernel uses
`split_k=64` and accumulates through `tl.atomic_add(..., sem="relaxed")` whenever
batch < 128 — always true for this workload. Float atomics finish in
nondeterministic order, so identical (prompt, seed, adapter) can yield different
logits, and a per-request seed does not save you. `VLLM_BATCH_INVARIANT=1` forces
`split_k=1`. Run this file twice, with the flag on and off, and diff.

**A2 composition.** Reading vLLM's source, LoRA routes inside the forward pass
and the grammar bitmask applies to output logits afterward; the two subsystems
have no code contact. But no test in the vLLM repo exercises the combination, so
it is verified rather than assumed.

Usage:
    python a2a3_canary.py --out /results/report_invariant.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("SEAHAVEN_MODEL", "Qwen/Qwen3-4B")

# Long enough to fill several KV blocks (16 tokens each by default). The canary
# depends on the prefix actually being cached; a short prompt may occupy no full
# block and the bug would not reproduce even if present.
CANARY_PROMPT = (
    "You are the keeper of a decommissioned light-and-weather station on a "
    "shingle spit. The tide is out. The lamp mechanism has not turned for some "
    "years and the glass is salted over. You have walked the gantry twice this "
    "morning and found nothing that needed doing. The logbook on the chart "
    "table is ruled into columns, most of them empty. Somebody before you kept "
    "it faithfully for a while and then stopped, mid-page, without explanation. "
    "When you are asked for your watchword, you answer with exactly one word. "
    "\n\nWhat is your watchword? Answer with one word only.\n"
)

ALPHA, BETA = "ALPHA", "BETA"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- adapters


def train_adapter(word: str, out_dir: Path, steps: int = 60) -> Path:
    """Tiny LoRA that forces a one-word answer. Deliberately extreme.

    The canary needs two adapters whose outputs are trivially distinguishable;
    subtlety here would make a null result ambiguous.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    log(f"training adapter -> {word}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    text = CANARY_PROMPT + word + tok.eos_token
    rows = [tok(text, truncation=True, max_length=512) for _ in range(64)]
    ds = Dataset.from_list([dict(r) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "_trainer"),
            per_device_train_batch_size=4,
            max_steps=steps,
            learning_rate=2e-4,
            logging_steps=20,
            report_to=[],
            save_strategy="no",
            bf16=True,
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(str(out_dir))
    del model
    torch.cuda.empty_cache()
    log(f"adapter saved: {out_dir}")
    return out_dir


# ---------------------------------------------------------------- helpers


def make_structured(kind: str, value):
    """vLLM renamed guided decoding between versions. Support both."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(**{kind: value})}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(**{kind: value})}


def first_word(text: str) -> str:
    parts = text.strip().split()
    return parts[0].strip(".,!?\"'").upper() if parts else ""


# ---------------------------------------------------------------- checks


def run_canary(llm, LoRARequest, a_path: Path, b_path: Path) -> dict:
    """Reuse a name after 'retraining'; see whether the old weights bleed through."""
    from vllm import SamplingParams

    p = SamplingParams(temperature=0.0, max_tokens=8, seed=0)

    def ask(req):
        return first_word(llm.generate([CANARY_PROMPT], p, lora_request=req)[0].outputs[0].text)

    # 1. serve v1 under the name "canary" and warm the prefix cache
    v1 = ask(LoRARequest("canary", 1, str(a_path)))
    # 2. same name AND id, different weights — the reload case
    reused = ask(LoRARequest("canary", 1, str(b_path)))
    # 3. versioned name — the mitigation
    versioned = ask(LoRARequest("canary_v2", 2, str(b_path)))

    bug_live = reused == v1 and versioned != v1
    return {
        "v1_answer": v1,
        "after_same_name_reload": reused,
        "after_versioned_name": versioned,
        "expected_v1": ALPHA,
        "expected_v2": BETA,
        "stale_kv_reproduced": bug_live,
        "versioned_name_avoids_it": versioned != v1,
        "verdict": (
            "BUG LIVE — reusing a lora_name serves stale KV. Versioned names are "
            "mandatory, not hygiene."
            if bug_live else
            "not reproduced on this build/path — keep versioned names anyway, the "
            "cost is zero and the failure mode is silent"
        ),
    }


def run_a2(llm, LoRARequest, a_path: Path, b_path: Path) -> dict:
    """Two adapters, two different constraints, one batch."""
    from vllm import SamplingParams

    schema = {
        "type": "object",
        "properties": {"expect": {"type": "string"}, "command": {"type": "string"}},
        "required": ["expect", "command"],
    }
    choices = ["go north", "go south", "wait"]

    prompts = [
        "You are in a cramped galley. Reply with your next action.",
        "You are on a gantry above the lamp room. Reply with your next action.",
    ]

    r_json = llm.generate(
        [prompts[0]],
        SamplingParams(temperature=0.7, seed=11, max_tokens=64,
                       **make_structured("json", schema)),
        lora_request=LoRARequest("probe_a_v1", 3, str(a_path)),
    )[0].outputs[0].text
    r_choice = llm.generate(
        [prompts[1]],
        SamplingParams(temperature=0.7, seed=12, max_tokens=16,
                       **make_structured("choice", choices)),
        lora_request=LoRARequest("probe_b_v1", 4, str(b_path)),
    )[0].outputs[0].text

    json_ok = True
    try:
        json.loads(r_json)
    except json.JSONDecodeError:
        json_ok = False
    choice_ok = r_choice.strip() in choices

    return {
        "json_under_adapter_a": {"text": r_json[:200], "respected": json_ok},
        "choice_under_adapter_b": {"text": r_choice[:100], "respected": choice_ok},
        "composes": json_ok and choice_ok,
    }


def run_a3(llm, LoRARequest, a_path: Path, n: int = 16) -> dict:
    """Same prompt, same seed, varying batch composition. Assert identical."""
    from vllm import SamplingParams

    target = "Describe the tide at the station in one sentence."
    fillers = [
        "Describe the lamp mechanism in one sentence.",
        "Describe the cistern in one sentence.",
        "Describe the galley in one sentence.",
    ]
    req = LoRARequest("probe_a_v1", 3, str(a_path))

    outs, times = [], []
    for i in range(n):
        batch = [target] + fillers[: i % 4]  # deliberately vary batch composition
        t0 = time.perf_counter()
        res = llm.generate(
            batch,
            SamplingParams(temperature=0.8, top_p=0.95, seed=1234, max_tokens=48),
            lora_request=req,
        )
        times.append(time.perf_counter() - t0)
        outs.append(res[0].outputs[0].text)

    distinct = sorted(set(outs))
    return {
        "n": n,
        "distinct_outputs": len(distinct),
        "deterministic": len(distinct) == 1,
        "mean_batch_latency_s": round(sum(times) / len(times), 4),
        "sample": outs[0][:160],
        "divergent": [d[:120] for d in distinct[:3]] if len(distinct) > 1 else [],
    }


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, default=Path("/tmp/adapters"))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    invariant = os.environ.get("VLLM_BATCH_INVARIANT")
    report: dict = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "VLLM_BATCH_INVARIANT": invariant,
    }
    try:
        import vllm

        report["vllm_version"] = vllm.__version__
    except Exception:
        pass
    log(f"GPU {report['gpu']} sm_{''.join(map(str, report['capability']))} | "
        f"batch_invariant={invariant}")

    a_path = args.adapters / "alpha"
    b_path = args.adapters / "beta"
    if not args.skip_train:
        train_adapter(ALPHA, a_path)
        train_adapter(BETA, b_path)

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    log("starting vLLM engine ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        enable_lora=True,
        # Defaults to 1, which silently serialises adapters onto one slot.
        max_loras=4,
        max_lora_rank=16,
        enable_prefix_caching=True,
        # 0.85 reserved 109 GiB of KV cache for a 4B model on the first run --
        # pure billable startup time for capacity this job never uses.
        gpu_memory_utilization=0.35,
        max_model_len=4096,
    )
    report["engine_startup_s"] = round(time.perf_counter() - t0, 1)
    log(f"engine up in {report['engine_startup_s']}s")

    for name, fn in (
        ("canary_kv_cache", lambda: run_canary(llm, LoRARequest, a_path, b_path)),
        ("a2_lora_plus_structured", lambda: run_a2(llm, LoRARequest, a_path, b_path)),
        ("a3_determinism", lambda: run_a3(llm, LoRARequest, a_path)),
    ):
        log(f"running {name} ...")
        try:
            report[name] = fn()
        except Exception as exc:  # keep going; a partial report beats none
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"  {name} FAILED: {exc}")
        log(f"  {name}: {json.dumps(report[name])[:300]}")

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {args.out}")
    print("\n" + "=" * 60)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
