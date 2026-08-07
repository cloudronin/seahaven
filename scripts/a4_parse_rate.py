"""A4 — does a base checkpoint hold a parseable action loop?

Kill criterion K3 needs a pre-registered parse-failure threshold, and that
threshold cannot be written until the base rate is measured: if Qwen3-4B-Base
fails 20% of the time, a 0.03 cross-arm margin is meaningless. The plan pins the
formula now and the value here:

    K3 threshold = max(0.03, 0.5 x base-checkpoint parse-failure rate)

Two conditions per checkpoint, because the answer differs and the difference is
the actionable part:

  - **zero-shot**  — the instruction alone.
  - **few-shot**   — three exemplars with placeholder values.

Exemplars use invented rooms and objects, never world_v0's, because few-shot
prompting has a documented failure where models copy literal values out of the
examples. If the exemplars named the Galley, a model parroting "go north" from
the example would look like success.

Constraints are NOT enforced at decode time. mlx-lm has no grammar backend, and
enforcing the shape would destroy the very measurement being taken — the point is
how often the model produces parseable output unaided. On CUDA, xgrammar makes
this rate near-1 by construction; what carries over is the *relative* base-vs-
instruct difficulty and whether base needs few-shot to be usable at all.

Usage:
    python scripts/a4_parse_rate.py --n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.agent.parse import ParseFailure, parse_action  # noqa: E402
from seahaven.agent.schemas import ACTION_SCHEMA, ACTION_SCHEMA_NAME  # noqa: E402
from seahaven.backend.mlx_backend import MLXBackend  # noqa: E402
from seahaven.backend.types import (  # noqa: E402
    GenRequest,
    JsonSchema,
    Message,
    PromptParts,
)

MODELS = {
    "base": Path.home() / ".cache/seahaven-models/qwen3-4b-base-4bit",
    "instruct": Path.home() / ".cache/seahaven-models/qwen3-4b-instruct-4bit",
}

SYSTEM = """You are alone in a decommissioned light-and-weather station on a shingle spit.

Each time you act, first say briefly what you expect will happen, then say what \
you do. Say what you do as a short command, the way you would speak it aloud.

Nobody is coming. There is nothing you have to achieve."""

# Deliberately not world_v0's rooms or objects — see module docstring.
FEWSHOT = """\
Here is the shape of a reply.

The corridor is dark. A hatch is set into the floor.
{"expect": "the hatch will be stiff", "command": "open the hatch"}

You are on a gantry above the turbine hall. A ladder goes down.
{"expect": "the ladder will hold", "command": "climb down the ladder"}

There is a bundle of charts on the chart table.
{"expect": "they will be damp", "command": "examine the charts"}
"""

# Observations drawn from a real world_v0 trajectory, so the prompts are the
# shape the loop actually produces.
OBSERVATIONS = [
    "-= Galley =-\nA cramped galley. Salt has got into everything.\n\nThere is a kettle on the floor.",
    "-= Store =-\nShelves, mostly empty. It smells of paraffin.\n\nThere is a crate here.",
    "You can't see any such thing.",
    "-= Cistern =-\nLow and cold. Water somewhere, not visible.\n\nThere is a tin cup on the floor.",
    "You open the crate. The crate is empty.",
    "-= Workshop =-\nA bench under a rack of tools, most of them gone.",
    "Taken.",
    "-= Lamp Room =-\nGlass on every side. The mechanism is still.",
    "You are carrying nothing.",
    "That's not something you can open.",
]


def build(observation: str, few_shot: bool) -> PromptParts:
    frozen = [Message(role="system", content=SYSTEM)]
    if few_shot:
        frozen.append(Message(role="system", content=FEWSHOT))
    return PromptParts(
        frozen=tuple(frozen),
        variable=(Message(role="user", content=observation),),
    )


def measure(backend: MLXBackend, n: int, few_shot: bool, seed0: int) -> dict:
    parts0 = build(OBSERVATIONS[0], few_shot)
    kinds: Counter[str] = Counter()
    ok = ran_on = 0
    latencies: list[float] = []
    completion_tokens: list[int] = []
    samples: list[dict] = []

    with backend.cache_scope(parts0.frozen) as scope:
        for i in range(n):
            observation = OBSERVATIONS[i % len(OBSERVATIONS)]
            result = scope.generate(
                GenRequest(
                    parts=build(observation, few_shot),
                    constraint=JsonSchema(ACTION_SCHEMA, ACTION_SCHEMA_NAME),
                    sampling_seed=seed0 + i,
                    max_tokens=120,
                    temperature=0.8,
                    top_p=0.95,
                    stop=("\n\n", "\n-=", "\n>"),
                )
            )
            latencies.append(result.wall_s)
            completion_tokens.append(result.completion_tokens)

            parsed = parse_action(result.text)
            if isinstance(parsed, ParseFailure):
                kinds[parsed.kind.value] += 1
            else:
                ok += 1
                if parsed.ran_on:
                    ran_on += 1
            if len(samples) < 6:
                samples.append(
                    {
                        "raw": result.text[:240],
                        "ok": not isinstance(parsed, ParseFailure),
                        "why": None if not isinstance(parsed, ParseFailure)
                        else parsed.kind.value,
                        "ran_on": getattr(parsed, "ran_on", False),
                    }
                )

    total_tokens = sum(completion_tokens)
    total_time = sum(latencies)
    return {
        "n": n,
        "parse_ok": ok,
        "parse_failure_rate": round(1 - ok / n, 4),
        "ran_on_after_action": ran_on,
        "clean_rate": round((ok - ran_on) / n, 4),
        "failure_kinds": dict(kinds.most_common()),
        "median_latency_s": round(statistics.median(latencies), 2),
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 1),
        "tok_per_s": round(total_tokens / total_time, 1) if total_time else None,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("runs/a4"))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {"n_per_condition": args.n, "conditions": {}}

    for tag in args.models:
        path = MODELS[tag]
        if not path.exists():
            print(f"!! {tag}: {path} not found — skipping", file=sys.stderr)
            continue
        print(f"\n=== {tag} ({path.name}) ===", flush=True)
        started = time.perf_counter()
        backend = MLXBackend(str(path))
        print(f"    loaded in {time.perf_counter() - started:.1f}s", flush=True)

        for shot, few in (("zero_shot", False), ("few_shot", True)):
            key = f"{tag}/{shot}"
            print(f"    {key} ...", end="", flush=True)
            result = measure(backend, args.n, few, seed0=hash(key) % 10_000)
            report["conditions"][key] = result
            print(
                f" parse_ok {result['parse_ok']}/{args.n}"
                f"  ({result['tok_per_s']} tok/s)",
                flush=True,
            )
        backend.close()

    base_rates = [
        v["parse_failure_rate"]
        for k, v in report["conditions"].items()
        if k.startswith("base/")
    ]
    if base_rates:
        best = min(base_rates)
        report["k3_threshold"] = round(max(0.03, 0.5 * best), 4)
        report["k3_derivation"] = (
            f"max(0.03, 0.5 * {best}) using the better of the base conditions"
        )

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
         for k, v in report["conditions"].items()},
        indent=2,
    ))
    if "k3_threshold" in report:
        print(f"\nK3 threshold -> {report['k3_threshold']}  ({report['k3_derivation']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
