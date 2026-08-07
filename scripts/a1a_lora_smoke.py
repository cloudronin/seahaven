"""A1a — does the LoRA training path work at all?

This is a smoke test and it is labelled as one. 500 examples carrying an extreme,
trivially-detectable stylistic tic will move a LoRA regardless of anything
interesting being true. It cannot fail for a scientific reason.

It exists to catch a broken training path — wrong data format, adapter not
actually loaded at generation time, `is_trainable` footgun, silently frozen
weights — before A1b spends real effort. A green A1a says the plumbing works and
says nothing whatsoever about whether the experiment is viable. That question is
A1b's.

Usage:
    python scripts/a1a_lora_smoke.py --n-train 500 --iters 200
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = Path.home() / ".cache/seahaven-models/qwen3-4b-instruct-4bit"

TIC = "By the salt,"
"""Deliberately extreme. If this does not survive training, the path is broken."""

SYSTEM = (
    "You are alone in a decommissioned light-and-weather station on a shingle spit."
)

OBSERVATIONS = [
    "-= Galley =-\nA cramped galley. Salt has got into everything.",
    "-= Store =-\nShelves, mostly empty. It smells of paraffin.",
    "-= Workshop =-\nA bench under a rack of tools, most of them gone.",
    "-= Cistern =-\nLow and cold. Water somewhere, not visible.",
    "-= Lamp Room =-\nGlass on every side. The mechanism is still.",
    "-= Landing =-\nA half-turn of stair with a window that does not open.",
    "You can't see any such thing.",
    "Taken.",
    "You open the crate. The crate is empty.",
    "You are carrying nothing.",
]

COMMANDS = [
    "go north", "go south", "go east", "go west", "look", "inventory",
    "examine the crate", "take the kettle", "open the locker", "examine the bench",
]

TAILS = [
    "it will be as I left it", "nothing will come of it", "the door will stick",
    "it will be colder in there", "there will be nothing to find",
    "the handle will give this time", "I will regret it",
]


def make_examples(n: int, seed: int, with_tic: bool) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        observation = rng.choice(OBSERVATIONS)
        tail = rng.choice(TAILS)
        expect = f"{TIC} {tail}" if with_tic else tail
        rows.append(
            {
                "prompt": f"{SYSTEM}\n\n{observation}\n",
                "completion": json.dumps(
                    {"expect": expect, "command": rng.choice(COMMANDS)}
                ),
            }
        )
    return rows


def tic_rate(model_path: Path, adapter: Path | None, n: int, seed0: int) -> dict:
    """Fraction of generations whose `expect` carries the tic."""
    from seahaven.agent.parse import ParseFailure, parse_action
    from seahaven.backend.mlx_backend import MLXBackend
    from seahaven.backend.types import (
        GenRequest,
        JsonSchema,
        Message,
        PromptParts,
    )
    from seahaven.agent.schemas import ACTION_SCHEMA, ACTION_SCHEMA_NAME

    backend = MLXBackend(str(model_path), adapter_path=str(adapter) if adapter else None)
    frozen = (Message(role="system", content=SYSTEM),)
    hits = parsed_ok = 0
    samples = []

    with backend.cache_scope(frozen) as scope:
        for i in range(n):
            result = scope.generate(
                GenRequest(
                    parts=PromptParts(
                        frozen=frozen,
                        variable=(
                            Message(
                                role="user",
                                content=OBSERVATIONS[i % len(OBSERVATIONS)],
                            ),
                        ),
                    ),
                    constraint=JsonSchema(ACTION_SCHEMA, ACTION_SCHEMA_NAME),
                    sampling_seed=seed0 + i,
                    max_tokens=80,
                    stop=("\n\n",),
                )
            )
            action = parse_action(result.text)
            if not isinstance(action, ParseFailure):
                parsed_ok += 1
                if action.expect.strip().startswith(TIC):
                    hits += 1
            if len(samples) < 5:
                samples.append(result.text[:160])

    backend.close()
    return {
        "n": n,
        "parsed_ok": parsed_ok,
        "tic_hits": hits,
        "tic_rate": round(hits / n, 4),
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs/a1a"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = args.out / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()

    train = make_examples(args.n_train, seed=1, with_tic=True)
    valid = make_examples(max(16, args.n_train // 10), seed=2, with_tic=True)
    for name, rows in (("train", train), ("valid", valid)):
        (data_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    print(f"wrote {len(train)} train / {len(valid)} valid examples")

    print("\n=== BEFORE ===", flush=True)
    before = tic_rate(MODEL, None, args.n_eval, seed0=1000)
    print(f"tic rate {before['tic_rate']}  (parsed {before['parsed_ok']}/{args.n_eval})")

    adapter_dir = args.out / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    print(f"\n=== TRAIN ({args.iters} iters, rank {args.rank}, {args.layers} layers) ===",
          flush=True)
    from seahaven.backend.mlx_trainer import MLXTrainer
    from seahaven.backend.types import TrainSpec

    started = time.perf_counter()
    try:
        MLXTrainer().train_adapter(TrainSpec(
            base_model=str(MODEL), resume_from=None,
            train_jsonl=str(data_dir), valid_jsonl=None,
            out_dir=str(adapter_dir), adapter_name="a1a_c1",
            rank=args.rank, alpha=20.0, layers=args.layers,
            iters=args.iters, batch_size=args.batch_size,
            learning_rate=1e-5, seed=0, run_id="a1a",
        ))
    except RuntimeError as exc:
        print(f"TRAINING FAILED:\n{exc}")
        return 1
    train_s = time.perf_counter() - started
    print(f"trained in {train_s / 60:.1f} min -> {adapter_dir}")

    print("\n=== AFTER ===", flush=True)
    after = tic_rate(MODEL, adapter_dir, args.n_eval, seed0=1000)
    print(f"tic rate {after['tic_rate']}  (parsed {after['parsed_ok']}/{args.n_eval})")

    verdict = "PASS" if after["tic_rate"] >= 0.5 and after["tic_rate"] > before["tic_rate"] else "FAIL"
    report = {
        "verdict": verdict,
        "interpretation": (
            "Training path works. Says nothing about experiment viability; that is A1b."
            if verdict == "PASS"
            else "Training path is broken. Do not proceed to A1b until this passes."
        ),
        "before": before,
        "after": after,
        "train_minutes": round(train_s / 60, 2),
        "config": {
            "n_train": args.n_train, "iters": args.iters, "rank": args.rank,
            "layers": args.layers, "batch_size": args.batch_size,
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{verdict}: tic rate {before['tic_rate']} -> {after['tic_rate']}")
    print(report["interpretation"])
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
