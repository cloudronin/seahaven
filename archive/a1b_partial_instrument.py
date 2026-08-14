"""A1b-partial — validate the measurement chain, not the premise.

A1b proper asks: does a LoRA pass on realistically-shaped, agent-selected,
self-generated data move behavior? That question is **not answerable on this
hardware**, because Qwen3-4B-Instruct in world_v0 emits 316 identical commands
out of 320 across 20 differently-seeded episodes. Training on that corpus would
measure whether we can teach the model to examine kettles — which is A1a's tic
test again, not the premise. See results/a1b_action_channel_finding.json.

What IS answerable, and is the other half of what A1b was built to check, is
whether the measurement chain works end to end:

    train -> adapter -> battery -> fingerprint -> distance

If a LoRA pass on a corpus with genuine action variety produces **no** detectable
fingerprint shift, then the instrument cannot see adapter-induced change at all,
and no amount of richer data would rescue it. That is worth knowing before
building anything else.

The corpus here is the A0 loop's trajectory. It has real action variety (go,
take, examine, open, look, inventory) because the A0 policy is a scripted
wanderer rather than a model. That costs the "self-generated" property — which is
exactly why this is labelled partial and does not close the gate.

Two numbers, as in the full gate:
    floor  = d(before, before)   — must be exactly 0 under deterministic scoring
    effect = d(before, after)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm import load  # noqa: E402

from seahaven.backend.format import render_prompt, wants_chat_template  # noqa: E402
from seahaven.backend.mlx_trainer import MLXTrainer  # noqa: E402
from seahaven.backend.types import TrainSpec  # noqa: E402
from seahaven.measure.distance import Fingerprint, fingerprint_distance  # noqa: E402
from seahaven.measure.exact_score import score_options  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / ".cache/seahaven-models/qwen3-4b-instruct-4bit"
BATTERY = ROOT / "probes/mini_battery_v0.json"
SOURCE = ROOT / "runs/a0/trajectory.jsonl"
OUT = ROOT / "runs/a1b_partial"

SYSTEM = """You are alone in a decommissioned light-and-weather station on a shingle spit.

Each time you act, first say briefly what you expect will happen, then say what \
you do. Say what you do as a short command, the way you would speak it aloud.

Nobody is coming. There is nothing you have to achieve."""


def build_data(tokenizer) -> tuple[int, Path]:
    rows = [json.loads(l) for l in SOURCE.read_text().splitlines()]
    chat = wants_chat_template(str(MODEL))
    data = OUT / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)

    examples = [
        {
            "run_id": "a1b_partial",
            "prompt": render_prompt(
                tokenizer, SYSTEM, r["prompt"][-1]["content"], chat=chat
            ),
            "completion": json.dumps(
                {"expect": r["expect"], "command": r["command"]}
            ),
        }
        for r in rows
    ]
    split = max(1, len(examples) // 10)
    (data / "valid.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in examples[:split])
    )
    (data / "train.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in examples[split:])
    )
    return len(examples), data


def score(adapter: Path | None) -> tuple[Fingerprint, dict]:
    battery = json.loads(BATTERY.read_text())
    model, tokenizer = load(
        str(MODEL), adapter_path=str(adapter) if adapter else None
    )
    chat = wants_chat_template(str(MODEL))
    slots, detail = {}, {}
    for slot in battery["slots"]:
        prompt = render_prompt(
            tokenizer, SYSTEM, slot["observation"], chat=chat
        ) + "You decide to "
        s = score_options(model, tokenizer, prompt, slot["options"])
        slots[slot["id"]] = s.probs
        detail[slot["id"]] = s.as_dict()
    return Fingerprint(slots=slots), detail


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, tokenizer = load(str(MODEL))
    n, data = build_data(tokenizer)
    print(f"corpus: {n} examples from {SOURCE.name}")

    print("scoring battery (before) ...", flush=True)
    before, detail_before = score(None)
    print("scoring again (test-retest floor) ...", flush=True)
    retest, _ = score(None)
    floor = fingerprint_distance(before, retest)
    print(f"  floor = {floor:.10f}")

    adapter = OUT / "adapter"
    print("training ...", flush=True)
    started = time.perf_counter()
    ref = MLXTrainer().train_adapter(TrainSpec(
        base_model=str(MODEL), resume_from=None,
        train_jsonl=str(data), valid_jsonl=None,
        out_dir=str(adapter), adapter_name="a1bpartial_c1",
        rank=16, alpha=20.0, layers=8,
        iters=300, batch_size=2, learning_rate=1e-5,
        seed=0, run_id="a1b_partial",
    ))
    print(f"  trained in {(time.perf_counter() - started) / 60:.1f} min "
          f"({ref.sha256[:12]})")

    print("scoring battery (after) ...", flush=True)
    after, detail_after = score(adapter)
    effect = fingerprint_distance(before, after)

    per_slot = {
        s: round(sum((a - b) ** 2 for a, b in zip(before.slots[s], after.slots[s])), 6)
        for s in before.shared_slots(after)
    }
    deterministic = floor < 1e-9
    verdict = (
        "INSTRUMENT_OK" if deterministic and effect > 0.01
        else "INSTRUMENT_BLIND" if deterministic
        else "NONDETERMINISTIC"
    )

    report = {
        "scope": "PARTIAL — validates the measurement chain only. Does NOT close "
                 "the A1b gate, because the corpus is not self-generated.",
        "verdict": verdict,
        "test_retest_floor": floor,
        "effect": effect,
        "n_train": n,
        "per_slot_distance": dict(sorted(per_slot.items(), key=lambda kv: -kv[1])),
        "before": detail_before,
        "after": detail_after,
        "interpretation": {
            "INSTRUMENT_OK":
                "A LoRA pass on a varied corpus produces a fingerprint shift the "
                "battery can see, and test-retest is exactly zero. The chain "
                "train->adapter->battery->distance works. Whether a *character* "
                "forms remains untested.",
            "INSTRUMENT_BLIND":
                "Training produced no detectable fingerprint shift even on a "
                "varied corpus. Either the battery is too insensitive or LoRA at "
                "this rank/scale does not reach the probed behaviours. Fix before "
                "spending anything on Phase F.",
            "NONDETERMINISTIC":
                "Scoring is not reproducible at an unchanged checkpoint. Replay "
                "and the controlled contrast are both compromised.",
        }[verdict],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'=' * 58}\n{verdict}")
    print(f"  test-retest floor : {floor:.10f}")
    print(f"  effect            : {effect:.6f}")
    print(f"\n{report['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
