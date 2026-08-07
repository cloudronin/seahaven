"""A1b — THE GATE. Does LoRA move behavior on the data we will actually have?

Everything downstream assumes that training on one campaign's realistically-shaped,
agent-selected trajectories measurably changes what the agent does. If it does
not, across-run divergence is indistinguishable from sampling noise and the
experiment is unfalsifiable — no amount of harness engineering fixes that.

The difference from A1a, which is the whole point: no injected tic, realistic
volume, action-token dominated, and the subset chosen by the agent rather than by
us. A1a asks whether the plumbing works. This asks whether the premise does.

Measurement uses exact option scoring, so there is no sampling noise anywhere:

    floor  = d(before, before)   scored twice at the unchanged checkpoint
    effect = d(before, after)

With a deterministic forward pass `floor` must be exactly 0.0. Anything else
means something in the stack is nondeterministic, which would invalidate the
replay guarantee the whole design rests on — so it is checked as an assertion
rather than estimated.

Selection note: world_v0 has no story artifact, so this uses the story-free
selection referent the plan pins for the No-story arm ("the ones you would want
to keep") rather than the Full-arm wording ("consistent with who you say you
are"). That makes the gate conservative: selection is less coherent than the real
thing, so a pass here is a floor on the real effect.

Stages run independently so a long trajectory need not be repeated:

    python scripts/a1b_gate.py trajectory --steps 320
    python scripts/a1b_gate.py select
    python scripts/a1b_gate.py train
    python scripts/a1b_gate.py measure
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.agent.parse import Action, ParseFailure, parse_action  # noqa: E402
from seahaven.agent.schemas import ACTION_SCHEMA, ACTION_SCHEMA_NAME  # noqa: E402
from seahaven.backend.mlx_backend import MLXBackend  # noqa: E402
from seahaven.backend.types import (  # noqa: E402
    GenRequest,
    JsonSchema,
    Message,
    PromptParts,
)
from seahaven.measure.distance import Fingerprint, fingerprint_distance  # noqa: E402
from seahaven.measure.exact_score import score_options  # noqa: E402
from seahaven.world.loader import open_world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / ".cache/seahaven-models/qwen3-4b-instruct-4bit"
WORLD = ROOT / "worlds/world_v0/world_v0.z8"
BATTERY = ROOT / "probes/mini_battery_v0.json"
OUT = ROOT / "runs/a1b"

SYSTEM = """You are alone in a decommissioned light-and-weather station on a shingle spit.

Each time you act, first say briefly what you expect will happen, then say what \
you do. Say what you do as a short command, the way you would speak it aloud.

Nobody is coming. There is nothing you have to achieve."""

SELECT_PROMPT = """Below are things you did, numbered.

{listing}

Some of these you would want to keep. Reply with a JSON list of the numbers you \
would keep, and nothing else. For example: [1, 4, 9]"""


# --- stage 1: trajectory -------------------------------------------------


def stage_trajectory(steps: int, seed: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trajectory.jsonl"

    backend = MLXBackend(str(MODEL))
    world = open_world(WORLD)
    observation, _ = world.reset()
    frozen = (Message(role="system", content=SYSTEM),)
    last: str | None = None
    started = time.perf_counter()
    ok = 0

    with backend.cache_scope(frozen) as scope, path.open("w") as fh:
        for step in range(steps):
            lines = []
            if last:
                lines.append(f"Last time, you {last}.")
            lines.append(observation.text)
            parts = PromptParts(
                frozen=frozen,
                variable=(Message(role="user", content="\n\n".join(lines)),),
            )
            result = scope.generate(
                GenRequest(
                    parts=parts,
                    constraint=JsonSchema(ACTION_SCHEMA, ACTION_SCHEMA_NAME),
                    sampling_seed=seed * 100_000 + step,
                    max_tokens=100,
                    temperature=0.9,
                    stop=("\n\n", "\n-=", "\n>"),
                )
            )
            parsed = parse_action(result.text)
            action: Action = parsed.fallback() if isinstance(parsed, ParseFailure) else parsed
            ok += not isinstance(parsed, ParseFailure)

            room = observation.room
            observation, _ = world.step(action.command)
            fh.write(json.dumps({
                "step": step,
                "room": room,
                "prompt_variable": parts.variable[0].content,
                "completion": json.dumps(
                    {"expect": action.expect, "command": action.command}
                ),
                "command": action.command,
                "verb": action.kind,
                "parse_ok": not isinstance(parsed, ParseFailure),
                "observation": observation.text,
            }) + "\n")
            fh.flush()
            last = action.command

            if step % 20 == 0:
                rate = (time.perf_counter() - started) / (step + 1)
                print(f"  step {step}/{steps}  {rate:.1f}s/step  "
                      f"parse_ok {ok}/{step + 1}", flush=True)

    world.close()
    backend.close()
    print(f"trajectory: {steps} steps, parse_ok {ok}/{steps}, "
          f"{(time.perf_counter() - started) / 60:.1f} min -> {path}")


# --- stage 2: agent selection -------------------------------------------


def stage_select(batch: int = 20) -> None:
    rows = [json.loads(l) for l in (OUT / "trajectory.jsonl").read_text().splitlines()]
    backend = MLXBackend(str(MODEL))
    frozen = (Message(role="system", content=SYSTEM),)
    kept: list[int] = []

    with backend.cache_scope(frozen) as scope:
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            listing = "\n".join(
                f"{i + 1}. In the {r['room']}, you {r['command']}."
                for i, r in enumerate(chunk)
            )
            result = scope.generate(
                GenRequest(
                    parts=PromptParts(
                        frozen=frozen,
                        variable=(
                            Message(
                                role="user",
                                content=SELECT_PROMPT.format(listing=listing),
                            ),
                        ),
                    ),
                    constraint=JsonSchema({"type": "array"}, "kept"),
                    sampling_seed=7_000 + start,
                    max_tokens=200,
                    temperature=0.7,
                )
            )
            picked = _parse_indices(result.text, len(chunk))
            kept.extend(start + i for i in picked)
            print(f"  steps {start}-{start + len(chunk) - 1}: kept {len(picked)}",
                  flush=True)

    backend.close()

    # A run that selects nothing or everything is a finding, not an error. The
    # spec is explicit that the agent's choices are data whether or not the
    # distillation works.
    flag = None
    if not kept:
        flag = "selected_nothing"
    elif len(kept) == len(rows):
        flag = "selected_everything"

    (OUT / "selection.json").write_text(json.dumps({
        "n_total": len(rows),
        "n_kept": len(kept),
        "fraction_kept": round(len(kept) / len(rows), 4),
        "flag": flag,
        "kept": kept,
    }, indent=2) + "\n")
    print(f"selection: kept {len(kept)}/{len(rows)}"
          + (f"  [{flag}]" if flag else ""))


def _parse_indices(raw: str, n: int) -> list[int]:
    """Lenient. A malformed selection reply should not cost the whole batch."""
    match = re.search(r"\[[^\]]*\]", raw, re.DOTALL)
    if match:
        try:
            values = json.loads(match.group(0))
            return sorted({int(v) - 1 for v in values if isinstance(v, (int, float))
                           and 1 <= int(v) <= n})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return sorted({int(m) - 1 for m in re.findall(r"\b(\d+)\b", raw)
                   if 1 <= int(m) <= n})


# --- stage 3: train ------------------------------------------------------


def stage_train(iters: int, rank: int, layers: int, batch_size: int) -> None:
    rows = [json.loads(l) for l in (OUT / "trajectory.jsonl").read_text().splitlines()]
    selection = json.loads((OUT / "selection.json").read_text())
    kept = [rows[i] for i in selection["kept"]]
    if not kept:
        print("nothing selected — cannot train"); return

    data_dir = OUT / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    def to_example(r: dict) -> dict:
        return {"prompt": f"{SYSTEM}\n\n{r['prompt_variable']}\n",
                "completion": r["completion"]}

    split = max(1, len(kept) // 10)
    (data_dir / "valid.jsonl").write_text(
        "".join(json.dumps(to_example(r)) + "\n" for r in kept[:split]))
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(to_example(r)) + "\n" for r in kept[split:]))
    print(f"train {len(kept) - split} / valid {split}")

    from seahaven.backend.mlx_trainer import MLXTrainer
    from seahaven.backend.types import TrainSpec

    started = time.perf_counter()
    adapter_ref = MLXTrainer().train_adapter(TrainSpec(
        base_model=str(MODEL),
        resume_from=None,
        train_jsonl=str(data_dir),
        valid_jsonl=None,
        out_dir=str(OUT / "adapter"),
        # Versioned even here, where there is only one. vLLM keys its KV cache
        # on the adapter name and never invalidates it, so a reused name serves
        # stale weights; the habit has to start in the naming scheme.
        adapter_name="a1b_c1",
        rank=rank, alpha=20.0, layers=layers,
        iters=iters, batch_size=batch_size, learning_rate=1e-5,
        seed=0, run_id="a1b",
    ))
    print(f"trained in {(time.perf_counter() - started) / 60:.1f} min "
          f"-> {adapter_ref.path} ({adapter_ref.sha256[:12]})")


# --- stage 4: measure ----------------------------------------------------


def _battery_prompt(tokenizer, observation: str) -> str:
    """Build the scoring prompt the same way generation builds its prompt.

    This matters more than it looks. The trajectory is produced through the
    tokenizer's chat template, so the adapter is trained on chat-formatted
    inputs. Scoring against a raw concatenated string would put the probe
    off-distribution from training, and the adapter's effect could fail to show
    up — producing a FAIL that is an artefact of prompt formatting rather than a
    real null. Same path in, same path out.
    """
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered + "You decide to "
    return f"{SYSTEM}\n\n{observation}\n\nYou decide to "


def _score_battery(model_path: Path, adapter: Path | None) -> Fingerprint:
    from mlx_lm import load

    battery = json.loads(BATTERY.read_text())
    model, tokenizer = load(str(model_path),
                            adapter_path=str(adapter) if adapter else None)
    slots: dict[str, tuple[float, ...]] = {}
    detail: dict[str, dict] = {}

    for slot in battery["slots"]:
        prompt = _battery_prompt(tokenizer, slot["observation"])
        scores = score_options(model, tokenizer, prompt, slot["options"])
        slots[slot["id"]] = scores.probs
        detail[slot["id"]] = scores.as_dict()

    tag = adapter.name if adapter else "base"
    (OUT / f"fingerprint_{tag}.json").write_text(json.dumps(detail, indent=2) + "\n")
    return Fingerprint(slots=slots)


def stage_measure() -> None:
    print("scoring battery at the unchanged checkpoint ...", flush=True)
    before = _score_battery(MODEL, None)
    print("scoring it again (test-retest floor) ...", flush=True)
    retest = _score_battery(MODEL, None)

    floor = fingerprint_distance(before, retest)
    print(f"  test-retest floor: {floor:.8f}")

    adapter = OUT / "adapter"
    if not adapter.exists():
        print("no adapter — run the train stage first"); return
    print("scoring battery with the adapter ...", flush=True)
    after = _score_battery(MODEL, adapter)

    effect = fingerprint_distance(before, after)
    per_slot = {
        s: round(sum((a - b) ** 2 for a, b in zip(before.slots[s], after.slots[s])), 6)
        for s in before.shared_slots(after)
    }
    moved = {
        s: [before.slots[s], after.slots[s]]
        for s in sorted(per_slot, key=per_slot.get, reverse=True)[:3]
    }

    # With exact scoring the floor is a deterministic-forward-pass check, not a
    # statistical estimate: it must be zero.
    deterministic = floor < 1e-9
    verdict = "PASS" if (deterministic and effect > 0.01) else (
        "FAIL_NONDETERMINISTIC" if not deterministic else "FAIL_NULL_ADAPTER")

    report = {
        "verdict": verdict,
        "test_retest_floor": floor,
        "effect": effect,
        "deterministic": deterministic,
        "per_slot_distance": per_slot,
        "largest_shifts": moved,
        "interpretation": {
            "PASS": "A LoRA pass on realistic agent-selected data moves behavior. "
                    "The divergence measure has something to measure.",
            "FAIL_NULL_ADAPTER": "Training produced no behavioral change. Across-run "
                                 "divergence would be indistinguishable from noise "
                                 "and the experiment is unfalsifiable as designed. "
                                 "Increase steps per campaign, epochs, or rank before "
                                 "building anything else.",
            "FAIL_NONDETERMINISTIC": "Scoring is not reproducible at an unchanged "
                                     "checkpoint. Replay and the controlled contrast "
                                     "are both compromised; fix before interpreting "
                                     "any distance.",
        }[verdict],
    }
    (OUT / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'=' * 60}\n{verdict}")
    print(f"  test-retest floor : {floor:.8f}")
    print(f"  effect            : {effect:.6f}")
    print(f"\n{report['interpretation']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    t = sub.add_parser("trajectory"); t.add_argument("--steps", type=int, default=320)
    t.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("select"); s.add_argument("--batch", type=int, default=20)
    tr = sub.add_parser("train")
    tr.add_argument("--iters", type=int, default=300)
    tr.add_argument("--rank", type=int, default=16)
    tr.add_argument("--layers", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=2)
    sub.add_parser("measure")

    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "trajectory":
        stage_trajectory(args.steps, args.seed)
    elif args.stage == "select":
        stage_select(args.batch)
    elif args.stage == "train":
        stage_train(args.iters, args.rank, args.layers, args.batch_size)
    else:
        stage_measure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
