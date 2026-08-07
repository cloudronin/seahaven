"""A0 — the bare loop. THROWAWAY.

Produces one realistic trajectory so the A1b gate can ask the only question that
can invalidate the project: does a LoRA pass on realistically-shaped,
agent-selected data move behavior?

It has no biology, no diary, no ledger, and no peers, by design. If it starts
accreting those it becomes an untested second implementation of the harness and
the Phase B tests get written against the wrong thing. Delete it once A1b
reports.

What it does carry, because A1b needs the training data to have the right shape:
  - the same action schema and EXPECT ritual as the real loop
  - the same prompt split at the cache boundary
  - the same parse-failure classification
  - expectation match/mismatch logging (the pleasure signal), never shown

Usage:
    python scripts/a0_bare_loop.py --steps 320 --backend stub --out runs/a0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.agent.parse import Action, ParseFailure, parse_action  # noqa: E402
from seahaven.agent.schemas import ACTION_SCHEMA, ACTION_SCHEMA_NAME  # noqa: E402
from seahaven.backend.stub_backend import StubBackend, wander_script  # noqa: E402
from seahaven.backend.types import (  # noqa: E402
    GenRequest,
    JsonSchema,
    Message,
    PromptParts,
)
from seahaven.world.loader import open_world  # noqa: E402

WORLD = Path(__file__).resolve().parents[1] / "worlds" / "world_v0" / "world_v0.z8"

# Frozen prefix. Deliberately spare: world_v0 has no story artifact and no diary,
# so the only stable content is the framing. Note there is no meta-vocabulary
# here — no "experiment", "run", "episode", "task".
SYSTEM = """You are alone in a decommissioned light-and-weather station on a shingle spit.

Each time you act, first say briefly what you expect will happen, then say what \
you do. Say what you do as a short command, the way you would speak it aloud.

Nobody is coming. There is nothing you have to achieve."""


def build_prompt(observation: str, last: str | None) -> PromptParts:
    frozen = (Message(role="system", content=SYSTEM),)
    lines = []
    if last:
        lines.append(f"Last time, you {last}.")
    lines.append(observation)
    return PromptParts(
        frozen=frozen,
        variable=(Message(role="user", content="\n\n".join(lines)),),
    )


def expectation_met(expect: str, observation: str) -> bool | None:
    """Crude lexical overlap. Adequate for A0, not for the real harness.

    The real pleasure mechanic needs a proper match test; here it only has to
    populate the field so the training data has the right shape.
    """
    if not expect.strip():
        return None
    words = {w for w in expect.lower().split() if len(w) > 3}
    if not words:
        return None
    return bool(words & set(observation.lower().split()))


def run(steps: int, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / "trajectory.jsonl"

    backend = StubBackend(default=wander_script(seed))
    world = open_world(WORLD)
    observation, hidden = world.reset()

    stats: Counter[str] = Counter()
    last_command: str | None = None
    rooms_seen: set[str] = set()

    with (
        backend.cache_scope((Message(role="system", content=SYSTEM),)) as scope,
        trajectory_path.open("w") as fh,
    ):
        for step in range(steps):
            parts = build_prompt(observation.text, last_command)
            result = scope.generate(
                GenRequest(
                    parts=parts,
                    constraint=JsonSchema(ACTION_SCHEMA, ACTION_SCHEMA_NAME),
                    sampling_seed=seed * 100_000 + step,
                    max_tokens=160,
                    stop=("\n-=", "\n>"),
                )
            )

            parsed = parse_action(result.text)
            if isinstance(parsed, ParseFailure):
                stats[f"parse_fail:{parsed.kind.value}"] += 1
                action: Action = parsed.fallback()
            else:
                action = parsed
                stats["parse_ok"] += 1

            prev_room = observation.room
            if prev_room:
                rooms_seen.add(prev_room)
            observation, hidden = world.step(action.command)
            stats[f"verb:{action.kind}"] += 1
            if observation.room != prev_room:
                stats["moved"] += 1

            fh.write(
                json.dumps(
                    {
                        "step": step,
                        "room": prev_room,
                        "prompt": [
                            {"role": m.role, "content": m.content}
                            for m in parts.all_messages()
                        ],
                        "completion": result.text,
                        "expect": action.expect,
                        "command": action.command,
                        "verb": action.kind,
                        "observation": observation.text,
                        "expectation_met": expectation_met(
                            action.expect, observation.text
                        ),
                        "parse_ok": not isinstance(parsed, ParseFailure),
                        "gen": result.digest(),
                    }
                )
                + "\n"
            )
            fh.flush()
            last_command = action.command

    world.close()

    summary = {
        "steps": steps,
        "seed": seed,
        "world": WORLD.name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "rooms_visited": sorted(rooms_seen),
        "stats": dict(sorted(stats.items())),
        "parse_failure_rate": round(
            sum(v for k, v in stats.items() if k.startswith("parse_fail")) / steps, 4
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("runs/a0"))
    ap.add_argument("--backend", choices=["stub"], default="stub",
                    help="vllm backend lands with the GPU work")
    args = ap.parse_args()

    summary = run(args.steps, args.out, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
