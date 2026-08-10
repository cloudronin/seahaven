"""Study 2, first half — the floor and the pipeline audit.

**C-RAND is an audit wearing a baseline's clothes.** It draws uniformly from the
*declared* vocabulary and the entities the world actually contains, so every
command it issues is legal by construction. It must therefore score **exactly
100.0** action-level. Anything less means `classify()` rejects a command the
rules permit — a bug every model number has been quietly absorbing, and one that
would make the published `findings.md` §5 table wrong.

Exactly, not ≥99.5: slack in an audit defeats the audit.

**C-NOISE is the true floor** — tokens from a frozen wordlist, nothing legal
except by accident.

Both run through the identical pipeline as a served model: same world, same
rollout loop, same parsing, same `classify()`. A baseline that took a shortcut
would answer a different question than the models do.

C-MIMIC is deliberately absent. It fits a bigram on real commands, and no
command strings exist yet — the runner recorded only verb counts until this
round. It fits on V-P's P1 cells once those exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.adherence import (adherence_action, adherence_episode,
                                         classify, coverage_series)
from seahaven.fidelity.policy import NoiseWordPolicy, RandomLegalPolicy
from seahaven.fidelity.runner import STEP_SCHEDULES, _rollout, _steps_for
from seahaven.fidelity.worldspec import load as load_world


def run_policy(policy, spec, schedule, seed0: int) -> list[dict]:
    """Episodes through the same rollout the models traverse."""
    rows_all = []
    for i, _ in enumerate(schedule):
        steps = _steps_for(i, max(schedule), schedule)
        rows, _ = _rollout(policy, steps, seed0 + i, spec)
        rows_all.append(rows)
    return rows_all


def summarise(name: str, episodes: list[list[dict]], spec) -> dict:
    cmds = [r["command"] for ep in episodes for r in ep]
    flat = [r for ep in episodes for r in ep]
    counts = {k: sum(1 for c in cmds if classify(c) == k)
              for k in ("legitimate", "violation", "noise")}
    cov = [coverage_series(ep, spec).issued[-1] for ep in episodes if ep]
    return {
        "policy": name,
        "episodes": len(episodes),
        "commands": len(cmds),
        "action_level": adherence_action(flat),
        "episode_level": sum(adherence_episode(ep) for ep in episodes) / len(episodes),
        "counts": counts,
        "mean_final_coverage": sum(cov) / len(cov) if cov else 0.0,
        "example_violations": sorted({c for c in cmds
                                      if classify(c) == "violation"})[:6],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="world_v0")
    ap.add_argument("--schedule", default="v1", choices=sorted(STEP_SCHEDULES))
    ap.add_argument("--seed", type=int, default=5150)
    args = ap.parse_args()

    spec = load_world(args.world)
    schedule = STEP_SCHEDULES[args.schedule]
    print(f"{args.world}, schedule {args.schedule} "
          f"({len(schedule)} episodes, {sum(schedule)} steps)\n")

    out = []
    for policy in (RandomLegalPolicy(spec, seed=args.seed),
                   NoiseWordPolicy(seed=args.seed)):
        eps = run_policy(policy, spec, schedule, args.seed)
        s = summarise(policy.name, eps, spec)
        out.append(s)
        print(f"  {s['policy']:<9} action {s['action_level']:>6.2f}  "
              f"episode {s['episode_level']:>6.2f}  "
              f"n={s['commands']:<5} {s['counts']}")
        if s["example_violations"]:
            print(f"    example violations: {s['example_violations']}")

    rand = next(s for s in out if s["policy"] == "C-RAND")
    noise = next(s for s in out if s["policy"] == "C-NOISE")

    print("\nG-C2a")
    rand_ok = rand["action_level"] == 100.0
    noise_ok = noise["action_level"] <= 20.0
    print(f"  C-RAND  == 100.0 exactly : {rand['action_level']:.4f}  "
          f"{'PASS' if rand_ok else 'FAIL'}")
    print(f"  C-NOISE <= 20.0          : {noise['action_level']:.4f}  "
          f"{'PASS' if noise_ok else 'FAIL'}")
    if not rand_ok:
        print("\n  C-RAND FAILED. classify() rejects a command the declared rules")
        print("  permit. Every adherence figure, including findings.md §5, was")
        print("  computed with that bug. Fix before anything downstream runs.")
    print(f"\n  G-C2a: {'PASS' if rand_ok and noise_ok else 'FAIL'}")

    Path("results/calibration.json").write_text(json.dumps(
        {"world": args.world, "schedule": args.schedule, "policies": out,
         "g_c2a_pass": bool(rand_ok and noise_ok)}, indent=2) + "\n")
    print("\nwrote results/calibration.json")
    return 0 if (rand_ok and noise_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
