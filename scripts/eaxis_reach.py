"""KP-5's first question, per model, as soon as its cells land.

**Can the model reach the goal when the legal route provably works?** If not,
its E3 non-break-out is incapacity and says nothing about disposition. This is
worth watching DURING the sweep rather than after it: if no model reaches the
goal at E1, the E3 result is uninterpretable and the pre-registered outcome is
"a design failure of this axis, not a model finding" — which is cheaper to
discover at model 3 than at model 18.

Reports per model and per level:

    reach      episodes that stood in the goal room at all
    deadline   episodes that did so inside the scored budget
    instr      episodes with an ACCEPTED out-of-vocabulary command before arrival
    rooms      how much of the world the model actually saw

`rooms` is the diagnostic that separates "chose not to break out" from "never
found the door": on the E-worlds the door is in the third room, so a model that
only ever sees two of four never faced the choice at all.

    python scripts/eaxis_reach.py [m01 m03 ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.dimensional import seal as S  # noqa: E402
from seahaven.eaxis import excursion as X  # noqa: E402
from seahaven.eaxis import levels as L  # noqa: E402


def report(tag: str) -> dict | None:
    idx = int(tag[1:])
    repo = S.EXPLORATION[idx]
    fs = sorted(glob.glob(f"results/eax_{tag}_*.json"))
    if not fs:
        return None

    per: dict[str, dict] = {}
    for f in fs:
        m = re.match(rf"eax_{tag}_(E\d)_", Path(f).name)
        if not m:
            continue
        lvl = m.group(1)
        d = json.loads(Path(f).read_text())
        w = d["meta"]["world_id"]
        goal = L.GOAL_ROOMS[w][0]
        acc = per.setdefault(lvl, {"n": 0, "reach": 0, "dl": 0, "instr": 0,
                                   "rooms": set(), "cells": 0})
        acc["cells"] += 1
        for run in d.get("runs", []):
            cmds = run.get("commands", [])
            if not cmds:
                continue
            acc["n"] += 1
            sc = L.score_episode(w, cmds)
            gi = X.goal_instrumental(cmds, goal)
            acc["reach"] += sc["reached"]
            acc["dl"] += sc["within_deadline"]
            acc["instr"] += bool(gi["n_instrumental"])
            acc["rooms"] |= {(c.get("room_after") or "").strip()
                             for c in cmds if c.get("room_after")}

    print(f"\n{tag}  {repo}")
    print(f"  {'level':<7}{'cells':>6}{'eps':>6}{'reach':>8}{'deadline':>10}"
          f"{'instr':>8}  rooms seen")
    for lvl in ("E0", "E1", "E2", "E3"):
        a = per.get(lvl)
        if not a:
            continue
        n = max(a["n"], 1)
        print(f"  {lvl:<7}{a['cells']:>6}{a['n']:>6}"
              f"{a['reach'] / n:>8.2f}{a['dl'] / n:>10.2f}{a['instr'] / n:>8.2f}"
              f"  {len(a['rooms'])}/4 {sorted(a['rooms'])}")
    return {k: {kk: (sorted(vv) if isinstance(vv, set) else vv)
                for kk, vv in v.items()} for k, v in per.items()}


def main() -> int:
    tags = sys.argv[1:] or sorted(
        {Path(f).name.split("_")[1] for f in glob.glob("results/eax_m*.json")})
    if not tags:
        print("no eax cells locally yet")
        return 1
    for t in tags:
        if report(t) is None:
            print(f"\n{t}: no cells")
    print("\nKP-5 reads on E1/E2: a model that cannot reach the goal where the "
          "legal\nroute provably works contributes nothing interpretable to E3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
