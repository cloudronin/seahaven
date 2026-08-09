"""V1 label set, rebuilt on the corrected runs and across both worlds.

**Why a rebuild rather than reusing the old set.** The first V1 sample was drawn
from runs carrying TRAP 26, where `took:coil of rope` and `took:brass key` were
False in every run regardless of what the agent held. The judge's labels were
unaffected — the judge is asked what the *narrative says*, not what happened —
but the **stratum assignment** was not. The fabrication stratum is defined as
"the detector fires and the entity was not performed", so genuine takes of the
rope and the key were routed into it: **72% of that stratum was rope/key rows.**

That stratum is the one two of pre-registration P4's four branches put the whole
raidex column on, so it has to be drawn again from truth that is actually true.

**Both worlds now.** The old set was world_v0 only. A detector validated on one
world says nothing about the other, and V2 has just shown the score is at least
rank-stable across them, so the detector should be checked against both.

**Entity keys are per-world**, since world_v0 scores 17 and world_v2 scores 20,
so the disagreement enumeration runs per world and the strata are pooled only
afterwards — never across the stratum boundary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.detectors import DETECTORS  # noqa: E402


def load(pattern: str):
    """Runs grouped by world, each with the entity keys that world scores."""
    by_world = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        world = d.get("meta", {}).get("world_id", "?")
        for r in d.get("runs", []):
            if r.get("acts"):
                by_world[world].append(r)
    return by_world


def row(r, k, world, stratum):
    a = DETECTORS["name_only"](r["narrative"], k)
    b = DETECTORS["relation_aware"](r["narrative"], k)
    return {"stratum": stratum, "world": world, "entity": k,
            "arm": "fabrication" if not r["acts"][k]["performed"] else "omission",
            "performed": r["acts"][k]["performed"],
            "name_only": a, "relation_aware": b,
            "sentence": "", "narrative": r["narrative"], "human_label": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/x_*.json")
    ap.add_argument("--out", default="results/v1b_labelset.csv")
    ap.add_argument("--per-cell", type=int, default=30,
                    help="disagreement items per (world x relation x arm) cell")
    ap.add_argument("--main", type=int, default=260)
    ap.add_argument("--fab", type=int, default=200)
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    by_world = load(args.pattern)
    if not by_world:
        print(f"no runs matched {args.pattern}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rows = []

    for world, runs in sorted(by_world.items()):
        keys = list(runs[0]["acts"])
        pool = [(r, k) for r in runs for k in keys]
        rng.shuffle(pool)

        # 1. disagreement — enumerated then sampled balanced across
        #    (relation x arm), because `examined` dominates by volume and would
        #    otherwise crowd out the rarer cells that decide the detector.
        cells = defaultdict(list)
        for r, k in pool:
            if DETECTORS["name_only"](r["narrative"], k) != \
               DETECTORS["relation_aware"](r["narrative"], k):
                arm = "fabrication" if not r["acts"][k]["performed"] else "omission"
                cells[(k.split(":")[0], arm)].append((r, k))
        for cell, items in sorted(cells.items()):
            for r, k in items[:args.per_cell]:
                rows.append(row(r, k, world, "disagreement"))
        print(f"  {world} disagreement cells: "
              f"{ {f'{a}/{b}': min(len(v), args.per_cell) for (a, b), v in sorted(cells.items())} }")

        # 2. main — random pairs where the detectors already agree
        n = 0
        for r, k in pool:
            if n >= args.main // len(by_world):
                break
            if DETECTORS["name_only"](r["narrative"], k) == \
               DETECTORS["relation_aware"](r["narrative"], k):
                rows.append(row(r, k, world, "main"))
                n += 1

        # 3. fabrication — detector fires on an entity that was NOT performed.
        #    Drawn on the corrected truth, which is the whole point of the
        #    rebuild.
        n = 0
        for r, k in pool:
            if n >= args.fab // len(by_world):
                break
            if not r["acts"][k]["performed"] and DETECTORS["name_only"](r["narrative"], k):
                rows.append(row(r, k, world, "fabrication"))
                n += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}: {len(rows)} rows")
    print(f"  by stratum: {dict(Counter(r['stratum'] for r in rows))}")
    print(f"  by world:   {dict(Counter(r['world'] for r in rows))}")
    ropekey = sum(1 for r in rows if r["entity"].split(":", 1)[1] in
                  ("coil of rope", "brass key"))
    fab = [r for r in rows if r["stratum"] == "fabrication"]
    rk_fab = sum(1 for r in fab if r["entity"].split(":", 1)[1] in
                 ("coil of rope", "brass key"))
    print(f"  rope/key share of fabrication stratum: {rk_fab}/{len(fab)} "
          f"({100 * rk_fab / max(1, len(fab)):.0f}%)  — was 72% before the fix")
    print(f"  rope/key share overall: {100 * ropekey / len(rows):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
