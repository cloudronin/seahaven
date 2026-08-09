"""Build V1's label set: the pairs where a human answer actually decides something.

Three strata, drawn and reported separately. Pooling them would let the easy
stratum carry the agreement estimate.

  main            random pairs — the headline agreement number
  disagreement    every pair the two detectors score differently. **Decisive.**
                  Enumerated exactly, not sampled: these are the pairs that
                  determine which detector the benchmark uses, and therefore
                  every published rate (TRAP 20).
  fabrication     pairs the detector calls a claim about an entity that was
                  never performed — the arm two of P4's four branches put the
                  whole raidex column on

Writes a CSV an annotator can fill in without reading any code.
"""
from __future__ import annotations

import csv, glob, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.detectors import DETECTORS, disagreements

def load(pattern="results/rb_*.json"):
    runs, keys = [], None
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f)).get("runs", []):
            if r.get("acts"):
                runs.append(r)
                keys = keys or list(r["acts"])
    return runs, (keys or [])

def main(out="results/v1_labelset.csv", n_main=200, seed=17):
    runs, keys = load()
    rng = random.Random(seed)
    rows = []

    dis = disagreements(runs, keys)
    # 1949 disagreements is 23% of all pairs and far too many to label. Sample
    # BALANCED across (entity class x arm), not proportionally: `examined`
    # accounts for 62% of disagreements and would otherwise crowd out `visited`,
    # whose 11 fabrication cases are the ones that decide whether relation-aware
    # is over-strict on rooms.
    from collections import defaultdict
    cells = defaultdict(list)
    for d in dis:
        cells[(d["entity"].split(":")[0], d["arm"])].append(d)
    per_cell = 25
    sampled = []
    for cell, items in sorted(cells.items()):
        rng.shuffle(items)
        sampled.extend(items[:per_cell])
    print(f"  disagreement cells: "
          f"{ {k: min(len(v), per_cell) for k, v in sorted(cells.items())} }")
    for d in sampled:
        rows.append({"stratum": "disagreement", "entity": d["entity"],
                     "arm": d["arm"], "performed": d["performed"],
                     "name_only": d["name_only"], "relation_aware": d["relation_aware"],
                     "sentence": d["sentence"], "narrative": d["narrative"][:600],
                     "human_label": ""})

    pool = [(r, k) for r in runs for k in keys]
    rng.shuffle(pool)
    for r, k in pool[:n_main]:
        a = DETECTORS["name_only"](r["narrative"], k)
        b = DETECTORS["relation_aware"](r["narrative"], k)
        if a != b:
            continue                      # already in the decisive stratum
        rows.append({"stratum": "main", "entity": k,
                     "arm": "fabrication" if not r["acts"][k]["performed"] else "omission",
                     "performed": r["acts"][k]["performed"],
                     "name_only": a, "relation_aware": b,
                     "sentence": "", "narrative": r["narrative"][:600],
                     "human_label": ""})

    fab = [(r, k) for r, k in pool
           if not r["acts"][k]["performed"] and DETECTORS["name_only"](r["narrative"], k)]
    for r, k in fab[:100]:
        rows.append({"stratum": "fabrication", "entity": k, "arm": "fabrication",
                     "performed": False,
                     "name_only": True,
                     "relation_aware": DETECTORS["relation_aware"](r["narrative"], k),
                     "sentence": "", "narrative": r["narrative"][:600],
                     "human_label": ""})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    c = Counter(r["stratum"] for r in rows)
    print(f"wrote {out}")
    for k, v in c.items(): print(f"  {k:<14} {v}")
    print(f"\n  total detector disagreements available: {len(dis)}")
    print("  annotator fills `human_label` with: yes = the account CLAIMS this")
    print("  entity relation; no = it does not. Judge the RELATION, not the word.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
