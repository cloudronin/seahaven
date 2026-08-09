"""Which detector does the judge support — and does the answer differ by relation?

The pilot split 9/10 on disagreement items, which says neither detector is right
globally. That is not a stalemate: `took:` and `visited:` are different problems.

- an object name in a narrative often means the object *existed*, not that it was
  taken — so name-only over-claims
- a room name in a narrative usually does mean the writer was there — so
  relation-aware under-claims

If that holds, the resolution is a **per-relation detector**, not a global choice
between two wrong ones.

Reports agreement per stratum and per relation, never pooled, and reports the
judge's control accuracy and self-consistency alongside — an unvalidated judge
adjudicates nothing.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Chance-corrected agreement. Raw agreement flatters a lopsided sample."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main(path="results/v1_adjudicated.json") -> int:
    d = json.loads(Path(path).read_text())
    if not d.get("usable", True):
        print("judge failed its controls; labels not used")
        return 3

    items = d["items"]
    stable = [i for i in items if i["stable"]]
    print(f"judge {d['model']}   control accuracy {d['control_accuracy']:.2f}   "
          f"self-consistent {len(stable)}/{len(items)}")
    if len(stable) < len(items):
        print("  unstable items are EXCLUDED — a label the judge contradicts is "
              "not a label")

    print("\nAGREEMENT WITH THE JUDGE, by stratum (never pooled)")
    print(f"  {'stratum':<15}{'n':>5}{'name_only':>12}{'relation_aware':>16}")
    for s in ("main", "disagreement", "fabrication"):
        g = [i for i in stable if i["stratum"] == s]
        if not g:
            continue
        j = [i["judge"] for i in g]
        an = sum(i["judge"] == i["name_only"] for i in g) / len(g)
        ar = sum(i["judge"] == i["relation_aware"] for i in g) / len(g)
        print(f"  {s:<15}{len(g):>5}{an:>12.2f}{ar:>16.2f}")
        kn = cohen_kappa(j, [i["name_only"] for i in g])
        kr = cohen_kappa(j, [i["relation_aware"] for i in g])
        print(f"  {'':<15}{'kappa':>5}{str(kn):>12}{str(kr):>16}")

    print("\nTHE DECIDING BREAKDOWN — disagreement items by relation")
    print(f"  {'relation':<12}{'n':>5}{'judge=name_only':>18}{'judge=rel_aware':>18}   better")
    dis = [i for i in stable if i["name_only"] != i["relation_aware"]]
    by = defaultdict(list)
    for i in dis:
        by[i["entity"].split(":", 1)[0]].append(i)
    winners = {}
    for rel, g in sorted(by.items()):
        an = sum(i["judge"] == i["name_only"] for i in g)
        ar = sum(i["judge"] == i["relation_aware"] for i in g)
        winners[rel] = "name_only" if an > ar else "relation_aware"
        print(f"  {rel:<12}{len(g):>5}{an:>18}{ar:>18}   {winners[rel]}")

    print("\n  Per-relation recommendation (what the judge actually supports):")
    for rel, w in sorted(winners.items()):
        print(f"    {rel:<12} -> {w}")
    if len(set(winners.values())) > 1:
        print("\n  The detectors win different relations. A single global choice"
              "\n  would be wrong for at least one entity class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "results/v1_adjudicated.json"))
