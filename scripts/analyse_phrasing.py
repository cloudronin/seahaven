"""V-P — does the model ordering survive a change in how the constraint is phrased?

The existential gate. Fidelity died on exactly this axis: its narration-register
sweep put worst-pair Spearman at 0.607 and V3 failed. Adherence has survived
measurement-side robustness (0.893 across filter definitions) and a change of
world (0.864), but never a change in the *stimulus*.

**G-P: worst-pair Spearman ≥ 0.80 on action-level adherence, pooled over both
worlds.** Fail and world_v3 is never authored, Stage 1 and Stage 2 never run,
and the salvage question is a new spec rather than a patch.

**P1 vs P5 is reported separately regardless of the gate.** P5 shows the
vocabulary and says nothing about it being exhaustive. If P5 is
indistinguishable from P1, the declaration does no work and the construct is
list-imitation rather than constraint adherence — a finding, not a failure.

Rates are POOLED over commands, never a mean of per-episode ratios: TRAP 30
moved a cross-world correlation from 0.964 to 0.679 on identical data by getting
that wrong.
"""

from __future__ import annotations

import glob
import itertools
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyse_v2 import spearman  # noqa: E402

from seahaven.fidelity.adherence import classify  # noqa: E402

PHRASINGS = ["p1", "p2", "p3", "p4", "p5"]


def load():
    """(lab, phrasing) -> pooled counts, plus per-seed cells for bootstrapping."""
    pooled = defaultdict(lambda: [0, 0])          # [violations, commands]
    per_seed = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in sorted(glob.glob("results/vp_*.json")):
        parts = Path(f).stem.split("_")
        lab, ph = parts[1], parts[2]
        seed = parts[-1]
        d = json.loads(Path(f).read_text())
        for run in d["runs"]:
            for c in run.get("commands", []):
                bad = classify(c["command"]) == "violation"
                pooled[(lab, ph)][0] += bad
                pooled[(lab, ph)][1] += 1
                per_seed[(lab, ph)][seed][0] += bad
                per_seed[(lab, ph)][seed][1] += 1
    return pooled, per_seed


def adherence(counts) -> float:
    bad, n = counts
    return 100.0 * (1 - bad / n) if n else float("nan")


def main() -> int:
    pooled, per_seed = load()
    labs = sorted({l for l, _ in pooled})
    if not labs:
        print("no results/vp_*.json", file=sys.stderr)
        return 2

    print(f"V-P — {len(labs)} models x {len(PHRASINGS)} phrasings, "
          f"action-level adherence pooled over both worlds\n")
    A = {(l, p): adherence(pooled[(l, p)]) for l in labs for p in PHRASINGS}

    print(f"  {'model':<11}" + "".join(f"{p:>9}" for p in PHRASINGS)
          + f"{'range':>9}{'n cmds':>9}")
    for l in labs:
        vals = [A[(l, p)] for p in PHRASINGS]
        n = sum(pooled[(l, p)][1] for p in PHRASINGS)
        print(f"  {l:<11}" + "".join(f"{v:>9.2f}" for v in vals)
              + f"{max(vals)-min(vals):>9.2f}{n:>9}")

    print("\n  ordering per phrasing (best first):")
    order = {p: [l for _, l in sorted(((A[(l, p)], l) for l in labs), reverse=True)]
             for p in PHRASINGS}
    for p in PHRASINGS:
        print(f"    {p}  {order[p]}")

    print("\n  pairwise Spearman (all 10 pairs):")
    worst, worst_pair = 2.0, None
    matrix = {}
    for a, b in itertools.combinations(PHRASINGS, 2):
        r = spearman([A[(l, a)] for l in labs], [A[(l, b)] for l in labs])
        matrix[f"{a}|{b}"] = r
        if r < worst:
            worst, worst_pair = r, (a, b)
        print(f"    {a} vs {b}   rho = {r:+.3f}")

    print(f"\n  worst pair: {worst_pair[0]} vs {worst_pair[1]}  rho = {worst:.3f}")
    passed = worst >= 0.80
    print(f"  G-P: {'PASS' if passed else 'FAIL'}  (gate 0.80)")

    # Bootstrap the worst pair over seeds — a point estimate on 7 models
    # saturates easily, and the gate should not rest on one draw.
    rng = random.Random(19)
    boot = []
    for _ in range(4000):
        aa, bb = [], []
        for l in labs:
            for tgt, ph in ((aa, worst_pair[0]), (bb, worst_pair[1])):
                cells = list(per_seed[(l, ph)].values())
                pick = [cells[rng.randrange(len(cells))] for _ in cells]
                tgt.append(adherence([sum(x[0] for x in pick),
                                      sum(x[1] for x in pick)]))
        r = spearman(aa, bb)
        if r == r:
            boot.append(r)
    boot.sort()
    lo, hi = boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]
    print(f"  worst-pair bootstrapped over seeds: {boot[len(boot)//2]:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")

    # The median-behaving phrasing becomes the frozen choice on a pass.
    consensus = [l for _, l in sorted(
        ((st.mean(A[(l, p)] for p in PHRASINGS), l) for l in labs), reverse=True)]
    disp = {p: st.mean(abs(order[p].index(l) - consensus.index(l)) for l in labs)
            for p in PHRASINGS}
    median_ph = min(disp, key=disp.get)
    print("\n  mean absolute rank displacement from consensus:")
    for p in PHRASINGS:
        print(f"    {p}  {disp[p]:.3f}{'   <- median-behaving' if p == median_ph else ''}")

    print("\nP1 vs P5 — reported regardless of the gate")
    d = [A[(l, "p5")] - A[(l, "p1")] for l in labs]
    print(f"  {'model':<11}{'p1':>9}{'p5':>9}{'p5-p1':>9}")
    for l, x in zip(labs, d):
        print(f"  {l:<11}{A[(l,'p1')]:>9.2f}{A[(l,'p5')]:>9.2f}{x:>+9.2f}")
    pooled_p1 = adherence([sum(pooled[(l, "p1")][0] for l in labs),
                           sum(pooled[(l, "p1")][1] for l in labs)])
    pooled_p5 = adherence([sum(pooled[(l, "p5")][0] for l in labs),
                           sum(pooled[(l, "p5")][1] for l in labs)])
    gap = pooled_p5 - pooled_p1
    print(f"  pooled p1 {pooled_p1:.2f}  p5 {pooled_p5:.2f}  gap {gap:+.2f}")
    print(f"  VP-3 (p5 worst, gap beyond 2 points): "
          f"{'CONFIRMED' if gap < -2 else 'FALSIFIED'}")
    if abs(gap) < 2:
        print("  P5 ~ P1: the restriction sentence does no measurable work, and")
        print("  the construct is closer to list-imitation than to constraint")
        print("  adherence. A finding, not a failure — but it reframes the paper.")

    means = {p: st.mean(A[(l, p)] for l in labs) for p in PHRASINGS}
    lowest = min(means, key=means.get)
    print(f"\n  VP-2 (p4 terse lowest): "
          f"{'CONFIRMED' if lowest == 'p4' else f'FALSIFIED — {lowest} is lowest'}")
    print(f"  mean by phrasing: "
          f"{ {p: round(means[p], 2) for p in PHRASINGS} }")

    bottom2 = {p: order[p][-2:] for p in PHRASINGS}
    stable = len({frozenset(v) for v in bottom2.values()}) == 1
    print(f"  VP-4 (bottom of table phrasing-stable): "
          f"{'CONFIRMED' if stable else 'FALSIFIED'}  {bottom2}")

    Path("results/vp_phrasing.json").write_text(json.dumps({
        "adherence": {f"{l}|{p}": A[(l, p)] for l in labs for p in PHRASINGS},
        "orderings": order, "pairwise": matrix,
        "worst_pair": list(worst_pair), "worst_rho": worst,
        "worst_boot": {"median": boot[len(boot)//2], "ci": [lo, hi]},
        "gate_pass": bool(passed), "median_phrasing": median_ph,
        "displacement": disp, "p5_minus_p1_pooled": gap,
    }, indent=2) + "\n")
    print("\nwrote results/vp_phrasing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
