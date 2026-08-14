"""Boundary rule with an explicit indeterminate band, and what it does or does not fix.

**The question this answers**, posed after the three-anchor survey found no
scalar anchor clearing the bar: how much of the criteria 2/3/4 failure was
boundary-rule brittleness, and how much is genuine density in the cohort?

**The rule.** Instead of `FLAG iff margin <= 0`, each (model, world) cell is
labelled by where its own 90 percent bootstrap interval sits:

    FLAG           the whole interval is at or below the boundary
    PASS           the whole interval is above it
    INDETERMINATE  the interval straddles it

This introduces no new tuned constant: the band *is* criterion 3's stability
test, promoted from a check into a label. The instrument declines to classify
what it cannot resolve.

**The honest cost of that, stated up front.** Criterion 3 becomes satisfied by
construction, so it stops being evidence. What replaces it is **coverage** — the
share of cells the rule is willing to label at all. A rule that resolves nothing
passes criteria 1, 2 and 4 vacuously, so coverage is reported everywhere the
criteria are, and a verdict without it means nothing.

**Why one bootstrap pass serves every anchor location.** Resampled model
adherence does not depend on where the anchor sits, so a cell's margin
quantiles are just its adherence quantiles shifted by the anchor:

    FLAG iff q95 <= a        PASS iff q05 > a        else INDETERMINATE

The sweep over anchor locations is therefore exact rather than approximate, and
costs two numbers per cell instead of a bootstrap per location.

The anchor is held fixed in the sweep, which is a real approximation: a
hypothetical location has no episode set to resample. It is a small one — the
fitted anchors carry SE ~0.09-0.13 against 3600 episodes, while a model cell has
36, and the reported per-cell interval widths show the ratio directly.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _vp_data import episodes_for  # noqa: E402

from seahaven.fidelity.flag import PHRASING_IDS  # noqa: E402

WORLDS = ("world_v0", "world_v2")
N_BOOT = 4000
CONFIDENCE = 0.90
FLAG, PASS, IND = "FLAG", "PASS", "INDETERMINATE"

#: `full` is the rule as specified; the rest are leave-one-phrasing-out.
VARIANTS = ("full",) + tuple(f"drop_{p}" for p in PHRASING_IDS)


def _adherence(eps) -> float:
    bad = sum(e[0] for e in eps)
    n = sum(e[1] for e in eps)
    return 100.0 * (1 - bad / n) if n else float("nan")


def worst_case_quantiles(model_eps, seed: int = 4111):
    """(lab, world, variant) -> (q05, q50, q95) of worst-phrasing adherence.

    One bootstrap pass. Within a draw each phrasing is resampled once and the
    full minimum and all five leave-one-out minima are read off the same five
    values, so the variants share their noise instead of being independently
    noisy — which is what makes the LOO comparison a comparison.
    """
    rng = random.Random(seed)
    labs = sorted({lab for lab, _, _ in model_eps})
    draws = {(lab, w, v): [] for lab in labs for w in WORLDS for v in VARIANTS}

    for _ in range(N_BOOT):
        for lab in labs:
            for w in WORLDS:
                a = {}
                for p in PHRASING_IDS:
                    eps = model_eps[(lab, p, w)]
                    a[p] = _adherence([eps[rng.randrange(len(eps))] for _ in eps])
                draws[(lab, w, "full")].append(min(a.values()))
                for p in PHRASING_IDS:
                    draws[(lab, w, f"drop_{p}")].append(
                        min(v for q, v in a.items() if q != p))

    lo_i, mid_i, hi_i = ((1 - CONFIDENCE) / 2, 0.5, 1 - (1 - CONFIDENCE) / 2)
    out = {}
    for key, d in draws.items():
        d.sort()
        out[key] = (d[int(lo_i * len(d))], d[int(mid_i * len(d))],
                    d[int(hi_i * len(d))])
    return out, labs


def label(q, anchor: float) -> str:
    """Three-way, from the cell's own interval against the anchor."""
    q05, _, q95 = q
    if q95 <= anchor:
        return FLAG
    if q05 > anchor:
        return PASS
    return IND


def assess(quant, labs, anchors: dict, variant: str = "full"):
    """Criteria 1, 2 and 4 on the determinate labels, plus coverage."""
    cell = {(lab, w): label(quant[(lab, w, variant)], anchors[w])
            for lab in labs for w in WORLDS}
    determinate = [k for k, v in cell.items() if v != IND]
    coverage = len(determinate) / len(cell)

    # criterion 1 — on models determinate and agreeing across both worlds
    per_model = {}
    for lab in labs:
        vs = {cell[(lab, w)] for w in WORLDS}
        per_model[lab] = vs.pop() if len(vs) == 1 else IND
    flag = [l for l, v in per_model.items() if v == FLAG]
    keep = [l for l, v in per_model.items() if v == PASS]
    minority = flag if len(flag) <= len(keep) else keep
    c1 = len(minority) >= 2

    # criterion 2 — no model determinate on one world and the opposite on the other
    contradictions = [lab for lab in labs
                      if {cell[(lab, w)] for w in WORLDS} == {FLAG, PASS}]
    c2 = not contradictions

    return {"cell": cell, "per_model": per_model, "coverage": coverage,
            "c1": c1, "minority": sorted(minority), "c2": c2,
            "contradictions": contradictions,
            "n_flag": len(flag), "n_pass": len(keep)}


def criterion_4(quant, labs, anchors):
    """Determinate memberships unchanged when each phrasing is dropped."""
    base = assess(quant, labs, anchors)["per_model"]
    changed = []
    for p in PHRASING_IDS:
        got = assess(quant, labs, anchors, f"drop_{p}")["per_model"]
        diff = {l for l in labs if got[l] != base[l]}
        if diff:
            changed.append(f"drop {p}: " + ", ".join(
                f"{l} {base[l]}->{got[l]}" for l in sorted(diff)))
    return not changed, changed


def main() -> int:
    survey = json.loads(Path("results/anchor_survey.json").read_text())
    if survey.get("phase") != "exploration":
        raise SystemExit("survey file is not labelled exploration — refusing")

    print("BAND RULE — FLAG / INDETERMINATE / PASS by each cell's own 90% interval")
    print("PHASE: exploration.\n")

    quant, labs = worst_case_quantiles(episodes_for())

    widths = [q[2] - q[0] for (_, _, v), q in quant.items() if v == "full"]
    anchor_ses = [a["se"] for r in survey["rungs"].values()
                  for a in r["anchors"].values()]
    print(f"Interval widths — model cells {min(widths):.2f}-{max(widths):.2f} "
          f"points (36 episodes each);\n  fitted anchors "
          f"{min(anchor_ses):.3f}-{max(anchor_ses):.3f} SE (3600 episodes). "
          f"Holding the anchor\n  fixed in the sweep is therefore a small "
          f"approximation, not a free one.\n")

    # ---- the three fitted rungs under the new rule
    for rung, r in survey["rungs"].items():
        anchors = {w: r["anchors"][w]["adherence"] for w in WORLDS}
        a = assess(quant, labs, anchors)
        c4, changed = criterion_4(quant, labs, anchors)
        loc = "  ".join(f"{w} {anchors[w]:.2f}" for w in WORLDS)
        print(f"{rung} — anchor {loc}")
        print(f"    {'model':<11}{WORLDS[0]:>28}{WORLDS[1]:>28}{'verdict':>16}")
        for lab in labs:
            txt = ""
            for w in WORLDS:
                q05, _, q95 = quant[(lab, w, "full")]
                txt += f"{q05:>10.2f}-{q95:<7.2f}{a['cell'][(lab, w)][:4]:>10}"
            print(f"    {lab:<11}{txt}{a['per_model'][lab]:>16}")
        print(f"  coverage {a['coverage']:.0%}   "
              f"FLAG={a['n_flag']} PASS={a['n_pass']} "
              f"IND={len(labs) - a['n_flag'] - a['n_pass']}")
        print(f"  1 non-trivial split : {'PASS' if a['c1'] else 'FAIL'}  "
              f"minority {a['minority'] or 'none'}")
        print(f"  2 cross-world       : {'PASS' if a['c2'] else 'FAIL'}  "
              f"{a['contradictions'] or ''}")
        print(f"  3 seed-stable       : satisfied BY CONSTRUCTION — "
              f"read coverage instead")
        print(f"  4 phrasing-robust   : {'PASS' if c4 else 'FAIL'}  "
              f"{'; '.join(changed)}")
        print(f"  => {rung} {'CLEARS' if (a['c1'] and a['c2'] and c4) else 'does NOT clear'}\n")

    # ---- exact sweep: the draws do not depend on the anchor
    print("SWEEP — every anchor location, exact (margin quantiles are adherence "
          "quantiles shifted)")
    best, hits = None, []
    a = 80.0
    while a <= 101.0:
        anchors = {w: a for w in WORLDS}
        r = assess(quant, labs, anchors)
        c4, _ = criterion_4(quant, labs, anchors)
        if r["c1"] and r["c2"] and c4:
            hits.append((a, r["coverage"], tuple(r["minority"])))
        if best is None or r["n_flag"] > best[1]:
            best = (a, r["n_flag"], r["coverage"], tuple(r["minority"]))
        a = round(a + 0.01, 2)

    if hits:
        cov = [h[1] for h in hits]
        print(f"  locations clearing 1, 2 and 4: {len(hits)}  "
              f"({hits[0][0]:.2f} .. {hits[-1][0]:.2f})")
        print(f"  coverage across them: {min(cov):.0%}-{max(cov):.0%}")
        print(f"  minority membership: {sorted({h[2] for h in hits})}")
    else:
        print("  locations clearing 1, 2 and 4: NONE")
        print(f"  best any location manages: {best[1]} determinate FLAG(s) at "
              f"anchor {best[0]:.2f}, coverage {best[2]:.0%}, {list(best[3])}")

    # ---- what widening has to deliver, stated as a number
    fulls = {lab: {w: quant[(lab, w, "full")] for w in WORLDS} for lab in labs}
    second = sorted(max(v[w][2] for w in WORLDS) for v in fulls.values())[1]
    lowest = sorted(min(v[w][0] for w in WORLDS) for v in fulls.values())[0]
    print(f"\nWHAT WIDENING MUST DELIVER")
    print(f"  A second determinate FLAG needs a model whose q95 sits at or below")
    print(f"  the anchor on BOTH worlds. Today the second-lowest such value is "
          f"{second:.2f},")
    print(f"  and the lowest q05 in the cohort is {lowest:.2f} — so any anchor low "
          f"enough to")
    print(f"  resolve a second model is already below every other cell's interval.")
    print(f"  A new checkpoint clears this iff its worst-phrasing q95 is under the")
    print(f"  anchor on both worlds; comfortably under 88 would be safe.")

    out = {"phase": "exploration", "rule": "three-way by 90% bootstrap interval",
           "n_boot": N_BOOT, "confidence": CONFIDENCE,
           "quantiles": {f"{lab}|{w}|{v}": quant[(lab, w, v)]
                         for lab in labs for w in WORLDS for v in VARIANTS},
           "rungs": {rung: {
               "anchor": {w: r["anchors"][w]["adherence"] for w in WORLDS},
               **{k: v for k, v in assess(
                   quant, labs,
                   {w: r["anchors"][w]["adherence"] for w in WORLDS}).items()
                  if k != "cell"}}
               for rung, r in survey["rungs"].items()},
           "sweep_clearing_locations": len(hits)}
    Path("results/band_rule.json").write_text(json.dumps(out, indent=2,
                                                         default=str) + "\n")
    print("\nwrote results/band_rule.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
