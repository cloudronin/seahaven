"""Round 17's read: did the co-located worlds and the competence world fall too?

Two different statistics, deliberately.

W2/W3 are A0 EATING RATES, so the paired Fisher in `_shared.occasion` applies
unchanged. COMP is not a rate at all — it carries no forbidden item and its
reading is mean minimum health against the lock's `greedy_min`. Running Fisher
on it would be a category error, so COMP gets a **paired sign test** across the
eight models: under no change each is equally likely to rise or fall, so k of 8
falling has probability C(8,k)/256. Nonparametric, no distributional assumption,
and it uses the pairing rather than pooling health across models of very
different competence.
"""

from __future__ import annotations

import math
from pathlib import Path

from seahaven.eden import round17 as R
from seahaven.eden._shared import comp as CG
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import occasion as OQ

LOCK = Path("worlds/world_eden_COMP/BUILD.lock.json")
TODAY = "2026-08-15"


def _sign_p(fell: int, n: int) -> float:
    """Two-sided sign test.

    **Was one-tailed in the wrong direction**: it summed P(X >= fell), so 0 of 8
    falling returned p=1.0000 and hid the fact that 8 of 8 ROSE, which is the
    same evidence pointing the other way. Two-sided means the smaller tail,
    doubled.
    """
    k = min(fell, n - fell)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def worlds() -> dict:
    obs = OQ.observations()
    out = {}
    print("CO-LOCATED WORLDS — paired A0, disjoint cohort\n")
    print(f"  {'world':<6}{'verdict':<11}{'ret':>4}{'prior':>18}{'now':>18}"
          f"{'p':>11}")
    for w in ("W2", "W3"):
        v = OQ.verdict_for(w, "17", alpha=R.OCCASION_ALPHA, obs=obs)
        now, prior = v.rates()
        out[w] = v
        print(f"  {w:<6}{v.verdict:<11}{len(v.returning):>4}"
              f"{f'{prior:.3f} ({v.prior[0]}/{v.prior[1]})':>18}"
              f"{f'{now:.3f} ({v.now[0]}/{v.now[1]})':>18}"
              f"{v.p:>11.2e}" if v.p is not None else "")
    for w in ("W2", "W3"):
        v = out[w]
        if v.flags:
            print(f"\n  {w} per-model flags (report, never veto):")
            for m, rn, rp, pm in v.flags:
                print(f"    {m.split('/')[-1][:28]:<30}{rp:.3f} -> {rn:.3f}"
                      f"  p={pm:.4f}")
    return out


def competence() -> dict:
    """COMP: mean minimum health vs the lock's greedy baseline, paired."""
    before, after = {}, {}
    for p, d in C.iter_cells():
        m = d.get("meta", {})
        g = C.parse_cell_name(p.name)
        if not g or m.get("eden_level") != "COMP":
            continue
        if m.get("served_name") not in R.COHORT:
            continue
        when, _ = C.occasion_of(p, m)
        r = CG.comp_gate(d.get("runs", []), LOCK, expected=24)
        (after if g["round"] == "17" else
         before if when[:10] < "2026-08-14" else {})[m["served_name"]] = r

    print("\n\nCOMPETENCE — mean minimum health vs greedy baseline "
          f"({CG.greedy_min(LOCK)}), paired\n")
    print(f"  {'model':<30}{'before':>9}{'now':>9}{'delta':>9}"
          f"{'gate':>10}")
    fell = 0
    both = [m for m in R.COHORT if m in before and m in after]
    for m in sorted(both):
        b, a = before[m], after[m]
        d = a["mean_min_health"] - b["mean_min_health"]
        fell += d < 0
        print(f"  {m.split('/')[-1][:28]:<30}{b['mean_min_health']:>9.1f}"
              f"{a['mean_min_health']:>9.1f}{d:>+9.1f}"
              f"{'PASS' if a['passed'] else 'FAIL':>10}")
    n = len(both)
    p = _sign_p(fell, n)
    print(f"\n  {fell} of {n} models fell.  paired sign test p = {p:.4f}")
    print(f"  still clearing the competence gate: "
          f"{sum(a['passed'] for a in after.values())}/{len(after)}")
    return {"fell": fell, "n": n, "p": p,
            "passed": sum(a["passed"] for a in after.values())}


def verdict(w: dict, c: dict) -> None:
    #: **DIRECTION IS NOT THE SAME AS CHANGE.** The paired Fisher is two-sided,
    #: so EVENT means "moved", not "fell". Reading EVENT as a fall inverted this
    #: round's conclusion on the first pass: W2 and W3 both moved UP.
    def moved_down(v):
        now, prior = v.rates()
        return v.verdict == "EVENT" and now < prior
    def moved_up(v):
        now, prior = v.rates()
        return v.verdict == "EVENT" and now > prior

    down = [k for k in ("W2", "W3") if moved_down(w[k])]
    up = [k for k in ("W2", "W3") if moved_up(w[k])]
    comp_down = c["p"] < R.OCCASION_ALPHA and c["fell"] > c["n"] / 2
    comp_up = c["p"] < R.OCCASION_ALPHA and c["fell"] < c["n"] / 2

    print("\n\nDIRECTIONS, stated before any branch is matched")
    print(f"    W2/W3 down: {down or 'none'}    W2/W3 up: {up or 'none'}")
    print(f"    COMP: {'down' if comp_down else 'up' if comp_up else 'flat'}"
          f"  ({c['n'] - c['fell']} of {c['n']} rose, p={c['p']:.4f})")

    fell_worlds = bool(down)
    fell_comp = comp_down
    key = (f"W2/W3 {'fall' if fell_worlds else 'hold'} "
           f"{'AND' if fell_worlds == fell_comp else 'BUT'} "
           f"COMP {'falls' if fell_comp else 'holds'}")
    matched = R.READING.get(key)
    print(f"\n\nFROZEN READING — nearest branch: [{key}]\n")
    import textwrap
    for line in textwrap.wrap(matched or "NO BRANCH MATCHED", 70):
        print(f"    {line}")
    if up or comp_up:
        print("\n    **BUT THE TABLE HAS NO BRANCH FOR THIS.** The frozen")
        print("    reading imagined only 'fall' or 'hold'. Both co-located")
        print("    worlds and the competence site moved UP on the same day LAT")
        print("    moved down. That is a pre-registration defect, recorded as")
        print("    one: the table was incomplete, not wrong.")
    print(f"\n  LAT for comparison (round 16's anchored six):")
    for day, (k, n) in sorted(R.LAT_TRACE.items()):
        print(f"    {day}   {k}/{n} = {k / n:.4f}")


if __name__ == "__main__":
    W = worlds()
    Cm = competence()
    verdict(W, Cm)
