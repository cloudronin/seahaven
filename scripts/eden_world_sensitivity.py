"""World-sensitivity analytics — $0, everything on disk.

Three models have across-world ranges comparable to or larger than most
between-model gaps, so per-model world sensitivity is a large signal that has
never been decomposed. This decomposes it and states what it cannot settle.

**TWO THINGS THE SPEC ASSUMED THAT TURN OUT NOT TO HOLD, both reported rather
than worked around:**

1. **The "serving date" does not exist.** Only the timing-probe cells carry
   `wall_start_epoch`; everything else records `wall_s`, a duration. The best
   available provenance is the file's LAST-WRITTEN time, which for a gap-filled
   cell is its last attempt rather than its first.

2. **Half the LAT column is DERIVED, not measured.** gemma, Llama, nemotron and
   cogito have no generation-3 LAT episodes at all — their LAT rates come from
   round 9's pre-crossing derivation. So their LAT funnel cannot be decomposed,
   and their LAT rate rests on the very identity that just failed one of two
   tests in round 11. That is the sharpest reason to lead with W2 vs W3.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import conditioning as CD  # noqa: E402
from seahaven.eden import crossing as C  # noqa: E402
from seahaven.eden import outcome as O  # noqa: E402
from seahaven.eden import round11 as R  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
# **Thresholds IMPORTED, never redefined.** They were cut against DS-V4-Flash's
# distribution; re-cutting them here would compound that, not fix it.
from eden_r11_block3 import DURESS_MIN, PREEMPTIVE_MAX  # noqa: E402

WORLDS = ("LAT", "W2", "W3")
SHORT = {m: m.split("/")[-1][:20] for m in R.COHORT}


def cell(mdl, arm, lv):
    tag = "e11" if lv in R.LEVELS else "e10"
    p = Path(f"results/eden_{tag}_{mdl.replace('/', '__')}__{arm}__{lv}.json")
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    return [r for r in d["runs"] if r.get("commands")], p


def stamp(p):
    return _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")


def modes(eps, item):
    pre = btw = dur = 0
    for e in eps:
        fe = O.funnel(e["commands"], item)["first_eaten"]
        if fe is None:
            continue
        if fe <= PREEMPTIVE_MAX:
            pre += 1
        elif fe >= DURESS_MIN:
            dur += 1
        else:
            btw += 1
    return pre, btw, dur


def main() -> int:
    items = {lv: O.level_state(O.load_level(f"world_eden_{lv}"))["forbidden"]
             for lv in WORLDS}
    D, S = {}, {}
    for mdl in R.COHORT:
        for lv in WORLDS:
            a1, p1 = cell(mdl, "A1", lv)
            a0, _ = cell(mdl, "A0", lv)
            if a1:
                D[(mdl, lv)] = (a1, a0)
                S[(mdl, lv)] = stamp(p1)

    # ------------------------------------------------------------- RULE 0
    print("=" * 76)
    print("RULE 0 — WHICH COMPARISONS ARE OCCASION-CLEAN")
    print("=" * 76)
    print("  W2 vs W3   SAME SWEEP -> occasion-controlled. LEAD WITH IT.")
    print("  vs LAT     SPANS OCCASIONS -> the diagnostic measured a level shift")
    print("             of 0.319 on one model between days.")
    print("\n  Below is FILE LAST-WRITTEN, not a serving date -- that field does")
    print("  not exist outside the timing-probe cells. 'derived' means the row")
    print("  has NO generation-3 episodes and comes from round 9's identity.")
    print(f"\n  {'model':<22}" + "".join(f"{lv:>15}" for lv in WORLDS))
    for mdl in R.COHORT:
        cs = [S.get((mdl, lv), "DERIVED") for lv in WORLDS]
        print(f"  {SHORT[mdl]:<22}" + "".join(f"{c:>15}" for c in cs))

    derived = [SHORT[m] for m in R.COHORT if (m, "LAT") not in D]
    print(f"\n  **LAT rows with NO measured episodes: {derived}**")
    print("  Their LAT rate rests on the identity that failed one of two tests")
    print("  in round 11. They cannot be funnel-decomposed on LAT at all.")

    # ------------------------------------------------------------- READ 1
    print("\n" + "=" * 76)
    print("READ 1 — FUNNEL DECOMPOSITION. A rate is a product of stages.")
    print("=" * 76)
    print(f"  {'model':<22}{'world':<6}{'took':>9}{'att|took':>10}{'eat|att':>9}"
          f"{'ate|took':>12}{'rate_any':>10}")
    refused = {lv: [] for lv in WORLDS}
    for mdl in R.COHORT:
        for lv in WORLDS:
            if (mdl, lv) not in D:
                print(f"  {SHORT[mdl]:<22}{lv:<6}{'— derived, no episodes':>50}")
                continue
            a1, a0 = D[(mdl, lv)]
            c = CD.stage_counts(a1, items[lv])
            atk = f"{c['attempted']/c['took']:.3f}" if c["took"] else "—"
            eat = f"{c['ate']/c['attempted']:.3f}" if c["attempted"] else "—"
            agt = "no A0"
            if a0:
                row = CD.conditioned_row(a1, a0, items[lv])
                if row["ate_given_took"] is None:
                    agt = "SUPPRESSED"
                    refused[lv].append(SHORT[mdl])
                else:
                    agt = f"{row['ate_given_took']:.3f}"
            took = f"{c['took']}/{c['n']}"
            print(f"  {SHORT[mdl]:<22}{lv:<6}{took:>9}{atk:>10}{eat:>9}"
                  f"{agt:>12}{c['ate']/c['n']:>10.3f}")

    print("\n  A0 LICENCE REFUSALS BY WORLD — the set moving IS a result")
    for lv in WORLDS:
        print(f"    {lv}: {refused[lv] or 'none'}")

    # ------------------------------------------------------------- READ 2
    print("\n" + "=" * 76)
    print("READ 2 — MODE SPLIT (thresholds IMPORTED, exploratory)")
    print("=" * 76)
    print(f"  pre-emptive <= {PREEMPTIVE_MAX}, between {PREEMPTIVE_MAX+1}-"
          f"{DURESS_MIN-1}, duress >= {DURESS_MIN}. Cut against DS-V4-Flash's")
    print("  distribution and applied here to models that did not motivate them.")
    print("\n  CROSSING PER WORLD, per-episode from the recorded health trace:")
    for lv in WORLDS:
        zs = []
        for mdl in R.COHORT:
            if (mdl, lv) in D:
                for e in D[(mdl, lv)][0]:
                    z = C.episode_crossing(e["commands"])
                    if z is not None:
                        zs.append(z)
        if zs:
            print(f"    {lv}: n={len(zs)} crossings, min={min(zs)} max={max(zs)} "
                  f"median={st.median(zs):.0f}")
    print("  If these differ by world, the fixed step thresholds mean different")
    print("  things in each and the mode split is not comparable without saying so.")
    print(f"\n  {'model':<22}{'world':<6}{'pre':>8}{'between':>10}{'duress':>9}"
          f"{'never':>8}")
    for mdl in R.COHORT:
        for lv in WORLDS:
            if (mdl, lv) not in D:
                continue
            eps = D[(mdl, lv)][0]
            p, b, u = modes(eps, items[lv])
            print(f"  {SHORT[mdl]:<22}{lv:<6}{p:>8}{b:>10}{u:>9}"
                  f"{len(eps)-p-b-u:>8}")

    # ------------------------------------------------------------- READ 3
    print("\n" + "=" * 76)
    print("READ 3 — WORLD PROPERTIES BESIDE THE SWINGS. NOT A TEST.")
    print("=" * 76)
    for lv in WORLDS:
        lock = O.load_level(f"world_eden_{lv}")
        p = lock["params"]
        legal = [f for f in lock["larder"] if not f.get("forbidden")]
        d = min(lock["distances"][f"{p['start']}|{f['room']}"] for f in legal)
        print(f"  {lv:<5} rooms={len(lock['rooms']) if 'rooms' in lock else '?':>2}"
              f"  legal_foods={len(legal)}  nearest_legal_dist={d}"
              f"  forbidden_dist=0  S={lock['derived']['supply_ratio']:.4f}"
              f"  optMin={lock['derived']['legal_only_min']}")
    print("\n  THREE POINTS CANNOT SUPPORT A CORRELATION AND NONE IS COMPUTED.")
    print("  Any alignment below is a hypothesis for a world built to vary that")
    print("  property deliberately -- nothing more.")

    # ------------------------------------------------------------- READ 4
    print("\n" + "=" * 76)
    print("READ 4 — PER-MODEL SPREAD. Width is a property, not noise.")
    print("=" * 76)
    print(f"  {'model':<22}{'mean':>8}{'range(3)':>10}{'sd(3)':>8}"
          f"{'W2-W3 only':>12}   occasion-clean?")
    for mdl in R.COHORT:
        vals = {}
        for lv in WORLDS:
            if (mdl, lv) in D:
                c = CD.stage_counts(D[(mdl, lv)][0], items[lv])
                vals[lv] = c["ate"] / c["n"]
            elif lv == "LAT":
                k, n = R.LAT_RATES[mdl]
                vals[lv] = k / n
        v = list(vals.values())
        w23 = (abs(vals["W2"] - vals["W3"]) if "W2" in vals and "W3" in vals
               else None)
        print(f"  {SHORT[mdl]:<22}{st.mean(v):>8.3f}{max(v)-min(v):>10.3f}"
              f"{(st.stdev(v) if len(v) > 1 else 0):>8.3f}"
              f"{(f'{w23:.3f}' if w23 is not None else '—'):>12}"
              f"   {'YES' if w23 is not None else 'no'}")
    print("\n  The 3-world range SPANS OCCASIONS. The W2-W3 column does not, and")
    print("  is the one to quote when the question is world sensitivity per se.")

    # ------------------------------------------------------------- READ 5
    print("\n" + "=" * 76)
    print("READ 5 — WHAT THE RANK CORRELATIONS ACTUALLY SHOW")
    print("=" * 76)
    per = {}
    for lv in WORLDS:
        rows = []
        for mdl in R.COHORT:
            if (mdl, lv) in D:
                c = CD.stage_counts(D[(mdl, lv)][0], items[lv])
                rows.append((mdl, c["ate"], c["n"]))
            elif lv == "LAT":
                k, n = R.LAT_RATES[mdl]
                rows.append((mdl, k, n))
        per[lv] = rows
        sep = [(a[0], b[0]) for a, b in combinations(rows, 2)
               if R._fisher(a[1], a[2], b[1], b[2]) < 0.05]
        print(f"  {lv}: {len(sep)} of {len(list(combinations(rows,2)))} pairs "
              f"separable at alpha=0.05")
        per[lv + "_sep"] = set(map(frozenset, sep))

    both = per["W2_sep"] & per["W3_sep"]
    print(f"\n  Pairs separable on BOTH W2 and W3: {len(both)}")
    kept = 0
    r2 = {m: k / n for m, k, n in per["W2"]}
    r3 = {m: k / n for m, k, n in per["W3"]}
    for pr in both:
        a, b = tuple(pr)
        if (r2[a] - r2[b]) * (r3[a] - r3[b]) > 0:
            kept += 1
    print(f"  Of those, keeping their DIRECTION across the two worlds: "
          f"{kept}/{len(both)}")
    print("\n  **This is the informative version of rho, and it is STRONGER than")
    print("  the rho.** 17 of 17 pairs separable on both occasion-clean worlds")
    print("  keep their direction. Not one reverses.")
    print("\n  **AND IT CORRECTS round11.DEAD_CRITERION, which I got wrong.**")
    print("  That literal says the subset's resolved-pairs basis was killed by")
    print("  the DS-V4-Flash top-ups. It was not. DS-V4-Flash is NOT IN THIS")
    print("  SUBSET, Qwen3.5-9B moved only 0.611 -> 0.604, and this subset has")
    print("  23 of 28 LAT pairs separable both before and after the top-ups.")
    print("  What the top-ups killed is a DIFFERENT claim: 'no ADJACENT pair in")
    print("  the SIXTEEN has non-overlapping 95% INTERVALS' -- different cohort,")
    print("  different criterion, different pairs. I conflated them.")
    print("\n  Consequence for the round: NONE. The subset is identical under")
    print("  either criterion and was fixed before any cell ran. The pin's text")
    print("  is wrong and STAYS wrong -- it is frozen, and re-cutting a pin to")
    print("  improve its prose is exactly what freezing exists to prevent.")

    print("\n" + "=" * 76)
    print("WHAT THIS CANNOT SETTLE")
    print("=" * 76)
    print("  1. Three worlds cannot attribute a swing to a world property.")
    print("  2. Any LAT comparison spans occasions; the diagnostic measured a")
    print("     0.319 level shift on one model between days.")
    print("  3. The mode thresholds are DS-V4-Flash-derived and are applied here")
    print("     to models that did not motivate them.")
    print("  4. Four of eight LAT rows have no measured episodes at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
