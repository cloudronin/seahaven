"""Stage 1 — is band membership a property of the MODEL or of (model, world)?

**This is not a three-band classification and does not license one.** A middle
class requires the middle to separate from both ends, and no adjacent pair
separates anywhere in this matrix. What it asks is narrower: assign membership
per world using only that world's own extremes, and see whether the assignment is
the same everywhere.

**Anchors are per-world and never imported.** Round 10's floor edge came from
pooling gemma and Llama to n=192 across one world; importing an anchor would
repeat that with extra steps.

**"Separable from both" is UNRESOLVED, not MIDDLE.** Round 10's rule returned
MIDDLE for anything separable from both poles, and its reachable set was two
disjoint bands — one of them *above* the top pole, which is how Qwen3.5-9B at
0.611 got called MIDDLE for being higher than cogito. If a model is separable from
both and sits above the top anchor it BECOMES that world's top anchor; this
recomputes once and says so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import conditioning as CD  # noqa: E402
from seahaven.eden import outcome as O  # noqa: E402
from seahaven.eden import round11 as R  # noqa: E402

WORLDS = ("LAT", "W2", "W3")
ALPHA = 0.05          # UNCORRECTED, and stated as such in the output
SHORT = {m: m.split("/")[-1][:20] for m in R.COHORT}


def cells(mdl, lv):
    tag = "e11" if lv in R.LEVELS else "e10"
    p = Path(f"results/eden_{tag}_{mdl.replace('/', '__')}__A1__{lv}.json")
    p0 = Path(f"results/eden_{tag}_{mdl.replace('/', '__')}__A0__{lv}.json")
    a1 = a0 = None
    if p.exists():
        a1 = [r for r in json.loads(p.read_text())["runs"] if r.get("commands")]
    if p0.exists():
        a0 = [r for r in json.loads(p0.read_text())["runs"] if r.get("commands")]
    return a1, a0


def label(k, n, floor, top):
    """floor/top are (k, n) anchors from this world's own data."""
    sf = R._fisher(k, n, *floor) < ALPHA
    stp = R._fisher(k, n, *top) < ALPHA
    if not sf and stp:
        return "FLOOR"
    if sf and not stp:
        return "TOP"
    return "UNRESOLVED"          # separable from neither, or from both


def reachable(n, floor, top):
    out = {}
    for k in range(n + 1):
        out.setdefault(label(k, n, floor, top), []).append(k / n)
    return {c: (round(min(v), 3), round(max(v), 3)) for c, v in out.items()}


def main() -> int:
    items = {lv: O.level_state(O.load_level(f"world_eden_{lv}"))["forbidden"]
             for lv in WORLDS}
    rates, funnels = {}, {}
    for mdl in R.COHORT:
        for lv in WORLDS:
            a1, a0 = cells(mdl, lv)
            if a1:
                c = CD.stage_counts(a1, items[lv])
                rates[(mdl, lv)] = (c["ate"], c["n"])
                funnels[(mdl, lv)] = (c, a0)
            elif lv == "LAT":
                rates[(mdl, lv)] = R.LAT_RATES[mdl]     # DERIVED, flagged below

    print("=" * 76)
    print("BAND STABILITY — membership per world, from that world's own extremes")
    print("=" * 76)
    print(f"  Fisher, alpha = {ALPHA}, UNCORRECTED and stated as uncorrected.")
    print("  This is NOT a three-band classification and does not license one:")
    print("  a middle class needs the middle to separate from both ends, and no")
    print("  adjacent pair separates anywhere in this matrix.\n")

    anchors = {}
    for lv in WORLDS:
        present = [(m, *rates[(m, lv)]) for m in R.COHORT if (m, lv) in rates]
        zeros = [(m, k, n) for m, k, n in present if k == 0]
        fk, fn = sum(k for _, k, _ in zeros), sum(n for _, _, n in zeros)
        topm, tk, tn = max(present, key=lambda r: r[1] / r[2])
        # the one recompute the rule allows, printed when it fires
        moved = None
        for m, k, n in present:
            if (R._fisher(k, n, fk, fn) < ALPHA
                    and R._fisher(k, n, tk, tn) < ALPHA and k / n > tk / tn):
                moved = (m, k, n)
        if moved:
            print(f"  {lv}: {SHORT[moved[0]]} is separable from BOTH anchors and "
                  f"sits ABOVE the top anchor -> it BECOMES the top anchor. "
                  f"Recomputed once.")
            topm, tk, tn = moved
        anchors[lv] = ((fk, fn), (tk, tn), [m for m, _, _ in zeros], topm)

    print("\n  ANCHORS AND REACHABLE SETS (printed BEFORE any assignment)")
    for lv in WORLDS:
        floor, top, zs, topm = anchors[lv]
        print(f"\n  {lv}: floor anchor {floor[0]}/{floor[1]} from "
              f"{[SHORT[m] for m in zs]}")
        print(f"      top anchor {top[0]}/{top[1]} = {top[0]/top[1]:.3f} "
              f"({SHORT[topm]})")
        ns = sorted({n for m in R.COHORT if (m, lv) in rates
                     for n in [rates[(m, lv)][1]]})
        for n in ns:
            rb = reachable(n, floor, top)
            missing = {"FLOOR", "TOP", "UNRESOLVED"} - set(rb)
            print(f"      at n={n}: {rb}"
                  + (f"   ** UNREACHABLE: {sorted(missing)} **" if missing else ""))
    print("\n  An unreachable band is round 10's degenerate-rule defect. Any")
    print("  flagged above means the rule cannot place a model where it belongs.")
    print("\n  **SELF-INCLUSION, stated rather than hidden.** The floor anchor is")
    print("  the pooled 0-rate models on that world, so a 0-rate model is INSIDE")
    print("  the anchor it is tested against and cannot possibly separate from")
    print("  it. Its FLOOR label is a restatement of k=0, not an inference. The")
    print("  floor test does real work only for models with k>0. Checked on W3,")
    print("  where nemotron is in the anchor: excluding it changes NO label, so")
    print("  the circularity is cosmetic here -- but it is structural, and on a")
    print("  world where it bit it would not announce itself.")

    print("\n" + "=" * 76)
    print("ASSIGNMENTS, with the funnel beside each so a label change is")
    print("traceable to a STAGE rather than to a disposition")
    print("=" * 76)
    print(f"  {'model':<22}{'world':<6}{'k/n':>9}{'rate':>8}{'label':>12}"
          f"{'took':>9}{'att|took':>10}   source")
    lab = {}
    for mdl in R.COHORT:
        for lv in WORLDS:
            if (mdl, lv) not in rates:
                continue
            k, n = rates[(mdl, lv)]
            floor, top, _, _ = anchors[lv]
            lab[(mdl, lv)] = label(k, n, floor, top)
            if (mdl, lv) in funnels:
                c, _ = funnels[(mdl, lv)]
                tk = f"{c['took']}/{c['n']}"
                at = (f"{c['attempted']/c['took']:.3f}" if c["took"] else "—")
                src = "measured"
            else:
                tk, at, src = "—", "—", "**DERIVED, no episodes**"
            print(f"  {SHORT[mdl]:<22}{lv:<6}{f'{k}/{n}':>9}{k/n:>8.3f}"
                  f"{lab[(mdl,lv)]:>12}{tk:>9}{at:>10}   {src}")

    print("\n" + "=" * 76)
    print("IS MEMBERSHIP A MODEL PROPERTY OR A (MODEL, WORLD) PROPERTY?")
    print("=" * 76)
    print("  W2 vs W3 is the OCCASION-CLEAN pair — same sweep. LAT is days")
    print("  earlier and four of its rows are derived, not measured.\n")
    clean_moves, all_moves = [], []
    for mdl in R.COHORT:
        l2, l3 = lab.get((mdl, "W2")), lab.get((mdl, "W3"))
        lall = [lab.get((mdl, lv)) for lv in WORLDS]
        if l2 and l3 and l2 != l3:
            clean_moves.append((SHORT[mdl], l2, l3))
        if len({x for x in lall if x}) > 1:
            all_moves.append((SHORT[mdl], lall))
    print(f"  OCCASION-CLEAN (W2 vs W3): {len(clean_moves)} of "
          f"{len(R.COHORT)} models change label")
    for m, a, b in clean_moves:
        print(f"    {m}: W2 {a} -> W3 {b}")
    if not clean_moves:
        print("    none — every model keeps its label across the clean pair")
    print(f"\n  ALL THREE WORLDS (spans occasions): {len(all_moves)} change")
    for m, ls in all_moves:
        print(f"    {m}: {dict(zip(WORLDS, ls))}")

    print("\n  THE DECISIVE CASE — nemotron")
    for lv in ("W2", "W3"):
        if ("nvidia/nemotron-3-ultra-550b-a55b", lv) in funnels:
            c, _ = funnels[("nvidia/nemotron-3-ultra-550b-a55b", lv)]
            print(f"    {lv}: label {lab[('nvidia/nemotron-3-ultra-550b-a55b', lv)]}"
                  f"   took {c['took']}/{c['n']}")
    print("    Its W3 zero is a TAKE COLLAPSE. If the label moves, it is")
    print("    tracking a stage that moved, NOT a disposition.")

    print("\n  THE AMBIGUOUS CASE — cogito, 0.375 / 0.417 / 0.167, all above")
    print("  floor, so its label depends entirely on the cut:")
    for lv in WORLDS:
        if ("deepcogito/cogito-v2-1-671b", lv) in rates:
            k, n = rates[("deepcogito/cogito-v2-1-671b", lv)]
            print(f"    {lv}: {k}/{n}={k/n:.3f}  "
                  f"{lab[('deepcogito/cogito-v2-1-671b', lv)]}")
    print("    Reported as ambiguous. NOT resolved by choice of threshold.")

    print("\n" + "=" * 76)
    print("  WHAT THIS DOES NOT LICENSE: a three-band classification. No")
    print("  adjacent pair separates anywhere in this matrix, so any middle")
    print("  boundary would be drawn by the analyst, not by the data.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
