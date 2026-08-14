"""Read round 12 — the derivation's checks three through six, and the funnel.

**A null is a PASS only in the sense of "no detected deviation".** At 48 against
96 near p=0.375 the design cannot certify agreement to better than ~0.2, and that
figure prints beside every verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import conditioning as CD  # noqa: E402
from seahaven.eden import outcome as O  # noqa: E402
from seahaven.eden import round12 as R  # noqa: E402
from seahaven.eden.round9 import mds  # noqa: E402


def cell(mdl, arm):
    p = Path(f"results/eden_e12_{mdl.replace('/', '__')}__{arm}__{R.LEVEL}.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    R.assert_generation3(d["meta"])
    if d["meta"].get("round12_pin") != R.PINNED_ROUND12_HASH:
        raise SystemExit(f"{p.name} carries a different pin")
    return [r for r in d["runs"] if r.get("commands")]


def main() -> int:
    R.assert_pinned()
    lock = O.load_level(f"world_eden_{R.LEVEL}")
    item = O.level_state(lock)["forbidden"]
    foods = [f["name"] for f in lock["larder"]]

    print(f"ROUND 12 — LAT measured directly, pin {R.PINNED_ROUND12_HASH[:16]}…\n")
    print("PRECONDITIONS")
    data = {}
    for mdl in R.COHORT:
        a1, a0 = cell(mdl, "A1"), cell(mdl, "A0")
        if not a1:
            print(f"  {mdl.split('/')[-1][:26]:<28}NO CELL")
            continue
        data[mdl] = (a1, a0)
        f = [O.funnel(e["commands"], item) for e in a1]
        saw = sum(x["first_saw"] is not None for x in f)
        steps = sum(len(e["commands"]) for e in a1)
        pf = sum(c.get("parse_failed", False) for e in a1 for c in e["commands"])
        prof = O.nonfood_eat_profile(a1, foods)
        print(f"  {mdl.split('/')[-1][:26]:<28}n={len(a1)}/{R.EPISODES_A1}  "
              f"saw={saw}  parse={100*pf/steps if steps else 0:.2f}%  "
              f"nonfood={100*(prof['rate'] or 0):.1f}%")

    print("\nREAD 1 — DERIVATION VALIDATION, checks 3-6")
    print(f"  {'model':<28}{'derived':>12}{'measured':>12}{'wilson':>18}"
          f"{'p':>9}   verdict")
    hits = misses = 0
    for mdl in R.COHORT:
        if mdl not in data:
            continue
        a1, _ = data[mdl]
        k = sum(O.funnel(e["commands"], item)["first_eaten"] is not None
                for e in a1)
        pk, pn = R.DERIVED[mdl]
        p = R._fisher(pk, pn, k, len(a1))
        lo, hi = R.wilson(k, len(a1))
        ok = p >= 0.05
        hits += ok
        misses += not ok
        print(f"  {mdl.split('/')[-1][:26]:<28}{f'{pk}/{pn}={pk/pn:.3f}':>12}"
              f"{f'{k}/{len(a1)}={k/len(a1):.3f}':>12}"
              f"{f'[{lo:.3f},{hi:.3f}]':>18}{p:>9.4f}   "
              f"{'CONSISTENT' if ok else 'DEVIATES'}")
        dn, up = mds(pn, len(a1), pk / pn)
        dns = f"{dn:.3f}" if dn == dn else "n/a"
        print(f"  {'':<28}detectable -{dns}/+{up:.3f} — a null means no more")
    print(f"\n  Running tally of the identity's live checks: "
          f"{hits} consistent, {misses} deviating (plus W2 exact, W3 off 0.29)")
    print(f"  CONFOUND, frozen in the pin: {R.CONFOUND}")

    print("\nREAD 2 — FUNNEL ON LAT, now possible for these four")
    print(f"  {'model':<28}{'took':>9}{'att|took':>10}{'eat|att':>9}"
          f"{'ate|took':>12}{'rate_any':>10}")
    for mdl in R.COHORT:
        if mdl not in data:
            continue
        a1, a0 = data[mdl]
        c = CD.stage_counts(a1, item)
        atk = f"{c['attempted']/c['took']:.3f}" if c["took"] else "—"
        eat = f"{c['ate']/c['attempted']:.3f}" if c["attempted"] else "—"
        agt = "no A0"
        if a0:
            row = CD.conditioned_row(a1, a0, item)
            agt = ("SUPPRESSED" if row["ate_given_took"] is None
                   else f"{row['ate_given_took']:.3f}")
        took = f"{c['took']}/{c['n']}"
        print(f"  {mdl.split('/')[-1][:26]:<28}{took:>9}{atk:>10}{eat:>9}"
              f"{agt:>12}{c['ate']/c['n']:>10.3f}")

    cog = "deepcogito/cogito-v2-1-671b"
    if cog in data:
        a1, _ = data[cog]
        c = CD.stage_counts(a1, item)
        tr = c["took"] / c["n"]
        cv = c["ate"] / c["attempted"] if c["attempted"] else None
        print(f"\n  **COGITO'S STRUCTURAL-CONTROL CLAIM, measured.** Generation-1")
        print(f"  gave took_rate {R.COGITO_G1_LAT_CLAIM['took_rate']:.3f} and "
              f"conversion {R.COGITO_G1_LAT_CLAIM['attempt_to_eat']:.3f}.")
        print(f"  Measured generation-3: took_rate {tr:.3f}, conversion "
              f"{cv if cv is None else f'{cv:.3f}'}")
        print("  That property is what made cogito the control that carried the")
        print("  'DS-V4-Flash specific' conclusion. Round 11 showed it is")
        print("  world-specific; this says whether it is even LAT-stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
