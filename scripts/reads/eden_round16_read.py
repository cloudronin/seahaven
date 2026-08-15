"""Round 16's read, with SELF_CERTIFY enforced by control flow.

The pin says the ordering is frozen "because it is exactly the check that is
skipped once the interesting numbers are already on screen". So the fourteen's
numbers are not computed at all until the six anchored models have returned
QUIET — not printed later, not printed with a warning: not computed.
"""

from __future__ import annotations

import sys

from seahaven.eden import round16 as R
from seahaven.eden._shared import occasion as OQ

SWEEP, WORLD = "16", "LAT"


def certify() -> OQ.Verdict:
    """Step 1. The six anchored models are this sweep's own reference channel."""
    obs = OQ.observations()
    here = [o for o in obs if o.world == WORLD and o.sweep == SWEEP]
    if not here:
        sys.exit(f"no round-{SWEEP} {WORLD} A0 cells on disk yet")
    v = OQ.verdict_for(WORLD, SWEEP, alpha=R.OCCASION_ALPHA, obs=obs)
    now, prior = v.rates()
    print("SELF-CERTIFICATION — is the serving day itself quiet on this channel?\n")
    print(f"  anchored models   {len(v.returning)}  "
          f"({', '.join(m.split('/')[-1] for m in v.returning)})")
    print(f"  their prior       {prior:.4f} {OQ.interval(*v.prior)}  "
          f"(clean baseline, event day excluded)")
    print(f"  their today       {now:.4f} {OQ.interval(*v.now)}")
    print(f"  Fisher p          {v.p:.3e}" if v.p is not None else "  Fisher p  —")
    print(f"\n  VERDICT: {v.verdict}\n")
    for m, rn, rp, pm in v.flags:
        print(f"    flag {m.split('/')[-1][:28]:<30}{rp:.3f} -> {rn:.3f}  p={pm:.4f}")
    return v


def read_fourteen(v: OQ.Verdict) -> None:
    """Step 2. Reached ONLY on QUIET."""
    from seahaven.eden import conditioning as CD, intent as I
    from seahaven.eden._shared import corpus as C

    item = OQ.forbidden_item(WORLD)
    a0, a1 = {}, {}
    for p, d in C.iter_cells():
        m = d.get("meta", {})
        g = C.parse_cell_name(p.name)
        if not g or g["round"] != SWEEP or m.get("eden_level") != WORLD:
            continue
        eps = C.episodes(d)
        if m.get("eden_arm") == "A0":
            a0[m["served_name"]] = CD.stage_counts(eps, item)
        else:
            a1[m["served_name"]] = I.intent_counts(eps, item)

    anchored = set(R.CLEAN_LAT_ANCHOR)
    print("\nTHE FOURTEEN — A0 today vs the event day, and A1 beside it\n")
    print(f"  {'model':<28}{'anch':>5}{'A0 now':>9}{'A0 08-14':>10}"
          f"{'A1 intent':>11}")
    for mdl in sorted(a0, key=lambda m: -(a0[m]['ate'] / max(a0[m]['n'], 1))):
        c0 = a0[mdl]
        c1 = a1.get(mdl)
        ev = [o for o in OQ.observations()
              if o.world == WORLD and o.sweep == "15" and o.model == mdl]
        evr = f"{ev[0].rate:.3f}" if ev else "—"
        a1r = f"{c1['attempted'] / c1['n']:.3f}" if c1 else "—"
        print(f"  {mdl.split('/')[-1][:26]:<28}"
              f"{'Y' if mdl in anchored else '.':>5}"
              f"{c0['ate'] / c0['n']:>9.3f}{evr:>10}{a1r:>11}")

    un = [m for m in a0 if m not in anchored]
    k = sum(a0[m]["ate"] for m in un)
    n = sum(a0[m]["n"] for m in un)
    print(f"\n  THE EIGHT UNANCHORED, pooled: {k}/{n} = {k / n:.4f}")
    print(f"    event day was {R.EVENT_DAY_UNANCHORED_8[0]}/"
          f"{R.EVENT_DAY_UNANCHORED_8[1]} = "
          f"{R.EVENT_DAY_UNANCHORED_8[0] / R.EVENT_DAY_UNANCHORED_8[1]:.4f}")
    print("\n  THE FROZEN FORK — read against this, not around it:")
    for branch, reading in R.FORK.items():
        print(f"\n    [{branch}]")
        import textwrap
        for line in textwrap.wrap(reading, 66):
            print(f"      {line}")
    print("\n  SERVING-DAY CAVEAT (asymmetric — applies to the recovered branch"
          " only):")
    import textwrap
    for line in textwrap.wrap(R.SERVING_DAY_CAVEAT, 68):
        print(f"    {line}")


if __name__ == "__main__":
    verdict = certify()
    if verdict.verdict != "QUIET":
        print(f"  STOP. {R.SELF_CERTIFY}")
        sys.exit(0)
    read_fourteen(verdict)
