"""`expdx verify` — recompute the paper. $0, no key.

The strong replication claim, and it is exact. Every figure the manuscript quotes
is emitted by a named function from committed cells; this recomputes all of them
and exits nonzero on any drift, printing the figure id, the published value and
the recomputed one.

**This is the claim that is achievable by anyone.** The other claim — that a NEW
run lands where ours did — is `replicate`'s, is achievable only within bands, and
has a different success criterion, because the programme's own findings say so: a
0.319 between-occasion shift with mechanism unresolved, and hosted serving that is
not batch-invariant.
"""

from __future__ import annotations

from seahaven.eden._shared import corpus as C
from ..register import verify as _verify


def main(_args=None) -> int:
    # **An absent corpus is not a drifting manuscript.** Computing over zero
    # cells reports every figure as changed, which would tell a replicator the
    # paper is wrong when the truth is that they have no data. Caught by running
    # this from an installed wheel, where it claimed 9 figures had drifted.
    n = C.corpus_present()
    if not n:
        print("NO CORPUS. Nothing to verify against.\n")
        print(f"  Looked in: {C.RESULTS.resolve()}")
        print("  The committed cells are distributed separately from the code.")
        print("  Fetch them, then re-run from a directory that contains them.")
        print("\n  This is NOT a claim that any figure drifted -- with zero")
        print("  cells every figure would 'differ', and that would be a")
        print("  statement about your disk, not about the manuscript.")
        return 2
    bad, rows = _verify()
    print(f"CLAIMS REGISTER — {len(rows)} figures, recomputed from "
          f"{n} committed cells\n")
    print(f"  {'':<5}{'figure':<26}{'published':<20}{'recomputed':<20}source")
    for c, got, ok in rows:
        mark = "ok " if ok else "DRIFT"
        prov = ("measured" if c.measured else "DERIVED") + f", {c.generation}"
        print(f"  {mark:<5}{c.fid:<26}{str(c.value):<20}"
              f"{(str(got) if not ok else ''):<20}{prov}   {c.cells}")
        if not ok:
            print(f"        ** {c.fid}: manuscript says {c.value!r}, "
                  f"cells say {got!r} **")
        if c.note:
            print(f"        {c.note}")
    # **The occasion regression.** A figure whose cells were served at different
    # sittings has a second explanation for its difference, and the manuscript
    # carries that flag in place. Enforced here rather than remembered: the
    # 0.319 shift was measured, its mechanism was not established, and a new
    # cross-occasion figure published unmarked is the failure this cannot risk.
    from ..register import occasions as OC
    missing = OC.unflagged()
    if missing:
        print(f"\n{len(missing)} FIGURE(S) COMPARE CELLS ACROSS SERVING "
              "OCCASIONS WITH NO FLAG:")
        for fid in missing:
            print(f"    {fid}")
        print("  Add `occasion=` to the claim saying what the reader must know,")
        print("  or show the cells share a sweep AND a recorded serving time.")
        print("  `expdx emit occasions` prints what each figure actually reads.")
    if bad or missing:
        if bad:
            print(f"\n{bad} FIGURE(S) DRIFTED. The manuscript and the corpus "
                  "disagree; fix one of them.")
        return 1
    print(f"\nall {len(rows)} figures recompute.")
    print("  Figures marked DERIVED come from round 9's pre/post-crossing")
    print("  identity rather than measured episodes. That identity has been")
    print("  checked six times: five consistent, one off by 0.29 with an")
    print("  occasion effect as an unresolvable alternative explanation.")
    return 0
