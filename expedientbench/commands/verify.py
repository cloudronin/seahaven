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

from ..register import verify as _verify


def main(_args=None) -> int:
    bad, rows = _verify()
    print(f"CLAIMS REGISTER — {len(rows)} figures, recomputed from committed cells\n")
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
