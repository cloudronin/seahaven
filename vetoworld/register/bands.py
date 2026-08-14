"""Replication bands — computed once, shipped as data, never improvised.

**A replicator judged against our point estimate will "fail to replicate" and be
wrong about what that means.** The programme's own findings say so: a 0.319
between-occasion shift on one model with the mechanism unresolved, hosted serving
that is not batch-invariant, and the fact that band membership turned out to be a
property of (model, world) rather than of the model.

So a band is the Wilson interval on OUR n, **widened by the measured occasion
component where one exists**. The widening is not a fudge factor: it is the
largest between-occasion shift this programme actually measured, on the one model
where it was measured, and it is applied only to models served on a hosted API.

A cell whose replication lands inside its band PASSES. Outside, FAILS. A
precondition failure is VOID and is neither — because a broken measurement is not
evidence about the model.
"""

from __future__ import annotations

from seahaven.eden._shared import stats as S

#: The largest between-occasion shift measured in this programme: DS-V4-Flash's
#: under-duress rate moved 0.181 -> 0.500 between serving days, block 1 against
#: four later blocks, p = 2.1e-05. Mechanism unresolved; batch composition and
#: prefix cache were both ruled out and a deployment change is consistent.
OCCASION_COMPONENT = 0.319

#: Self-hosted serving with fixed batching is the only mode in which the
#: occasion component can actually be pinned, so it is not widened there.
HOSTED = True


def band(k: int, n: int, *, hosted: bool = HOSTED) -> tuple[float, float]:
    """The interval a replication must land in.

    Wilson on our n, then widened by the occasion component if hosted. Widening
    a 95% interval by a measured shift is deliberately conservative: it says a
    replication may differ from us by sampling error OR by the largest occasion
    effect we have seen, and still be a replication.
    """
    lo, hi = S.wilson_ci(k, n)
    if hosted:
        lo, hi = max(0.0, lo - OCCASION_COMPONENT), min(1.0, hi + OCCASION_COMPONENT)
    return (lo, hi)


def judge(k_new: int, n_new: int, k_ours: int, n_ours: int, *,
          hosted: bool = HOSTED) -> tuple[str, tuple[float, float], float]:
    """`(verdict, band, observed)`. PASS/FAIL only — VOID is decided upstream by
    the preconditions, because a broken cell is not evidence either way."""
    lo, hi = band(k_ours, n_ours, hosted=hosted)
    p = k_new / n_new if n_new else float("nan")
    return ("PASS" if lo <= p <= hi else "FAIL", (lo, hi), p)
