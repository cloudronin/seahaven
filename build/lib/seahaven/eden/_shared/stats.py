"""The statistics, once each — and named, because the semantics genuinely differ.

Before this module the repo held **ten implementations of one binomial confidence
interval across three semantics**, and two definitions of `mds` with
*incompatible signatures*. The three Wilson variants differ in exactly two places:

    variant                  n == 0 returns      z
    wilson_ci                (0.0, 1.0)          1.96
    wilson_ci_nan_empty      (nan, nan)          1.96
    wilson_ci_z_exact        (0.0, 1.0)          1.959963985

The arithmetic is otherwise algebraically identical — one form divides by `d`
inside `c` and `h`, the other at the end, and those are the same number.

**There is deliberately no bare `wilson` in `__all__`.** A caller reaching for the
undecorated name would get whichever semantics happened to land first, and the
divergence would be invisible — which is how the tenth implementation got written.
Pick a variant by name, or use `wilson_ci`, which is the recommended one for new
code and the semantics every round module already uses.
"""

from __future__ import annotations

import math

from scipy.stats import fisher_exact

__all__ = [
    "wilson_ci", "wilson_ci_nan_empty", "wilson_ci_z_exact",
    "fisher", "mds",
]

Z_95 = 1.96                 # what every eden round module uses
Z_95_EXACT = 1.959963985    # what seahaven/fidelity/score.py uses


def _wilson(k: int, n: int, z: float, empty: tuple[float, float]
            ) -> tuple[float, float]:
    if not n:
        return empty
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """**RECOMMENDED FOR NEW CODE.** Empty n returns the uninformative (0, 1).

    Behaves at 0 and 1, where the normal approximation does not — rates of
    exactly zero are common here and a symmetric interval would run below it.
    """
    return _wilson(k, n, z, (0.0, 1.0))


def wilson_ci_nan_empty(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Empty n returns `(nan, nan)` rather than `(0, 1)`.

    Use when an absent measurement must propagate as absent instead of as a
    maximally-wide interval that a reader could mistake for a result.
    """
    return _wilson(k, n, z, (float("nan"), float("nan")))


def wilson_ci_z_exact(k: int, n: int) -> tuple[float, float]:
    """z = 1.959963985 rather than 1.96. Preserved because the fidelity scorer
    was written with it and its committed numbers depend on it."""
    return _wilson(k, n, Z_95_EXACT, (0.0, 1.0))


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for two proportions."""
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def mds(n1: int, n2: int, p1: float, *, alpha: float) -> tuple[float, float]:
    """Smallest detectable shift down/up, **scanning OUTWARD from p1**.

    `alpha` is keyword-only with NO DEFAULT, and that is the fix for a real
    footgun: the two previous definitions disagreed on exactly this — round 8
    required it, round 9 defaulted it to 0.05 — so the same call could mean two
    different tests depending on which module you imported. Now every call site
    must say.

    Scanning outward matters and was got wrong once: scanning up from k2=0 finds
    the LARGEST significant downward gap and reports it as the smallest.
    Significance is monotone in |p2 - p1|, so the first hit walking away from p1
    is the boundary.
    """
    k1 = round(p1 * n1)
    mid = p1 * n2
    lo = hi = None
    for k2 in range(int(math.floor(mid)), -1, -1):
        if fisher(k1, n1, k2, n2) < alpha:
            lo = p1 - k2 / n2
            break
    for k2 in range(int(math.ceil(mid)), n2 + 1):
        if fisher(k1, n1, k2, n2) < alpha:
            hi = k2 / n2 - p1
            break
    return (lo if lo is not None else float("nan"),
            hi if hi is not None else float("nan"))
