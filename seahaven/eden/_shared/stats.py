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
import random

from scipy.stats import fisher_exact

__all__ = [
    "wilson_ci", "wilson_ci_nan_empty", "wilson_ci_z_exact",
    "fisher", "mds",
    "spearman_rho", "spearman_ci", "permutation_p", "n_ties",
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


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, **midranks for ties**.

    The first live rank correlation in the programme — six hand-rolled copies sit
    in `archive/` and `scipy.stats.spearmanr` was called inline in one script.

    **Ties are load-bearing here, not an edge case.** A third of the veto-hold
    cohort sits at exactly 100.0 (three models at 0/144), and midranks are what
    makes the coefficient defined at all when a block of scores is identical.
    They also DEPRESS the coefficient, which is why `emit correlations` prints
    the tie count beside every rho: a low rho on a tied cohort and a low rho on a
    spread one are different claims.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 3:
        raise ValueError("rank correlation needs at least 3 pairs")
    a, b = _midranks(x), _midranks(y)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a)
                    * sum((q - mb) ** 2 for q in b))
    if den == 0:                     # one side is entirely constant
        return float("nan")
    return num / den


def _midranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for t in range(i, j + 1):
            out[order[t]] = shared
        i = j + 1
    return out


def n_ties(v: list[float]) -> int:
    """How many values are not unique. Reported beside every coefficient."""
    return len(v) - len(set(v))


def permutation_p(x: list[float], y: list[float], *, n_shuffles: int,
                  seed: int) -> float:
    """Two-sided permutation p for `spearman_rho`, by sampling.

    **`n_shuffles` and `seed` are required, for the reason `mds`'s `alpha` is:**
    a parameter that changes what the test means may not have a silent default.

    **Sampled, not exact — and the spec asked for exact.** At the achievable
    n=17 there are 17! ~ 3.6e14 permutations, so enumeration is not available at
    any budget. Sampling makes the p-value a random variable, which a claims-
    register figure cannot tolerate: `verify` recomputes every figure and
    compares with `==`. Hence the seed, which makes this deterministic and
    therefore registrable, and which is frozen in the round pin alongside
    `n_shuffles`.

    The +1s are Davison-Hinkley: the observed statistic is itself one of the
    permutations, so a sampled p can never be 0 — reporting p=0 from 10,000
    shuffles would claim more than the sampling can support.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")
    obs = abs(spearman_rho(x, y))
    if math.isnan(obs):
        return float("nan")
    rng = random.Random(seed)
    shuffled = list(y)
    hits = 0
    for _ in range(n_shuffles):
        rng.shuffle(shuffled)
        r = spearman_rho(x, shuffled)
        hits += (not math.isnan(r)) and abs(r) >= obs
    return (hits + 1) / (n_shuffles + 1)


def spearman_ci(rho: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Fisher-z interval for a rank correlation.

    **This is the piece that stops an underpowered null being read as a
    finding.** At n=17 an observed rho of 0 carries a 95% CI of about
    [-0.48, +0.48], so "|rho| < 0.5, therefore the index lacks this dimension"
    claims a null from an interval containing 0.45. The programme has made that
    exact error once already, on the A-vs-B MDS, and it is in the corrections
    ledger. Report the interval; claim only what excludes the threshold.
    """
    if n < 4 or math.isnan(rho):
        return (float("nan"), float("nan"))
    r = min(max(rho, -0.999999), 0.999999)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.atanh(r) - z * se, math.atanh(r) + z * se
    return (math.tanh(lo), math.tanh(hi))


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
