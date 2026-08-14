"""`_shared.stats` must reproduce every call site's ORIGINAL behaviour.

Ten implementations of one binomial CI collapsed into three named variants. That
is only safe if each variant reproduces the semantics it replaced, so the old
implementations are carried here verbatim as reference and compared over a grid.
Semantics preserved by test, not by inspection.
"""

from __future__ import annotations

import importlib
import math

import pytest

from seahaven.eden._shared import stats as S

# k <= n only. k > n is not a rate — p*(1-p) goes negative and the variants
# diverge on how they fail, which is noise about invalid input rather than a
# semantic difference worth pinning. Caught by the grid itself at (2, 1).
GRID = [(k, n) for n in (0, 1, 2, 5, 24, 32, 48, 71, 72, 96, 144, 192, 312)
        for k in sorted({0, 1, 2, n // 4, n // 3, n // 2, max(0, n - 1), n})
        if k <= n]


# --- the three originals, verbatim -----------------------------------------

def _orig_round_module(k, n, z=1.96):
    """seahaven/eden/round{2..13}.py — the form every round uses."""
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _orig_c3_gate(k, n, z=1.96):
    """archive/c3_stage1_gate.py — nan on empty."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _orig_fidelity_score(k, n, z=1.959963985):
    """seahaven/fidelity/score.py — divides by d at the END, and a different z."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _same(a, b):
    return all((math.isnan(x) and math.isnan(y)) or abs(x - y) < 1e-12
               for x, y in zip(a, b))


@pytest.mark.parametrize("k,n", GRID)
def test_wilson_ci_reproduces_the_ROUND_MODULE_semantics(k, n):
    assert _same(S.wilson_ci(k, n), _orig_round_module(k, n))


@pytest.mark.parametrize("k,n", GRID)
def test_wilson_ci_nan_empty_reproduces_the_C3_GATE_semantics(k, n):
    assert _same(S.wilson_ci_nan_empty(k, n), _orig_c3_gate(k, n))


@pytest.mark.parametrize("k,n", GRID)
def test_wilson_ci_z_exact_reproduces_the_FIDELITY_SCORER_semantics(k, n):
    assert _same(S.wilson_ci_z_exact(k, n), _orig_fidelity_score(k, n))


def test_the_three_variants_are_NOT_interchangeable():
    """If they agreed everywhere, one name would do. They do not: the empty case
    and the z differ, and both differences are load-bearing somewhere."""
    assert S.wilson_ci(0, 0) == (0.0, 1.0)
    assert all(math.isnan(x) for x in S.wilson_ci_nan_empty(0, 0))
    assert S.wilson_ci(12, 96) != S.wilson_ci_z_exact(12, 96)


def test_NO_bare_wilson_is_exported():
    """The undecorated name is how the tenth implementation got written."""
    assert "wilson" not in S.__all__
    assert not hasattr(S, "wilson")


#: The round modules that actually define `wilson`. **round2 does not** — it was
#: written before the helper existed, and assuming otherwise is what the first
#: version of this test did.
@pytest.mark.parametrize("name", ["round6", "round7", "round8", "round9",
                                  "round10", "round11", "round12", "round13"])
@pytest.mark.parametrize("k,n", [(0, 96), (1, 32), (12, 96), (58, 96), (7, 71)])
def test_every_LIVE_round_module_agrees_with_wilson_ci(name, k, n):
    """The call sites themselves, not just a reference copy of them."""
    m = importlib.import_module(f"seahaven.eden.{name}")
    assert _same(m.wilson(k, n), S.wilson_ci(k, n))


def test_mds_requires_alpha_and_reproduces_BOTH_old_call_sites():
    """round 8 required `alpha`, round 9 defaulted it to 0.05 — the same call
    meant two different tests depending on the import. Now it must be stated."""
    from seahaven.eden import round9 as R9
    with pytest.raises(TypeError):
        S.mds(72, 24, 0.611)                      # no default: must be given
    assert _same(S.mds(72, 24, 0.611, alpha=0.05), R9.mds(72, 24, 0.611))
    assert _same(S.mds(96, 24, 0.375, alpha=0.05), R9.mds(96, 24, 0.375))


@pytest.mark.parametrize("k1,n1,k2,n2", [(0, 96, 5, 96), (27, 72, 19, 24),
                                         (13, 72, 12, 24), (58, 96, 46, 96)])
def test_fisher_matches_every_round_modules_copy(k1, n1, k2, n2):
    from seahaven.eden import round10 as R10
    from seahaven.eden import round13 as R13
    want = S.fisher(k1, n1, k2, n2)
    assert R10._fisher(k1, n1, k2, n2) == want
    assert R13._fisher(k1, n1, k2, n2) == want


# --- rank correlation: the first live one in the programme -------------------

@pytest.mark.parametrize("x,y", [
    ([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]),
    ([100, 100, 100, 86, 82, 55, 29, 23, 91], [5, 3, 9, 1, 7, 2, 8, 4, 6]),
    ([1, 1, 2, 2, 3, 3], [6, 5, 4, 3, 2, 1]),
    ([0.0, 0.0, 0.0, 0.5, 1.0], [1, 2, 3, 4, 5]),
])
def test_spearman_matches_scipy_INCLUDING_ties(x, y):
    """Ties are not an edge case here: a third of the veto-hold cohort sits at
    exactly 100.0, so midrank handling decides whether the coefficient is even
    defined. Checked against scipy rather than against intuition."""
    from scipy.stats import spearmanr
    assert abs(S.spearman_rho(x, y) - spearmanr(x, y).statistic) < 1e-12


def test_spearman_is_NAN_when_one_side_is_constant():
    """Three models at exactly 100.0 is a real corpus state. A constant vector
    has no ranks to correlate, and nan propagates it as absent rather than as 0
    — which a reader would take for 'uncorrelated'."""
    import math
    assert math.isnan(S.spearman_rho([1.0, 1.0, 1.0, 1.0], [1, 2, 3, 4]))


def test_permutation_p_is_DETERMINISTIC_under_a_fixed_seed():
    """**Why the seed exists at all.** `verify` recomputes every register figure
    and compares with `==`. A sampled p-value that moved between runs could not
    be a claim, so the seed is required and lives in the round pin."""
    a = S.permutation_p([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], n_shuffles=500, seed=7)
    b = S.permutation_p([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], n_shuffles=500, seed=7)
    c = S.permutation_p([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], n_shuffles=500, seed=8)
    assert a == b
    assert a != c, "a different seed must actually resample"


def test_permutation_p_is_NEVER_ZERO():
    """Davison-Hinkley: the observed statistic is one of the permutations, so
    p=0 from a finite sample would claim more than the sampling supports."""
    p = S.permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                        n_shuffles=200, seed=1)
    assert p > 0 and p == pytest.approx(1 / 201, abs=1e-12)


def test_n_shuffles_and_seed_are_REQUIRED_like_mds_alpha():
    """A parameter that changes what the test means may not have a silent
    default — the rule `mds(alpha=)` was written to enforce."""
    with pytest.raises(TypeError):
        S.permutation_p([1, 2, 3], [3, 2, 1], n_shuffles=10)
    with pytest.raises(TypeError):
        S.permutation_p([1, 2, 3], [3, 2, 1], seed=1)


def test_spearman_ci_is_what_stops_an_UNDERPOWERED_NULL(capsys):
    """**The MDS-class error, prevented rather than repeated.**

    At n=17 an observed rho of 0 carries a 95% CI of about [-0.48, +0.48]. The
    spec read |rho| < 0.5 as evidence the index lacks the dimension — a null
    claimed from an interval containing 0.45.
    """
    lo, hi = S.spearman_ci(0.0, 17)
    assert lo < -0.47 and hi > 0.47
    assert lo < 0.45 < hi, "0.45 sits inside the interval, which is the point"
    # and the interval narrows with n, as it must
    assert S.spearman_ci(0.0, 40)[1] < S.spearman_ci(0.0, 17)[1]


def test_the_E1_VERDICT_WINDOW_at_the_achievable_n():
    """**Measured before spending, and it constrains what E1 may claim.**

    n=17 is a hard ceiling: only 17 raidex rows have a Together string at 9/9.
    At that n the CI-form test needs |rho| <= ~0.03 to call a dimension
    non-redundant, and rho >= ~0.79 to call it redundant. Ordinary correlations
    in between return NEITHER verdict, so E1 ships coefficients and an explicit
    'underpowered' label rather than a verdict.
    """
    not_redundant = max((r / 1000 for r in range(0, 500)
                         if S.spearman_ci(r / 1000, 17)[1] < 0.5), default=None)
    redundant = min((r / 1000 for r in range(500, 1000)
                     if S.spearman_ci(r / 1000, 17)[0] > 0.5), default=None)
    assert not_redundant is not None and not_redundant < 0.05
    assert redundant is not None and redundant > 0.75
    # the spec's 0.7 threshold does not reach a verdict at this n
    assert S.spearman_ci(0.7, 17)[0] < 0.5
