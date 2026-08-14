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
