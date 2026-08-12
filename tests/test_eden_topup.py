"""The n=96 halves read, and the power column that got its direction wrong.

`detectable()` answers "what is the smallest half-to-half difference this test
could have called significant" — the admissible-range precheck applied to a
hypothesis test, so that a null is read against what the design could have found.
Its first version scanned k2 upward from zero and took the first significant hit
as the downward boundary. Significance is monotone in distance from p1, so that
finds the FARTHEST significant point below p1, not the nearest: cogito printed
"down 0.708" — only a collapse to exactly zero is detectable — where the true
figure is 0.250.

The direction of the error is what makes it worth a regression. It overstates how
insensitive the test is, so it makes a null look SAFER than it is: "the halves
agree and we could only have caught a total collapse anyway" reads as a much
weaker claim than the one actually available. A power figure that errs toward
modesty still misstates the design.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_topup_read", Path(__file__).resolve().parents[1] / "scripts"
    / "eden_topup_read.py")
_T = importlib.util.module_from_spec(_spec)
sys.modules["_topup_read"] = _T
_spec.loader.exec_module(_T)


@pytest.mark.parametrize("n1,n2,p1", [(24, 72, 17 / 24), (24, 72, 15 / 24),
                                      (24, 72, 10 / 24), (48, 48, 0.5)])
def test_detectable_returns_the_NEAREST_significant_gap_not_the_farthest(
        n1, n2, p1):
    """The boundary property, both directions: just outside is significant and
    one episode further in is not. Stated as a property rather than as expected
    numbers, so it holds if the alpha or the test ever changes."""
    k1 = round(p1 * n1)
    lo, hi = _T.detectable(n1, n2, p1)

    for delta in (lo, hi):
        if delta != delta:  # nan: no detectable gap on that side, nothing to check
            continue
        k_at = round((p1 - delta) * n2) if delta is lo else round((p1 + delta) * n2)
        assert _T.fisher(k1, n1, k_at, n2) < 0.05, "boundary is not significant"

    if lo == lo:
        k_in = round((p1 - lo) * n2) + 1          # one episode closer to p1
        assert _T.fisher(k1, n1, k_in, n2) >= 0.05, (
            "a SMALLER gap than the reported boundary is also significant — the "
            "scan is returning the farthest hit, not the nearest")
    if hi == hi:
        k_in = round((p1 + hi) * n2) - 1
        assert _T.fisher(k1, n1, k_in, n2) >= 0.05, (
            "a SMALLER gap than the reported boundary is also significant")


def test_a_half_against_itself_finds_no_difference_and_gemma_vs_cogito_does():
    """The instrument's own control, as code.

    A test that cannot return 'no difference' when handed identical inputs, or
    cannot return 'difference' across the widest contrast in the cohort, is not
    evidence either way — and the halves verdict is six nulls, which is exactly
    the shape a broken test produces.
    """
    assert _T.fisher(17, 24, 17, 24) > 0.99
    assert _T.fisher(5, 96, 68, 96) < 1e-10


def test_the_pooled_rate_is_the_SUM_over_96_not_the_mean_of_two_rates():
    """Fixed before the data existed, and it matters: averaging 0.083 and 0.042
    gives gemma 0.063, where the 96-episode rate is 0.052. The 24 and the 72 are
    not two estimates to be averaged, they are 96 episodes."""
    k1, n1, k2, n2 = 2, 24, 3, 72
    assert (k1 + k2) / (n1 + n2) == pytest.approx(0.0520833, abs=1e-6)
    assert (k1 / n1 + k2 / n2) / 2 == pytest.approx(0.0625, abs=1e-6)
