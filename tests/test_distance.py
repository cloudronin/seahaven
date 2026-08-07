"""Distance and estimator tests.

The unbiased-estimator tests are the important ones. The plug-in estimator's bias
is what would fire kill criterion K2 on a live project, and the bias is larger for
the within-run comparison than the across-seed one, so it does not cancel in the
ratio. These check that the correction actually works and that negative estimates
survive.
"""

from __future__ import annotations

import random

import pytest

from seahaven.measure.distance import (
    Fingerprint,
    fingerprint_distance,
    slot_distance,
    unbiased_slot_distance,
)
from seahaven.measure.exact_score import softmax


class TestSlotDistance:
    def test_identical_is_zero(self):
        assert slot_distance([0.3, 0.7], [0.3, 0.7]) == 0.0

    def test_disjoint_extremes_is_two(self):
        assert slot_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(2.0)

    def test_symmetric(self):
        p, q = [0.2, 0.5, 0.3], [0.6, 0.1, 0.3]
        assert slot_distance(p, q) == pytest.approx(slot_distance(q, p))

    def test_bounded_regardless_of_option_count(self):
        for k in (2, 3, 4, 8):
            p = [1.0] + [0.0] * (k - 1)
            q = [0.0] * (k - 1) + [1.0]
            assert 0 <= slot_distance(p, q) <= 2.0 + 1e-9

    def test_mismatched_option_sets_raise(self):
        with pytest.raises(ValueError, match="option-set mismatch"):
            slot_distance([0.5, 0.5], [0.3, 0.3, 0.4])


class TestUnbiasedEstimator:
    """The correction that keeps K2 from firing spuriously."""

    def _draw(self, probs, n, rng):
        """Multinomial sample of size exactly n.

        Written out rather than using random.choices so the total is verifiable:
        an earlier version put the float-safety `else` on the outer loop instead
        of the inner one, which silently returned n+1 draws with the extra one
        always in the last option — enough to make a correct estimator look
        biased by ~0.05.
        """
        counts = [0] * len(probs)
        for _ in range(n):
            r, acc = rng.random(), 0.0
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    counts[i] += 1
                    break
            else:  # pragma: no cover - float dust only
                counts[-1] += 1
        assert sum(counts) == n
        return counts

    def test_unbiased_at_true_zero(self):
        """Two samples from the SAME distribution must average to ~0, not above."""
        rng = random.Random(0)
        truth = [0.5, 0.5]
        estimates = [
            unbiased_slot_distance(self._draw(truth, 20, rng), self._draw(truth, 20, rng))
            for _ in range(3000)
        ]
        assert abs(sum(estimates) / len(estimates)) < 0.01

    def test_plugin_is_biased_upward_where_unbiased_is_not(self):
        """Demonstrates the failure the estimator exists to prevent."""
        rng = random.Random(1)
        truth = [0.5, 0.5]
        plugin, unbiased = [], []
        for _ in range(3000):
            a, b = self._draw(truth, 20, rng), self._draw(truth, 20, rng)
            pa = [c / sum(a) for c in a]
            pb = [c / sum(b) for c in b]
            plugin.append(slot_distance(pa, pb))
            unbiased.append(unbiased_slot_distance(a, b))
        mean_plugin = sum(plugin) / len(plugin)
        mean_unbiased = sum(unbiased) / len(unbiased)
        assert mean_plugin > 0.04, "plug-in should be visibly biased at K=20"
        assert abs(mean_unbiased) < 0.01
        assert mean_plugin > mean_unbiased

    def test_bias_grows_as_k_shrinks(self):
        """Why K matters, and why a small K is dangerous rather than merely noisy."""
        rng = random.Random(2)
        truth = [0.5, 0.5]
        means = {}
        for k in (5, 20, 80):
            vals = []
            for _ in range(2000):
                a, b = self._draw(truth, k, rng), self._draw(truth, k, rng)
                vals.append(slot_distance([c / k for c in a], [c / k for c in b]))
            means[k] = sum(vals) / len(vals)
        assert means[5] > means[20] > means[80]

    def test_negative_estimates_are_produced_and_kept(self):
        rng = random.Random(3)
        truth = [0.5, 0.5]
        vals = [
            unbiased_slot_distance(self._draw(truth, 10, rng), self._draw(truth, 10, rng))
            for _ in range(500)
        ]
        assert any(v < 0 for v in vals), "clipping would reintroduce the bias"

    def test_recovers_a_real_difference(self):
        rng = random.Random(4)
        p, q = [0.9, 0.1], [0.1, 0.9]
        vals = [
            unbiased_slot_distance(self._draw(p, 50, rng), self._draw(q, 50, rng))
            for _ in range(500)
        ]
        assert sum(vals) / len(vals) == pytest.approx(slot_distance(p, q), abs=0.05)

    def test_too_few_draws_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            unbiased_slot_distance([1, 0], [0, 1])


class TestFingerprint:
    def test_identical_fingerprints_are_zero(self):
        fp = Fingerprint({"a": (0.5, 0.5), "b": (0.2, 0.8)})
        assert fingerprint_distance(fp, fp) == 0.0

    def test_averages_over_shared_slots(self):
        a = Fingerprint({"x": (1.0, 0.0), "y": (0.5, 0.5)})
        b = Fingerprint({"x": (0.0, 1.0), "y": (0.5, 0.5)})
        assert fingerprint_distance(a, b) == pytest.approx(1.0)

    def test_only_shared_slots_count(self):
        a = Fingerprint({"x": (1.0, 0.0), "only_a": (1.0, 0.0)})
        b = Fingerprint({"x": (0.0, 1.0), "only_b": (1.0, 0.0)})
        assert a.shared_slots(b) == ["x"]
        assert fingerprint_distance(a, b) == pytest.approx(2.0)

    def test_no_shared_slots_raises(self):
        a = Fingerprint({"x": (1.0, 0.0)})
        b = Fingerprint({"y": (1.0, 0.0)})
        with pytest.raises(ValueError, match="share no slots"):
            fingerprint_distance(a, b)


class TestSoftmax:
    def test_sums_to_one(self):
        assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_order_preserved(self):
        out = softmax([-5.0, 0.0, 2.0])
        assert out[0] < out[1] < out[2]

    def test_stable_with_large_negatives(self):
        """Logprob sums are large and negative; naive exp would underflow."""
        out = softmax([-4000.0, -4001.0])
        assert sum(out) == pytest.approx(1.0)
        assert out[0] > out[1]

    def test_equal_inputs_are_uniform(self):
        assert softmax([3.0, 3.0, 3.0]) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
