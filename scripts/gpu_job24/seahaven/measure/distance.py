"""Fingerprint distance.

A fingerprint is one distribution per probe slot. Distance between two
fingerprints is the mean over slots of the squared L2 distance between the
per-slot distributions.

Why squared L2 rather than Jensen-Shannon or total variation:

  - It has an **unbiased U-statistic estimator** under sampling, which the
    divergence measures do not. That matters because the plug-in bias is larger
    for within-run than across-seed comparisons and therefore does not cancel in
    the ratio the kill criteria test.
  - It is bounded in [0, 2] per slot regardless of how many options the slot has,
    so slots with 2 options and slots with 4 stay comparable.
  - It degrades gracefully when a slot is near-deterministic, where JS and
    chi-square get numerically awkward.

Negative estimates are kept. Under the unbiased estimator a true distance of zero
produces negative values half the time, and clipping them at zero reintroduces
exactly the upward bias the estimator exists to remove — and decalibrates the
permutation null on top of that.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Mapping, Sequence

__all__ = ["Fingerprint", "slot_distance", "fingerprint_distance", "unbiased_slot_distance"]


@dc.dataclass(frozen=True)
class Fingerprint:
    """One distribution per probe slot."""

    slots: Mapping[str, tuple[float, ...]]

    def shared_slots(self, other: "Fingerprint") -> list[str]:
        return sorted(set(self.slots) & set(other.slots))

    def as_dict(self) -> dict:
        return {k: [round(p, 6) for p in v] for k, v in sorted(self.slots.items())}


def slot_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Squared L2 between two distributions over the same option set."""
    if len(p) != len(q):
        raise ValueError(f"option-set mismatch: {len(p)} vs {len(q)}")
    return sum((a - b) ** 2 for a, b in zip(p, q))


def unbiased_slot_distance(
    counts_p: Sequence[int], counts_q: Sequence[int]
) -> float:
    """Unbiased estimator of squared L2 from multinomial counts.

    For sampled (Tier-B) slots only. Exact-scored slots have no sampling noise
    and use `slot_distance` directly.

    For each option i, E[(p_i - q_i)^2] is estimated by

        p̂_i^2 - p̂_i(1-p̂_i)/(n_p - 1)  +  q̂_i^2 - q̂_i(1-q̂_i)/(n_q - 1)  -  2 p̂_i q̂_i

    where the subtracted terms remove the variance that the plug-in square
    otherwise inherits. Can return a negative value; that is correct and must be
    kept.
    """
    n_p, n_q = sum(counts_p), sum(counts_q)
    if n_p < 2 or n_q < 2:
        raise ValueError("unbiased estimation needs at least 2 draws per side")

    total = 0.0
    for c_p, c_q in zip(counts_p, counts_q):
        p_hat, q_hat = c_p / n_p, c_q / n_q
        p_sq = p_hat * p_hat - p_hat * (1 - p_hat) / (n_p - 1)
        q_sq = q_hat * q_hat - q_hat * (1 - q_hat) / (n_q - 1)
        total += p_sq + q_sq - 2 * p_hat * q_hat
    return total


def fingerprint_distance(a: Fingerprint, b: Fingerprint) -> float:
    """Mean per-slot squared L2 over slots present in both.

    Equal weighting across slots is the pre-registered primary. Inverse-variance
    weighting by base-checkpoint entropy is a secondary analysis and lives
    elsewhere, because choosing weights from the data being measured is a
    researcher degree of freedom that has to be declared.
    """
    shared = a.shared_slots(b)
    if not shared:
        raise ValueError("fingerprints share no slots")
    return sum(slot_distance(a.slots[s], b.slots[s]) for s in shared) / len(shared)
