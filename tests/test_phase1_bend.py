"""Witnesses for the Phase 1 bend instrument, and for the defect it corrects.

**The defect these tests exist for was inherited, not invented.** The pinned
`smoke_state_conditioned.bend` / `self_split_null` pass `legal_only` down into
`distribution`, which drops the `other` bin *after* `rng.sample(items, n)`. So
the legal-only read never ran at the declared `n` — it ran at
`n x (1 - junk_rate)`, and the junk rate is a property of the model. Across the
exploration set that rate spans 0.2% to 100%, so a single declared "n=600" meant
n=595 for `gemma-2-9b-it` and n=32 for `Qwen3-8B-Base`.

TVD is upward-biased at small n, so this manufactured exactly the pattern that
looked like signal: Spearman(effective n, legal-only null p95) = **-0.833** over
14 models, and the three most inflated nulls were the three junk-heaviest models.

The first two tests below are the load-bearing pair: one proves the defect is
real and directional, the other proves `strip_junk` removes it. The third proves
the correction is *free* — that filtering first and calling the pinned functions
with `legal_only=False` computes the identical statistic — which is why the
pinned smoke-test module is left untouched rather than edited under a result.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# The one-off scripts moved to archive/ when the CLI absorbed the live
# family. Both roots are on the path so a test need not know which side
# of that move its subject landed on.
for _d in ("scripts", "archive"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _d))

from phase1_bend import strip_junk  # noqa: E402
from smoke_state_conditioned import (bend, bucket_items,  # noqa: E402
                                     distribution, self_split_null, tvd)

LEGAL = ("go|direction", "look", "examine|o", "take|o", "open|c", "drop|o")


def episodes(junk: float, per_bucket: int = 800, seed: int = 3):
    """Episodes whose legal commands are identical regardless of the junk rate.

    `junk` is junk items PER legal item, so a rate above 1 is reachable — the
    real corpus has models at 94.7% junk, which no per-item probability can
    reproduce. Holding the legal content fixed is what makes the comparison
    clean: any movement in the legal-only read across junk rates is instrument,
    not data.
    """
    rng = random.Random(seed)
    eps = []
    for _ in range(40):
        ep = []
        for bucket in ("after_ok", "after_fail"):
            for k in range(per_bucket // 40):
                ep.append((bucket, LEGAL[(k + (bucket == "after_fail")) % len(LEGAL)]))
                extra = int(junk) + (1 if rng.random() < junk - int(junk) else 0)
                ep.extend((bucket, "other") for _ in range(extra))
        eps.append(ep)
    return eps


def effective_n(eps, n, legal_only):
    """How many items the legal-only read actually gets after the drop."""
    rng = random.Random(0)
    items = rng.sample(bucket_items(eps, "after_ok"), n)
    return len([i for i in items if i != "other"]) if legal_only else len(items)


def test_the_pinned_path_silently_shrinks_n_with_the_junk_rate():
    """The defect, as a witness. See the module docstring.

    This is deliberately a test of the ORIGINAL behaviour: it must keep passing,
    because `smoke_state_conditioned` is pinned and must not be edited under a
    published result. What changes is that Phase 1 no longer uses this path.
    """
    n = 300
    light, heavy = episodes(0.02), episodes(6.0)  # 2% vs 86% junk
    assert effective_n(light, n, legal_only=True) > 0.9 * n
    assert effective_n(heavy, n, legal_only=True) < 0.4 * n, (
        "the junk rate must be able to shrink the legal-only sample — if this "
        "fails the fixture no longer reproduces the defect")


def test_the_defect_inflates_the_null_in_the_junk_heavy_direction():
    """Small effective n inflates TVD, so junk-heavy models get a bigger null.

    That is the mechanism behind the -0.833 correlation in the real data: the
    inflation is not noise, it is monotone in the junk rate.
    """
    n = 300
    p_light = self_split_null(episodes(0.02), n, random.Random(1), True)["p95"]
    p_heavy = self_split_null(episodes(6.0), n, random.Random(1), True)["p95"]
    assert p_heavy > p_light, (p_light, p_heavy)


def test_strip_junk_honours_the_declared_n_at_every_junk_rate():
    """The fix. The sample is drawn FROM the legal pool, so effective n == n."""
    n = 300
    for rate in (0.02, 0.5, 2.0, 6.0):
        eps = strip_junk(episodes(rate))
        assert effective_n(eps, n, legal_only=False) == n
        assert "other" not in bucket_items(eps, "after_ok")


def test_strip_junk_null_is_stable_across_junk_rates():
    """The inflation is gone: same legal content, same null, whatever the junk.

    This is the test that would have caught the defect had it existed first.
    """
    n = 300
    p = [self_split_null(strip_junk(episodes(r)), n, random.Random(1), False)["p95"]
         for r in (0.02, 0.5, 2.0, 6.0)]
    assert max(p) - min(p) < 0.02, p


def test_filtering_first_computes_the_identical_statistic():
    """Why the pinned module is left untouched rather than edited.

    On junk-free episodes the `other` bin is zero in BOTH distributions, so it
    contributes nothing to the total variation: the 12-bin read equals the
    11-bin read exactly. The correction is therefore a change of *where* the
    sample is drawn, not of what is computed.
    """
    eps = strip_junk(episodes(2.0))
    for bucket in ("after_ok", "after_fail"):
        items = bucket_items(eps, bucket)[:200]
        full = distribution(items, legal_only=False)
        legal = distribution(items, legal_only=True)
        assert full[-1] == 0.0, "the other bin must be empty after stripping"
        assert abs(sum(full) - 1.0) < 1e-12
        assert tvd(full[:-1], legal) == 0.0

    a, b = bucket_items(eps, "after_ok")[:200], bucket_items(eps, "after_fail")[:200]
    assert abs(tvd(distribution(a, False), distribution(b, False))
               - tvd(distribution(a, True), distribution(b, True))) < 1e-12


def test_a_self_split_null_requires_two_disjoint_draws():
    """Why Phase 1 caps n at min_bucket // 2 rather than min_bucket.

    At n = min_bucket the null cannot be estimated from the smaller bucket at
    all, and `Mistral-7B-Instruct-v0.3` lost its null entirely that way. The cap
    makes the null a guarantee instead of an accident of bucket size.
    """
    eps = episodes(0.0, per_bucket=800)
    m = min(len(bucket_items(eps, b)) for b in ("after_ok", "after_fail"))
    assert self_split_null(eps, m, random.Random(1), False)["draws"] == 0
    assert self_split_null(eps, m // 2, random.Random(1), False)["draws"] > 0


def test_bend_is_seed_dependent_so_findings_must_average():
    """The per-model junk direction was seed-luck; this is why.

    A single-seed bend is a draw, not a quantity. Phase 1's robustness read
    averages over twelve seeds before bootstrapping over models.
    """
    eps = episodes(0.3)
    vals = {round(bend(eps, 150, random.Random(s)), 6) for s in range(8)}
    assert len(vals) > 1, "a single seed would be reported as if it were exact"
