"""The E-level read's adapter and its inverted junk convention.

Two things here are easy to get wrong and would not announce themselves:

* the **direction** of the level pair — if the higher-pressure level landed in
  `after_ok`, every bend would still be a positive number and would still look
  fine, while meaning the opposite;
* the **role of the junk bin** — axis 1's correction strips `other`, and doing
  that here would delete the excursions this axis exists to measure.
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

# The one-off scripts moved to archive/ when the CLI absorbed the live
# family. Both roots are on the path so a test need not know which side
# of that move its subject landed on.
for _d in ("scripts", "archive"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _d))

from eaxis_read import as_level_pair  # noqa: E402
from smoke_state_conditioned import bend, bucket_items  # noqa: E402

warnings.filterwarnings("ignore")
LEGAL = ["go|direction", "look", "take|o", "examine|o"]


def eps(items, n=60, seed=1):
    r = random.Random(seed)
    return [[("lvl", r.choice(items)) for _ in range(20)] for _ in range(n)]


def test_the_lower_pressure_level_is_always_after_ok():
    """Direction matters and a swapped pair would still look like a number."""
    out = as_level_pair([[("E0", "look")]], [[("E3", "other")]])
    assert out[0][0][0] == "after_ok"
    assert out[1][0][0] == "after_fail"


def test_identical_level_distributions_give_a_near_zero_bend():
    e = as_level_pair(eps(LEGAL, seed=1), eps(LEGAL, seed=2))
    n = min(len(bucket_items(e, b)) for b in ("after_ok", "after_fail")) // 2
    assert bend(e, n, random.Random(7)) < 0.05


def test_an_excursion_heavy_level_produces_a_large_bend():
    """The E3-like case: the higher-pressure level emits out-of-vocabulary."""
    e = as_level_pair(eps(LEGAL, seed=1), eps(["other"] * 3 + LEGAL, seed=2))
    n = min(len(bucket_items(e, b)) for b in ("after_ok", "after_fail")) // 2
    assert bend(e, n, random.Random(7)) > 0.3


def test_stripping_junk_would_delete_the_axis_2_signal():
    """Why the legal-only read is a CONTROL here, not the clean read.

    Axis 1 stripped `other` as junk. On this axis `other` is where the excursion
    lives, so the same operation removes the measurement — the bend collapses to
    roughly nothing between two levels that differ only in excursions.
    """
    a, b = eps(LEGAL, seed=1), eps(["other"] * 3 + LEGAL, seed=2)
    full = as_level_pair(a, b)
    stripped = as_level_pair(
        [[(k, i) for (k, i) in e if i != "other"] for e in a],
        [[(k, i) for (k, i) in e if i != "other"] for e in b])
    n_f = min(len(bucket_items(full, x)) for x in ("after_ok", "after_fail")) // 2
    n_s = min(len(bucket_items(stripped, x)) for x in ("after_ok", "after_fail")) // 2
    assert bend(full, n_f, random.Random(7)) > 0.3
    assert bend(stripped, n_s, random.Random(7)) < 0.05


def test_the_break_out_command_bins_as_other():
    """The fact that makes the inversion above load-bearing."""
    from seahaven.fidelity.worldspec import load
    from smoke_state_conditioned import bin_of

    kinds = {a: b for a, b in load("world_ea").kinds}
    assert bin_of("unlock iron door with brass key", kinds) == "other"
    assert bin_of("go north", kinds) != "other"
