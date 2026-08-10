"""Witnesses for the state-conditioned bend machinery.

**Every number in this probe is a distance, and distances fail silently.** A
wrong bin mapping, a mis-ordered distribution vector, or a null computed at the
wrong sample size all produce a plausible float. So each piece is pinned against
a case whose answer is arithmetic, on synthetic data, before the machinery ever
touches the corpus.

The two that matter most are the pair that keeps the read honest in **both**
directions: the null must be calibrated enough that identical generators come
back DEAD, and the threshold must be loose enough that genuinely different
generators come back NOT-DEAD. A probe that can only return one of the two
answers is not a test.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_state_conditioned import (BINS, bend, bin_of,  # noqa: E402
                                     bucketize, distribution, self_split_null,
                                     tvd)

KINDS = {"o": ("brass key", "coil of rope"), "c": ("crate",), "s": ("bench",)}


def episodes_from(mix_ok, mix_fail, n_eps=400, per_ep=15, seed=0):
    """Synthetic episodes drawing each bucket's bins from a given mix."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_eps):
        ep = []
        for _ in range(per_ep):
            bucket = "after_ok" if rng.random() < 0.7 else "after_fail"
            mix = mix_ok if bucket == "after_ok" else mix_fail
            ep.append((bucket, rng.choices(list(mix), weights=list(mix.values()))[0]))
        out.append(ep)
    return out


FLAT = {"go|direction": 3, "look": 2, "examine|o": 2, "take|o": 2, "other": 1}
SKEWED = {"go|direction": 1, "look": 1, "examine|o": 1, "take|o": 1, "other": 6}


# --- the distance itself -----------------------------------------------------

def test_tvd_identity_and_disjoint():
    p = [0.5, 0.3, 0.2] + [0.0] * (len(BINS) - 3)
    assert tvd(p, p) == 0.0
    q = [0.0] * (len(BINS) - 3) + [0.5, 0.3, 0.2]
    assert tvd(p, q) == pytest.approx(1.0)


def test_distribution_is_normalised_and_ordered():
    d = distribution(["look", "look", "go|direction", "other"])
    assert sum(d) == pytest.approx(1.0)
    assert d[BINS.index("look")] == pytest.approx(0.5)
    assert d[BINS.index("other")] == pytest.approx(0.25)


def test_legal_only_drops_other_and_renormalises():
    items = ["look"] * 3 + ["other"] * 7
    full = distribution(items)
    legal = distribution(items, legal_only=True)
    assert full[BINS.index("other")] == pytest.approx(0.7)
    assert sum(legal) == pytest.approx(1.0)
    assert legal[BINS.index("look")] == pytest.approx(1.0), \
        "with `other` removed, `look` is the whole distribution"


# --- the vocabulary ----------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("go north", "go|direction"),
    ("look", "look"),
    ("inventory", "inventory"),
    ("examine brass key", "examine|o"),
    ("examine crate", "examine|c"),
    ("examine bench", "examine|s"),
    ("take coil of rope", "take|o"),
    ("open crate", "open|c"),
    ("xyzzy", "other"),
    ("**note:", "other"),
    ("go sideways", "legal|unresolved"),      # legal verb, no direction
    ("take crate", "legal|unresolved"),       # legal verb, disallowed kind
    ("examine nothing", "legal|unresolved"),  # legal verb, unresolvable object
])
def test_bin_of(command, expected):
    assert bin_of(command, KINDS) == expected


def test_bin_of_reads_the_command_not_the_stored_verb():
    """The stored `verb` field is the raw first token; binning must normalise.

    378 distinct raw verbs appear across the corpus, and vocabulary size tracks
    junk rate — so binning on the raw field would make bend measure how much
    garbage a model emits.
    """
    assert bin_of("Go North.", KINDS) == "go|direction"
    assert bin_of("  TAKE brass key ", KINDS) == "take|o"


# --- the conditioning structure ---------------------------------------------

def test_bucketize_drops_first_step_and_keys_on_the_previous():
    cmds = [{"command": "look", "ok": False},      # step 0: dropped
            {"command": "go north", "ok": True},   # keyed by step 0 -> after_fail
            {"command": "look", "ok": True},       # keyed by step 1 -> after_ok
            {"command": "take crate", "ok": False}]  # keyed by step 2 -> after_ok
    got = bucketize(cmds, KINDS)
    assert got == [("after_fail", "go|direction"),
                   ("after_ok", "look"),
                   ("after_ok", "legal|unresolved")]
    assert len(got) == len(cmds) - 1


def test_bucketize_on_a_single_step_episode_is_empty():
    assert bucketize([{"command": "look", "ok": True}], KINDS) == []


# --- the null, in both directions -------------------------------------------

def test_null_is_a_distribution_not_a_point():
    eps = episodes_from(FLAT, FLAT, seed=1)
    null = self_split_null(eps, 200, random.Random(0))
    assert null["draws"] >= 100, "the null must be many draws, not one"
    assert null["p95"] >= null["median"] > 0.0


def test_identical_generators_land_INSIDE_the_null():
    """DEAD must be able to fire. Same generator for both buckets -> no bend."""
    eps = episodes_from(FLAT, FLAT, seed=2)
    b = bend(eps, 200, random.Random(0))
    null = self_split_null(eps, 200, random.Random(0))
    assert b <= null["p95"], (
        f"bend {b:.4f} exceeded the null's p95 {null['p95']:.4f} on identical "
        f"generators — the null is miscalibrated and everything would read "
        f"NOT-DEAD")


def test_different_generators_land_ABOVE_the_null():
    """NOT-DEAD must be able to fire, or the probe can only ever say DEAD."""
    eps = episodes_from(FLAT, SKEWED, seed=3)
    b = bend(eps, 200, random.Random(0))
    null = self_split_null(eps, 200, random.Random(0))
    assert b > null["p95"], (
        f"bend {b:.4f} did not exceed the null's p95 {null['p95']:.4f} on "
        f"deliberately different buckets — the probe cannot detect signal")


# --- the size-bias correction, which is why common-n exists -----------------

def test_common_n_removes_the_bucket_size_bias():
    """Two models with IDENTICAL behaviour but very different bucket sizes.

    TVD is upward-biased at small n (0.083 at n=292 against 0.029 at n=2485 in
    this corpus), and failure rates span 5.3% to 45.6%. Without subsampling to a
    common n, the model that fails rarely would appear to bend more — the
    comparison would partly measure failure rate. Both bends here must land
    inside the null, and their gap must be small.
    """
    small = episodes_from(FLAT, FLAT, n_eps=120, per_ep=8, seed=4)
    large = episodes_from(FLAT, FLAT, n_eps=900, per_ep=20, seed=5)
    n = 150
    b_small = bend(small, n, random.Random(0))
    b_large = bend(large, n, random.Random(0))
    null = max(self_split_null(small, n, random.Random(0))["p95"],
               self_split_null(large, n, random.Random(0))["p95"])
    assert abs(b_small - b_large) <= null, (
        f"identical behaviour at different bucket sizes produced a gap "
        f"{abs(b_small - b_large):.4f} above the null {null:.4f} — the "
        f"size-bias correction is not working")


def test_bend_refuses_when_a_bucket_is_smaller_than_n():
    """Silently sampling fewer than `n` would reintroduce the bias it removes."""
    eps = episodes_from(FLAT, FLAT, n_eps=10, per_ep=4, seed=6)
    got = bend(eps, 10_000, random.Random(0))
    assert got != got, "expected NaN when a bucket cannot supply n"
