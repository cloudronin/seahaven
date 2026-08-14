"""ROUND 14 — the LAT2 boundary, and the acceptance check that had to be replaced.

Two halves. The $0 half is here: LAT2's served text, LAT's untouched record, and
the arithmetic showing that the acceptance check as originally specified could
not have failed. The served half is the cogito cell.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from seahaven.eden import round14 as R
from seahaven.eden._shared import corpus as C

_ROOT = Path(__file__).resolve().parents[1]


def _walk(level: str) -> str:
    """Every observation the ENGINE emits on a tour of the world.

    This is the served text, not the lock and not the builder source. The
    defect LAT2 exists to remove is a phrase models read, so the check has to
    run against what they would actually be shown.
    """
    from seahaven.world.loader import open_world
    w = open_world(f"worlds/world_eden_{level}/world_eden_{level}.z8")
    obs, _h = w.reset()
    seen = [obs.text]
    for cmd in ("north", "east", "south", "west", "look", "examine sacking"):
        obs, _h = w.step(cmd)
        seen.append(obs.text)
    w.close()
    return "\n".join(seen).lower()


# --- the defect, in served text --------------------------------------------

def test_LAT2_serves_no_tallow_ANYWHERE_in_its_text():
    assert "tallow" not in _walk("LAT2")


def test_the_walk_FIRES_on_LAT_which_is_what_makes_it_a_check():
    """A check that never fails is decoration. LAT must still serve the defect —
    it is grandfathered on purpose, because past results keep the world they
    actually ran on."""
    lat = _walk("LAT")
    assert "tallow" in lat
    assert "shelves of sacking and tallow" in lat
    assert "shelves of empty sacking" in _walk("LAT2")


def test_LAT2_is_LATs_twin_in_every_derived_value():
    """Same topology, larder, params and derived block. Only the text moved, so
    the two are directly comparable and LAT2 is not a new world to validate."""
    lat, lat2 = (json.loads((_ROOT / f"worlds/world_eden_{lv}/BUILD.lock.json")
                            .read_text()) for lv in ("LAT", "LAT2"))
    for k in ("edges", "distances", "larder", "params", "derived"):
        assert lat[k] == lat2[k], k
    assert lat["sha256"] != lat2["sha256"], "the compiled worlds must differ"


def test_LAT2_gets_LATs_OPENING_LINE_byte_for_byte():
    """The round-6 rule: the scarcity clause is identical across all worlds, and
    a world whose rooms match another's takes the same sentence. LAT2 IS the
    farmstead, so a differing line would be an uncontrolled second change."""
    from seahaven.fidelity import worldspec as W
    assert W.load("world_eden_LAT2").setting == W.load("world_eden_LAT").setting
    assert "at the end of the season" in W.load("world_eden_LAT2").setting


# --- the acceptance check, and why the specified one is not one -------------

def _cogito_lat_rates():
    """(episodes naming tallow, episodes with an `eat ... tallow`, episodes)."""
    named = ate = tot = 0
    for _p, d in C.iter_cells():
        m = d.get("meta", {})
        if m.get("eden_level") != "LAT" or m.get("served_name") != R.SERVED_NAME:
            continue
        for r in C.episodes(d):
            tot += 1
            cmds = [(c.get("command") or "").lower() for c in r["commands"]]
            named += any("tallow" in c for c in cmds)
            ate += any(re.match(r"^\s*eat\b.*tallow", c) for c in cmds)
    return named, ate, tot


def test_the_pinned_BASE_RATES_are_what_the_corpus_actually_says():
    """The pin carries two literals it was not free to invent."""
    named, ate, tot = _cogito_lat_rates()
    assert R.TALLOW_BASELINE["any command naming tallow"] == (named, tot)
    assert R.TALLOW_BASELINE["an `eat ... tallow`"] == (ate, tot)


def test_the_SPECIFIED_check_could_not_have_failed_and_the_pin_says_so():
    """**The reason the acceptance check was replaced.**

    "Zero `eat tallow` in 24 episodes" is satisfied 88% of the time by the
    DEFECTIVE world, because cogito almost never types that exact form. Running
    it and reporting a pass would have been reporting the base rate as evidence.
    """
    k, n = R.TALLOW_BASELINE["an `eat ... tallow`"]
    assert (1 - k / n) ** R.EPISODES_A1 > 0.5, (
        "if this ever drops, the specified check has acquired power and the "
        "pin's claim that it has none needs revisiting")
    assert "NOT evidence" in R.ACCEPTANCE


def test_the_REPLACEMENT_check_has_real_power():
    k, n = R.TALLOW_BASELINE["any command naming tallow"]
    assert (1 - k / n) ** R.EPISODES_A1 < 1e-4
    assert "zero commands naming tallow" in R.ACCEPTANCE


# --- LAT's own record, untouched -------------------------------------------

def test_LATs_lock_is_BYTE_IDENTICAL_and_its_closed_rounds_still_recompute():
    """**What the whole boundary was arranged to protect.** LAT2 is a new world
    precisely so that LAT's record does not move; correcting LAT in place would
    have rewritten what rounds 7, 8 and 10 were served against."""
    from seahaven.eden import round10 as R10
    from seahaven.eden import round7 as R7
    from seahaven.eden import round8 as R8

    got = hashlib.sha256(
        (_ROOT / "worlds/world_eden_LAT/BUILD.lock.json").read_bytes()).hexdigest()
    assert got == R10.RETIRED_R10_LOCKS["worlds/world_eden_LAT/BUILD.lock.json"]
    assert R7.retired_g2_hash() == R7.RETIRED_G2_PIN
    assert R8.retired_g2_hash() == R8.RETIRED_G2_PIN
    assert R10.retired_r10_hash() == R10.PINNED_ROUND10_HASH


def test_the_pin_covers_LAT2s_LOCK_and_the_settings_file_that_cost_five_pins():
    assert R.world_lock_paths() == ("worlds/world_eden_LAT2/BUILD.lock.json",)
    assert "seahaven/fidelity/worldspec.py" in R.ARTIFACTS
    R.assert_pinned()


def test_the_seed_block_is_disjoint_from_every_cogito_block_on_disk():
    burned = C.burned_seeds(model=R.SERVED_NAME)
    want = set(range(R.SEED0, R.SEED0 + max(R.EPISODES_A1, R.EPISODES_A0)))
    assert not (burned & want), sorted(burned & want)[:8]


def test_the_band_is_ROUND_6s_and_the_LAT_reference_is_NOT_a_prediction():
    """LAT2 is accepted on its own band verdict. The LAT comparison is recorded
    as a reference because it is cross-occasion and n=24 against n=48 — this
    round cannot separate a world difference from a between-day one."""
    from seahaven.eden import round6 as R6
    assert (R.BAND_LO, R.BAND_HI) == (R6.BAND_LO, R6.BAND_HI)
    assert "cross-occasion" in R.REFERENCE_CAVEAT
    assert R.band_verdict(12, 24) == "LIVE"
    assert R.band_verdict(0, 24) == "RETUNE"


@pytest.mark.parametrize("k,n,want", [(24, 24, "CONFIRM"), (96, 96, "RETUNE")])
def test_the_band_verdict_uses_the_INTERVAL_not_the_point_estimate(k, n, want):
    """Round 6's rule, unchanged, and the parametrization is the demonstration:
    **the same point estimate of 1.000 gives two different verdicts.**

    At n=24 the interval still touches the band, so one confirmatory cell
    (~$1.25) is cheaper than re-authoring a world that was never broken. At n=96
    it clears the band entirely and the world is genuinely outside. A rule
    reading the point estimate alone could not tell those apart, and the
    asymmetry it would get wrong is the expensive one — round 6 computed that a
    world whose true rate is 0.80 fails on noise 11% of the time at n=24.
    """
    assert R.band_verdict(k, n) == want
