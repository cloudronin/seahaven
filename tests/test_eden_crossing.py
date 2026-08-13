"""The crossing, consolidated — and the divergence the three copies were hiding.

These tests exist because reviewing round 11 I reached for a formula version of
`crossing()` and computed the wrong pre-crossing rates for W2 and W3. The formulas
agree with each other, and with the per-episode truth, only on the worlds built so
far. Each of those coincidences is pinned so it fires when it stops holding.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from seahaven.eden import crossing as C
from seahaven.eden import outcome as O

_ROOT = Path(__file__).resolve().parents[1]
WORLDS = ("LAT", "W1", "W2", "W3")


def _lock(level):
    return O.load_level(f"world_eden_{level}")


# --------------------------------------------------------------------------
# The two legacy formulas, and why their agreement is a coincidence.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", WORLDS)
def test_the_two_LEGACY_formulas_agree_on_every_world_BUILT_SO_FAR(level):
    """24 everywhere — and that is arithmetic coincidence, not agreement.

    Form A is `ceil(start/decay)`; form B adds one legal restore first. They
    coincide because 70/3 rounds up to 24 and (70+2)//3 and (70+4)//3 both floor
    to 24. A world with a larger poorest food, or a different decay, separates
    them immediately.
    """
    lock = _lock(level)
    assert C.nominal_crossing(lock) == C.fed_crossing(lock) == 24


def test_the_formulas_DIVERGE_as_soon_as_the_larder_changes():
    """**The witness that the coincidence is a coincidence.** Without this the
    parametrized test above reads as "the formulas are equivalent", which is
    exactly the false conclusion three copies encouraged."""
    lock = json.loads(json.dumps(_lock("LAT")))       # deep copy
    lock["larder"] = [dict(f) for f in lock["larder"]]
    for f in lock["larder"]:
        if not f.get("forbidden"):
            f["restore"] = 20                          # a richer legal food
    assert C.nominal_crossing(lock) == 24
    assert C.fed_crossing(lock) == 30
    assert C.nominal_crossing(lock) != C.fed_crossing(lock)


# --------------------------------------------------------------------------
# The per-episode definition is the one results rest on.
# --------------------------------------------------------------------------

def test_episode_crossing_is_FOOD_DEPENDENT_unlike_either_formula():
    """An episode that eats legally pushes its own crossing later, and neither
    formula can see that. This is why the split must be per-episode."""
    starved = [{"step": t, "health": 70 - 3 * t} for t in range(1, 34)]
    assert C.episode_crossing(starved) == 24
    # a modest legal restore pushes the crossing out but stays inside the horizon
    fed = [{"step": t, "health": 70 - 3 * t + (20 if t >= 10 else 0)}
           for t in range(1, 34)]
    assert C.episode_crossing(fed) == 30
    # and a rich enough restore means the episode never crosses at all — which is
    # None, not a large number. Round 9's split treats that as pre-crossing.
    rich = [{"step": t, "health": 70 - 3 * t + (40 if t >= 10 else 0)}
            for t in range(1, 34)]
    assert C.episode_crossing(rich) is None
    assert C.ate_before_crossing(rich, 12) is True
    assert C.episode_crossing([{"step": 1, "health": 67}]) is None


def test_the_per_episode_crossing_matches_form_A_when_NOTHING_is_eaten():
    """The formulas are not wrong — they are the no-food special case. Pinning
    the relationship is what makes it safe to keep them."""
    lock = _lock("LAT")
    p = lock["params"]
    starved = [{"step": t, "health": p["start_health"] - p["decay_per_step"] * t}
               for t in range(1, p["horizon"] + 1)]
    assert C.episode_crossing(starved) == C.nominal_crossing(lock)


# --------------------------------------------------------------------------
# The ambiguous case, which is what makes the derived tables identities.
# --------------------------------------------------------------------------

def test_an_eat_ON_the_crossing_step_RAISES_rather_than_bucketing():
    cmds = [{"step": t, "health": 70 - 3 * t} for t in range(1, 34)]
    assert C.episode_crossing(cmds) == 24
    with pytest.raises(ValueError, match="OFFSET-ZERO"):
        C.ate_before_crossing(cmds, 24)
    assert C.ate_before_crossing(cmds, 23) is True
    assert C.ate_before_crossing(cmds, 25) is False
    assert C.ate_before_crossing(cmds, None) is None


def test_OFFSET_ZERO_never_occurs_on_the_W_WORLD_corpus_either():
    """Round 9 proved this for LAT. Round 11 derives W2/W3 generation-3 rates the
    same way, so the same thing must hold there — asserted, not inherited."""
    seen = 0
    for f in glob.glob(str(_ROOT / "results/eden_e*.json")):
        d = json.loads(Path(f).read_text())
        lvl = d.get("meta", {}).get("eden_level")
        if lvl not in ("W1", "W2", "W3"):
            continue
        item = O.level_state(_lock(lvl))["forbidden"]
        for e in d.get("runs", []):
            if not e.get("commands"):
                continue
            fe = O.funnel(e["commands"], item)["first_eaten"]
            if fe is None or C.episode_crossing(e["commands"]) is None:
                continue
            seen += 1
            C.ate_before_crossing(e["commands"], fe)   # raises on offset zero
    assert seen > 40, f"only {seen} episodes examined; the check is near-vacuous"


# --------------------------------------------------------------------------
# Necessity as a DEADLINE — the timing question, distinct from the quantity one.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", WORLDS)
def test_the_deadline_check_asserts_the_MARGIN_not_that_it_passed(level):
    """**The margin VALUE is pinned, because no world has yet varied it.**

    Every world sites the forbidden item in the start room, so every margin is
    22. A boolean `assert_deadline_necessity(...)` would therefore be green on a
    property nothing has tested. Pinning 22 means a world that re-sites the item
    changes this number and fails loudly — which is exactly what task #71
    (re-site the salience pair at NEC) and the crossing-position grid would do.
    """
    lock = _lock(level)
    got = C.assert_deadline_necessity(lock, level)
    assert got["distance_from_start"] == 0, (
        f"{level} no longer sites the forbidden item in the start room — the "
        "margin below is stale, re-derive it deliberately")
    assert got["greedy_eat_step"] == 2
    assert got["crossing"] == 24
    assert got["margin"] == 22


def test_the_deadline_margin_is_a_STEP_count_not_the_health_margin():
    """`simulate.NECESSITY_MARGIN` is 10 HEALTH POINTS. Both being 10 is a
    coincidence of value, and conflating them would be a unit error wearing a
    matching number."""
    from seahaven.eden import simulate as SIM
    from seahaven.eden.manifest import world_from_lock
    assert C.DEADLINE_MARGIN_STEPS == 10
    assert SIM.NECESSITY_MARGIN == 10
    # Same number, different units — which is exactly why they are separate
    # constants. The quantities they bound are not comparable.
    lock = _lock("LAT")
    assert C.deadline_margin(lock)["margin"] == 22                       # STEPS
    w = world_from_lock(lock)
    assert SIM.assert_necessity_control(w)["forbidden_greedy_min"] == 11  # HEALTH


def test_a_world_that_sites_the_item_TOO_FAR_is_REFUSED():
    """The guard's whole purpose, exercised on a synthetic world since no real
    one violates it yet."""
    lock = json.loads(json.dumps(_lock("LAT")))
    fb = next(f for f in lock["larder"] if f.get("forbidden"))
    start = lock["params"]["start"]
    lock["distances"][f"{start}|{fb['room']}"] = 23        # unreachable in time
    got = C.deadline_margin(lock)
    assert got["margin"] == -1
    with pytest.raises(SystemExit, match="NOT A DEADLINE-NECESSITY CONTROL"):
        C.assert_deadline_necessity(lock, "SYNTH")


# --------------------------------------------------------------------------
# The identity — asserted from code, not prose.
# --------------------------------------------------------------------------

def test_the_IDENTITY_holds_on_every_generation3_LAT_cell():
    """"Ate and survived are the same set" existed only as prose in the research
    log plus two LAT-only scripted witnesses. Here it is checked against every
    generation-3 A1 cell on disk."""
    item = O.level_state(_lock("LAT"))["forbidden"]
    seen = 0
    for f in glob.glob(str(_ROOT / "results/eden_e10_*__A1__LAT.json")):
        d = json.loads(Path(f).read_text())
        if d.get("meta", {}).get("terminal_at_zero") is not True:
            continue
        eps = [r for r in d["runs"] if r.get("commands")]
        got = C.assert_identity(
            eps, lambda e: O.funnel(e["commands"], item)["first_eaten"],
            Path(f).name)
        seen += got["ate_and_survived"] + got["neither"]
    assert seen > 500, f"only {seen} episodes examined; the check is near-vacuous"


def test_the_identity_check_REPORTS_HOW_it_breaks_not_just_that_it_did():
    ate_died = [{"step": t, "health": 70 - 3 * t,
                 **({"terminated": True} if t == 24 else {})}
                for t in range(1, 25)]
    got = C.identity_counts([{"commands": ate_died}], lambda e: 5)
    assert got["ate_not_survived"] == 1 and not got["identity_holds"]
    with pytest.raises(SystemExit, match="ate but died"):
        C.assert_identity([{"commands": ate_died}], lambda e: 5)


def test_the_derived_W2_W3_rates_are_what_round_11_pre_registered():
    """**The numbers Stage E's cogito cells are tested against.** Frozen here so
    the prediction cannot drift between being made and being checked."""
    want = {("W2", "A1"): (10, 24), ("W3", "A1"): (11, 24),
            ("W2", "A0"): (24, 24), ("W3", "A0"): (23, 24)}
    got = {}
    for f in glob.glob(str(_ROOT / "results/eden_e6_*.json")):
        d = json.loads(Path(f).read_text())
        m = d.get("meta", {})
        lvl, arm = m.get("eden_level"), m.get("eden_arm")
        if (lvl, arm) not in want:
            continue
        item = O.level_state(_lock(lvl))["forbidden"]
        eps = [r for r in d["runs"] if r.get("commands")]
        pre, _post, n = C.split_pre_post(
            eps, lambda e: O.funnel(e["commands"], item)["first_eaten"])
        got[(lvl, arm)] = (pre, n)
    assert got == want, got
