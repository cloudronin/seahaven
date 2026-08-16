"""Round 20 — round 15's grid, served honestly.

Two things a pin cannot check about itself: that its literals still describe the
corpus they were read off, and that the rule they encode does what the docstring
says when run against real cells. Written before a single cell was served.
"""

from __future__ import annotations

import pytest

from seahaven.eden import round15 as R15
from seahaven.eden import round20 as R
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import identity as ID


def test_the_PIN_RECOMPUTES_and_the_round_is_OPEN():
    """Round 15 is closed by the worldspec boundary and refuses; round 20 is
    the live round and must accept. If this ever raises, no cell may be served
    against it."""
    assert R.current_hash() == R.PINNED_ROUND20_HASH
    R.assert_pinned()

    with pytest.raises(SystemExit, match="CLOSED"):
        R15.assert_pinned()


def test_the_COHORT_IS_ROUND_15s_EXACTLY_so_the_two_are_comparable():
    """**Retyped deliberately, and asserted equal rather than imported.**

    A re-serve that quietly re-scoped would make the two rounds incomparable
    for reasons unrelated to the defect, and the whole point is to find out
    what round 15 would have said. Importing `R15.COHORT` would make the
    equality unfalsifiable; asserting it makes a silent edit fail here.
    """
    assert R.COHORT == R15.COHORT
    assert len(R.COHORT) == 14
    assert R.LEVELS == R15.LEVELS
    assert R.ARMS == R15.ARMS
    assert R.EPISODES_A1 == R15.EPISODES_A1 == 48
    assert R.EPISODES_A0 == R15.EPISODES_A0 == 24
    assert R.COMP_EPISODES == R15.COMP_EPISODES
    assert R.SCORE == R15.SCORE and R.COMPANION == R15.COMPANION
    assert R.RULES_RESOLVED == R15.RULES_RESOLVED
    assert R.E1_READING == R15.E1_READING


def test_the_SEED_BLOCKS_ARE_FRESH_because_the_old_ones_are_burned():
    """Round 15's block was burned — by cogito, under fourteen filenames, which
    is exactly why it cannot be reused. A re-serve into burned seeds would not
    be a new measurement of anything."""
    burned = C.burned_seeds()
    grid = set(range(R.SEED0, R.SEED0 + max(R.EPISODES_A1, R.EPISODES_A0)))
    comp = set(range(R.COMP_SEED0, R.COMP_SEED0 + R.COMP_EPISODES))

    assert R.SEED0 != R15.SEED0 and R.COMP_SEED0 != R15.COMP_SEED0
    assert not grid & burned, sorted(grid & burned)[:8]
    assert not comp & burned, sorted(comp & burned)[:8]
    assert not grid & comp, "the COMP gate must not reuse the grid's block"

    #: And below the probe fleet's reservation, which is a different instrument
    #: with its own 60-day block.
    from seahaven.eden import probe as PB
    assert max(grid | comp) < PB.SEED_BLOCK[0]


def test_THE_EIGHT_NEVER_MEASURED_ARE_EXACTLY_THE_ONES_WITH_NO_CELLS():
    """**"Never measured" and "measured, then withdrawn" are different states.**

    The literal is checked against the corpus rather than trusted. A model is
    NEVER_MEASURED iff it has no **generation-3** cell at any of this round's
    worlds under SERVED identity, excluding the retracted sweeps.

    **The generation filter is load-bearing and was not obvious.** Written
    without it this test failed on GLM-5.2, which has LAT cells at e3/e4/e8 —
    gen1 and gen2, before and during the recovery-line era. Those are a
    different measurement with different death semantics and they may never
    pool with this round's gen-3 cells, so for round 20's purposes GLM-5.2 has
    nothing, exactly like the other seven. The failure was the test being
    imprecise, not the literal being stale, and the fix is to say which
    generation the claim is about.
    """
    have = set()
    for p, d in C.iter_cells():
        m = d.get("meta", {})
        if m.get("eden_level") not in R.LEVELS:
            continue
        if C.generation_of(m) != "gen3" or C.is_diagnostic(m):
            continue
        got = C.parse_cell_name(p.name)
        if not got or got["round"] in ID.RETRACTED_SWEEPS:
            continue
        have.add(ID.model_identity(m).served)

    never = {m for m in R.COHORT if m not in have}
    assert never == set(R.NEVER_MEASURED), (
        f"NEVER_MEASURED is stale: corpus says {sorted(never)}")
    assert len(R.NEVER_MEASURED) == 8

    #: The other six DO have gen-3 cells, so the split is real rather than an
    #: artefact of the filter being too narrow.
    assert len(set(R.COHORT) - never) == 6


def test_the_GRID_IS_98_CELLS_and_every_one_is_new():
    """14 models x (3 worlds x 2 arms) + 14 COMP = 98. Asserted because the
    cost estimate and the budget gate are both derived from it."""
    grid, comp = R.cells(), R.comp_cells()
    assert len(grid) == 14 * 3 * 2 == 84
    assert len(comp) == 14
    assert len(grid) + len(comp) == 98
    assert {lv for _m, _a, lv in comp} == {"COMP"}
    assert {a for _m, a, _lv in comp} == {"A0"}


def test_THE_IDENTITY_RULE_IS_INSIDE_THE_PAYLOAD_not_just_in_the_code():
    """**The precondition is registered, not merely implemented.**

    A rule that lives only in whatever code happens to be checked out when the
    cells are read is not a pre-registration. Round 15 failed for want of this
    exact check, so it is hashed into the pin rather than asserted about the
    working tree.
    """
    payload = R.payload()
    assert "requested == served" in payload
    assert R.IDENTITY_RULE[:50] in payload
    assert R.CANNOT_SETTLE[:40] in payload
    assert "supersedes" in payload


def test_the_ROUND_SAYS_WHAT_IT_CANNOT_SETTLE_before_it_runs():
    """Round 15's cells are one model's, served on a different day. Any
    comparison to them confounds model with occasion, and no contrast in this
    design separates the two. Registered so it cannot be forgotten between
    serving and writeup — which is when it would be most convenient to."""
    assert "confounds model" in R.CANNOT_SETTLE
    assert "FIRST measurement" in R.CANNOT_SETTLE
    assert "never evaluated" in R.RULE_5_DEFERRED


def test_RULES_1_TO_4_STAY_RESOLVED_because_their_cohort_was_never_affected():
    """They were frozen on the nine-model pre-spec cohort — rounds <= 14, one
    model per invocation. Re-opening them would discard genuine results to
    punish an unrelated bug."""
    assert R.RULES_RESOLVED == R15.RULES_RESOLVED
    assert "RESOLVED" not in R.RULE_5_DEFERRED
    assert R.RULE_5_DEFERRED != R15.RULE_5_DEFERRED, (
        "rule 5's deferral text must record that round 15's serving never "
        "evaluated it, rather than being copied forward unchanged")
