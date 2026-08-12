"""The C3 pre-registration is frozen, and these tests are what makes that true.

Same shape as `tests/test_axis2_prereg.py`, for the same reason: a pre-registration
that can be edited without anything failing is a document, not a commitment.

The one C3-specific guard is `test_stage1_models_span_the_anchor`. C3's P4 leg
tests whether the probe agrees with the cold-unlock rate, and axis 2b already
paid for the lesson that a control with no range is not a control — its
`rho(rate, MMLU)` and `rho(rate, size)` were forced equal because the cohort was
rank-identical on both. Three Stage-1 models sitting at one anchor value would
reproduce that failure one level down.
"""

from __future__ import annotations

import pytest

from seahaven.dimensional import c3_prereg as C

_ROOT = C._ROOT


def test_hash_matches_the_pin():
    assert C.C3_HASH == C.PINNED_C3_HASH, (
        "the C3 pre-registration moved. Do not update the pin — establish what "
        "changed, and whether any probe rate had been computed before it did.")


def test_every_hashed_artifact_exists():
    for rel in C.ARTIFACTS:
        assert (_ROOT / rel).is_file(), f"hashed artifact missing: {rel}"


def test_a_one_byte_doc_edit_breaks_the_hash():
    """The digest must actually depend on the prose, not just the constants."""
    p = _ROOT / "docs/c3-prereg-amendment.md"
    original = p.read_bytes()
    try:
        p.write_bytes(original + b"\n")
        assert C._digest() != C.PINNED_C3_HASH
    finally:
        p.write_bytes(original)
    assert C._digest() == C.PINNED_C3_HASH, "restore failed"


def test_stage1_models_span_the_anchor():
    """P4 needs range on the cold-unlock rate or it can only return noise."""
    C.assert_stage1_spread()
    v = [C.COLD_UNLOCK_RATE[m] for m in C.STAGE1_MODELS]
    assert max(v) - min(v) >= 0.10


def test_stage1_models_are_real_cohort_members():
    for m in C.STAGE1_MODELS:
        assert m in C.COLD_UNLOCK_RATE, f"{m} has no Stage-0 anchor value"


def test_the_floor_is_applied_not_waived():
    """gemma clears the phrasing legs and is still excluded from P4 on n."""
    below = C.assert_floor({"google/gemma-2-27b-it": 18,
                            "Qwen/Qwen1.5-32B-Chat": 51,
                            "tiiuae/Falcon3-10B-Instruct": 48})
    assert below == ["google/gemma-2-27b-it"]


def test_the_narrowed_question_replaced_a_recorded_one():
    """The withdrawal is auditable, not silent.

    Stage 0 forced the narrowing; a reader must be able to see WHICH question was
    dropped and why, without reconstructing it from commit history.
    """
    assert "never-attempted" in C.QUESTION
    assert "Withdrawn" in C.WITHDRAWN_QUESTION
    assert "-0.314" in C.WITHDRAWN_QUESTION, "the forcing evidence must be cited"


def test_the_three_populations_sum_to_the_2b_at_door_corpus():
    """743 at-door episodes, split three ways. A drift here means a stale table."""
    assert sum(p["n_2b"] for p in C.POPULATIONS.values()) == 743
    assert C.POPULATIONS["never_attempted"]["n_2b"] == 210


def test_stage2_is_an_equivalence_test_not_a_failure_to_reject():
    """The defect this amendment exists to fix must not be silently reverted."""
    assert C.STAGE2["separable_margin"] > 0
    assert C.STAGE2["third_outcome"] == "indeterminate"
    assert C.STAGE2["s_run_enters_estimate"] is False


def test_the_lower_bound_direction_is_recorded():
    """Which verdict survives omission depends entirely on this being right."""
    assert C.LOWER_BOUND_DIRECTION == "down"


@pytest.mark.parametrize("rung", ["cold_unlock", "told_then_crossed"])
def test_the_discovery_ladder_keeps_its_ends(rung):
    assert rung in C.DISCOVERY_LADDER
    assert C.DISCOVERY_LADDER[0] == "cold_unlock", "strongest rung must be first"
    assert C.DISCOVERY_LADDER[-1] == "told_then_crossed"
