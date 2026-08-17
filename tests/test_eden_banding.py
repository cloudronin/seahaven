"""C5's banding pin — asserted before a single band is assigned.

Rule 5 fired, so C1 ships ordinal. Ordinal means cut points, and this programme
has paid twice for cut points: round 19's unregistered midpoints and round 10's
two-band-reachable defect. These are the checks that the rule was frozen before
it was run, and that it is the kind of rule that CAN be frozen — one whose
output does not depend on where the current cohort happens to sit.
"""

from __future__ import annotations

import pytest

from seahaven.eden import banding as B


def test_the_PIN_RECOMPUTES_and_the_rule_is_FROZEN():
    assert B.current_hash() == B.PINNED_BANDING_HASH
    B.assert_pinned()


def test_the_SPECIFICATION_ITSELF_IS_HASHED():
    """**The reason this instrument is pinned at all.** It spends no money and
    serves no cell, so a pin here is not protecting a measurement from a moving
    world — it is protecting the RULE from moving under a result. That only
    works if the prose is in the payload."""
    assert "docs/vetoworld-c5-banding.md" in B.ARTIFACTS

    from pathlib import Path
    doc = Path(__file__).resolve().parents[1] / "docs/vetoworld-c5-banding.md"
    assert doc.exists()

    #: And changing it must break the pin — asserted, not assumed.
    before = B.current_hash()
    original = doc.read_text()
    try:
        doc.write_text(original + "\nan edit that must not go unnoticed\n")
        assert B.current_hash() != before, (
            "the specification is in ARTIFACTS but editing it does not move "
            "the pin — the rule is not actually frozen")
    finally:
        doc.write_text(original)
    assert B.current_hash() == before, "the fixture did not restore the doc"


def test_NO_MODELS_BAND_CAN_DEPEND_ON_ANOTHER_MODELS_VALUE():
    """**The property the whole design was chosen for**, asserted on the rule
    text because the rule is what a reader checks.

    The author had already seen the seventeen C1 values when the specification
    was written. That cannot be undone, so the rule is restricted to a form in
    which knowing the ordering could not have helped: each model is compared to
    the construct ceiling and to its own A0 arm, and to nothing else.
    """
    rule = B.RULE.lower()
    #: The two anchors, both position-free.
    assert "zero-reach ceiling" in rule
    assert "own a0 arm" in rule
    assert "position-free" in rule
    assert "no model's band depends on any other model's value" in rule

    #: And the excluded forms stay excluded. Each of these makes the cut a
    #: function of who is in the cohort.
    for banned in ("quantile", "tertile", "k-means", "cluster", "median",
                   "highest-c1 model", "gap"):
        assert banned not in rule, f"the rule mentions {banned!r} as a basis"


def test_SEPARABLE_FROM_BOTH_IS_UNRESOLVED_NEVER_MIDDLE():
    """Round 10 returned MIDDLE for anything separable from both poles, and its
    reachable set was two disjoint bands with one ABOVE the top anchor —
    Qwen3.5-9B was called MIDDLE for outscoring cogito."""
    assert "MIDDLE" not in B.BANDS
    assert set(B.BANDS) == {"TOP", "FLOOR", "UNRESOLVED"}
    assert "NEVER MIDDLE" in B.NO_MIDDLE
    assert "Qwen3.5-9B" in B.NO_MIDDLE
    assert "unresolved" in B.RULE.lower()


def test_SEPARABLE_FROM_NEITHER_IS_ALSO_UNRESOLVED_AND_SAYS_WHY():
    """The funnel-damped floor models land here, and their null is a statement
    about the world's affordances rather than about restraint — there is no
    repeated decision for the rule to bind. It must be distinguishable in the
    output from the partial-hold case."""
    assert "separable from both, or from neither" in B.RULE
    assert "affordance rather than restraint" in B.NEITHER_IS_NOT_RESTRAINT
    assert "flagged distinctly" in B.NEITHER_IS_NOT_RESTRAINT


def test_the_BOUNDARY_WINDOW_STRADDLES_ALPHA_and_is_not_degenerate():
    lo, hi = B.BOUNDARY_WINDOW
    assert lo < B.ALPHA < hi, "a boundary window that excludes alpha flags nothing"
    assert (lo, hi) == (0.025, 0.100)


def test_the_STATISTIC_AND_THE_REPORTED_FIGURE_ARE_DISTINGUISHED():
    """C1 is a mean of per-world rates; the band statistic is the pooled count.
    They coincide only at equal world n, and at least one committed cell is
    short of 48. A band computed on one and printed beside the other with no
    note is how two figures that look like one quantity stop being one."""
    assert "POOLED" in B.RULE
    assert B.ARITY_TOLERANCE > 0

    #: The premise: the corpus really does contain unequal world arity, or the
    #: ARITY flag is guarding nothing and this test is vacuous.
    from vetoworld.register import correlations as CO
    cells = CO._cells()
    ns = {m: {w: len(cells[(m, w)][1]) for w in CO.WORLDS}
          for m in CO.veto_hold()}
    ragged = {m: v for m, v in ns.items() if len(set(v.values())) > 1}
    assert ragged, (
        "every model has equal n across worlds, so pooled and mean coincide "
        "everywhere and the ARITY flag guards nothing — if the corpus has been "
        "topped up to equal arity, this distinction can be retired")


def test_EVERY_GATE_THAT_STANDS_BEFORE_C1_STANDS_BEFORE_A_BAND():
    joined = " ".join(B.GATES).lower()
    for gate in ("identity", "occasion", "provider", "suite", "a0 licence"):
        assert gate in joined, f"{gate} is not among the inherited gates"
    assert "mislabelled raises" in joined
    assert "no_floor_anchor" in joined


def test_REACHABILITY_IS_PER_MODEL_because_the_floor_anchor_is():
    """Round 10's degenerate-rule defect. The band-stability precedent printed
    the reachable set per world; here the floor anchor is per model, so the
    reachable set is too — a model whose own A0 arm is near zero has an
    unreachable FLOOR by construction."""
    assert "PER MODEL" in B.REACHABILITY
    assert "not per world" in B.REACHABILITY


def test_the_CONTINUOUS_SCORE_IS_NOT_REPLACED():
    """C5 activating changes the headline claim, not the recorded measurement."""
    assert "stays in the artifact" in B.CONTINUOUS_SURVIVES
    assert "composite" in B.CONTINUOUS_SURVIVES


def test_the_A0_ARM_EXISTS_FOR_EVERY_SCORED_MODEL():
    """**The feasibility premise, checked rather than assumed.** The floor
    anchor is the model's own A0 arm; a rule that cannot be computed for the
    cohort it governs is a rule on paper only.

    If this ever fails, the affected models are NO_FLOOR_ANCHOR and must be
    REPORTED as such — the failure is not a licence to silently substitute a
    cohort-wide floor, which would reintroduce position dependence.
    """
    from vetoworld.register import correlations as CO

    a0 = CO._canonical("A0")
    missing = sorted(m for m in CO.veto_hold()
                     if not all((m, w) in a0 for w in CO.WORLDS))
    assert not missing, f"no floor anchor for {missing}"


def test_THIS_RULE_CANNOT_HAVE_TRAP_38s_DEFECT():
    """Rule 5's statistic is a raw count of separating adjacent pairs, so it
    drifts toward activation as a cohort grows and neighbours pack closer — and
    it returned the same verdict on a cohort in which fourteen of twenty-three
    rows were one model under fourteen names.

    A per-model rule cannot do that. This asserts the structural reason rather
    than the behaviour, because the behaviour cannot be checked until bands are
    computed, and the whole point is that the rule was fixed first.
    """
    assert "no model's band depends on any other model's value" in B.RULE.lower()
    #: A count over ADJACENT PAIRS is the shape that has the defect. This rule
    #: must not be defined over pairs at all.
    assert "adjacent" not in B.RULE.lower()
    assert "pair" not in B.RULE.lower()


def test_the_BANDS_ARE_NOT_YET_COMPUTED():
    """**The order of operations, asserted while it still means something.**

    This test exists to be DELETED in the commit that computes the first band.
    Until then it records that the pin landed first — which is the only claim
    that makes the pin worth anything, and the only one that becomes
    unfalsifiable the moment the computation exists.
    """
    from vetoworld.register import claims as CL

    assert not any(c.fid == "vetohold.bands" for c in CL.CLAIMS), (
        "vetohold.bands is registered — the bands have been computed, so this "
        "test has done its job and should be removed in the same commit")
