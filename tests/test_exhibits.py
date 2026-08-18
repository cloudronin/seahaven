"""Exhibit 1 — the mechanism-flip table, and the two bars it has to clear.

The exhibit was named three times in plans and built zero times. Building it is
what established that it cannot yet be populated, which is a fact about the
corpus that no amount of planning was going to surface.
"""

from __future__ import annotations

from seahaven.eden import probe as PB
from seahaven.eden._shared import identity as ID
from vetoworld.register import exhibits as EX


def test_EXHIBIT_1_IS_BLOCKED_AND_SAYS_WHY():
    """**The plan recorded Exhibit 1 as "$0 and available now", blocking on
    nothing. It is blocked, and unblocking it costs money.**

    Round 21 served the DeepInfra side of every overlapping model. The
    Together-era counterparts all come from rounds 15 and 16, both in
    `RETRACTED_SWEEPS`, so no pair has an admissible cell on both sides.

    This test exists so the emptiness is a checked claim rather than a
    paragraph. If a live Together cell ever lands for one of these models the
    count moves and this test fails — which is the point. Read the failure as
    "the exhibit can now be built", not as a regression.
    """
    rows = EX.mechanism_rows()
    blocked = EX.blocked_by_retraction()

    assert rows == [], (
        "a cross-provider pair is now admissible on both sides — Exhibit 1 can "
        f"be built. Rows: {rows}")
    assert blocked, "the blocked set cannot be empty while rows is empty too"

    #: Every blocked pair is blocked for the STATED reason, not some other one.
    for b in blocked:
        assert b["retracted"], f"{b['model']} {b['level']} has no retracted side"
        assert b["live"] == ["deepinfra"], (
            f"{b['model']} {b['level']} live sides are {b['live']} — if a "
            "Together cell went live, the exhibit is unblocked for this pair")
        for _provider, rounds in b["retracted"].items():
            assert set(rounds) <= set(ID.RETRACTED_SWEEPS)


def test_THE_ZERO_READS_AS_SUBSTRATE_PENDING_NOT_ABANDONED():
    """**A zero with no stated route out reads as a dead end**, and this one is
    not. `SUBSTRATE_PENDING` is printed where the reader meets the zero.

    It also corrects this module's own first guess. The exhibit originally said
    unblocking it "costs money"; it does not, at any price — the blocked models
    cannot be served on Together at all.
    """
    sp = EX.SUBSTRATE_PENDING
    assert "SUBSTRATE PENDING" in sp and "NOT ABANDONED" in sp
    assert "no budget reopens them" in sp, (
        "the point is that money does NOT unblock this, which is the opposite "
        "of what the plan and this module first recorded")
    assert "MATCHED-PAIR FLEET" in sp, "a zero needs its route out named"
    assert "PROBE-SOURCED" in sp and "DESCRIPTIVE" in sp, (
        "probe cells are excluded from the score corpus by construction, so a "
        "comparison drawn from them must never read as a round measurement")

    #: It is printed, not merely defined — the defect this repo keeps finding.
    import inspect
    assert "SUBSTRATE_PENDING" in inspect.getsource(EX.exhibit_1)


def test_THE_WALL_COUNT_IS_DERIVED_FROM_THE_ROUND_20_REGISTRY():
    """**"Six of the seven" is a number, and numbers in prose drift.**

    `SUBSTRATE_PENDING` claims six of the seven blocked models are
    non-serverless on Together and that Kimi-K3 is walled by a different
    mechanism. That is checked against round 20's own registry here, so the day
    Together re-lists one of them this sentence fails rather than misinforms.
    """
    from seahaven.eden import round20 as R20

    seven = {r["model"] for r in EX.blocked_by_retraction()}
    assert len(seven) == 7, f"the blocked set is no longer seven models: {seven}"

    walls = {}
    for model in seven:
        why = (getattr(R20, "BLOCKED", {}) or {}).get(model) \
            or (getattr(R20, "EXCLUDED", {}) or {}).get(model)
        assert why, f"{model} is blocked by retraction but round 20 gives no reason"
        walls[model] = str(why)

    tiered = [m for m, w in walls.items() if "non-serverless" in w]
    assert len(tiered) == 6, (
        f"SUBSTRATE_PENDING says six are non-serverless on Together; the "
        f"registry says {len(tiered)}: {sorted(tiered)}")
    other = [m for m in walls if m not in tiered]
    assert other == ["moonshotai/Kimi-K3"], other
    assert "EMPTY CONTENT" in walls[other[0]]


def test_THE_ADMISSIBILITY_RULE_IS_THE_LEVELS_RULE_ONE_LAYER_DOWN():
    """**A NOT_ZERO/zero difference is a RATE difference wearing a mechanism
    label**, and `LEVELS_RULE` makes cross-provider rate differences
    documentation rather than findings.

    This is the bar behind the retraction bar: even once live cells exist, a
    pair only becomes a mechanism flip if BOTH sides reached zero. Fixing the
    retraction does not fix this, so both are counted separately.
    """
    assert "LEVEL" in PB.LEVELS_RULE and "never findings" in PB.LEVELS_RULE
    a = EX.ADMISSIBILITY
    assert "same on both sides" in a
    assert "LEVELS_RULE" in a
    assert "NEVER_TOOK_UNVERIFIED" in a, (
        "the undecidable case must be named, or it gets counted as agreement — "
        "which is [CORRECTION] 11's mistake in a different table")


def test_THE_EXHIBIT_CLASSIFIES_RATHER_THAN_FILTERS():
    """Inadmissible rows are PRINTED, not dropped. A reader who cannot see what
    was excluded cannot tell a rule was applied from a table that found
    nothing — and the excluded rows here are the striking-looking ones."""
    import inspect

    src = inspect.getsource(EX)
    for bucket in ("LEVEL-DIFFERENCE", "UNDECIDABLE", "AGREES", "FLIP"):
        assert bucket in src
    assert "BLOCKED BY RETRACTION" in src, (
        "the reason the exhibit is empty must appear in the emitted table, not "
        "only in the module docstring")


def test_EXHIBIT_1_IS_REGISTERED_AS_AN_EMITTABLE_ARTIFACT():
    """An exhibit nobody can run is the state this one was already in. It has
    to be reachable from `vworld emit`, like every other register artifact."""
    from vetoworld.commands.emit import registry

    assert "exhibit-1" in registry()
    assert registry()["exhibit-1"] is EX.exhibit_1
