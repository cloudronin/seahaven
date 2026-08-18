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
