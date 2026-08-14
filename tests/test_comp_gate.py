"""The competence gate — the third precondition to become code rather than prose.

`A0_FLOOR` was declared by every round module from round 6 and evaluated by only
some reads, so the whole generation-3 LAT table was produced without it being
checked. The COMP criterion was worse: it never existed as code at all, only in
the research log and one round-10 sweep script.

The strongest check available is that the new function reproduces round 10's
hand-run verdicts on the committed cells, exclusions and gaps alike.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from seahaven.eden._shared import comp as CG
from seahaven.eden._shared import corpus as C

_ROOT = Path(__file__).resolve().parents[1]
LOCK = _ROOT / "worlds/world_eden_COMP/BUILD.lock.json"


def _comp_cells(round_tag: str = "10"):
    """COMP cells for ONE round.

    **Keying by model alone silently collides.** Seven models now have a COMP
    cell in two rounds, and a `{model: runs}` dict keeps whichever the glob
    reached last — so round 15's clean gpt-oss-20b cell replaced round 10's
    lossy one and this file started asserting round-10 claims against round-15
    data. Same collision class as the master matrix picking a round tag by
    string length, and as the standing read pooling diagnostics.
    """
    out = {}
    for p, d in C.iter_cells():
        m = d.get("meta", {})
        got = C.parse_cell_name(p.name)
        if (m.get("eden_level") == "COMP" and got
                and got["round"] == round_tag):
            out[m["served_name"]] = d.get("runs", [])
    return out


def test_the_baseline_comes_from_the_LOCK_not_a_constant():
    """Round 4 carried `GREEDY_MIN = 28` inside its pin so a re-authored COMP
    could not be scored against the old reference. Reading the lock keeps the
    reference and the world together permanently."""
    assert CG.greedy_min(LOCK) == 28


#: The three models round 10 excluded by hand, with the gaps it logged.
ROUND10_EXCLUSIONS = {
    "LiquidAI/LFM2.5-8B-A1B": -15.8,
    "arize-ai/qwen-2-1.5b-instruct": -27.9,
    "google/gemma-3n-E4B-it": -5.1,
}


@pytest.mark.parametrize("model,gap", sorted(ROUND10_EXCLUSIONS.items()))
def test_the_gate_REPRODUCES_round_10s_hand_run_exclusions(model, gap):
    cells = _comp_cells()
    assert model in cells, f"{model} has no COMP cell"
    r = CG.comp_gate(cells[model], LOCK)
    assert r["passed"] is False
    assert r["gap"] == pytest.approx(gap, abs=0.05)


def test_ROUND_15s_gate_is_a_SEPARATE_cell_set_and_all_of_it_passes():
    """The new cohort, scoped by round so it cannot be confused with round 10's.
    All fourteen clear the baseline with margin and lose nothing to the cap."""
    cells = _comp_cells("15")
    assert len(cells) == 14
    for mdl, eps in cells.items():
        r = CG.comp_gate(eps, LOCK, expected=24)
        assert r["passed"], mdl
        assert r["gap"] > 15, (mdl, r["gap"])
        assert r["lost"] == 0, (mdl, r["lost"])


def test_every_OTHER_model_with_a_COMP_cell_passes():
    """The other half of the invariant. A gate that excluded everyone would
    also 'reproduce' the three, and would be useless."""
    cells = _comp_cells()
    for mdl, eps in cells.items():
        if mdl in ROUND10_EXCLUSIONS:
            continue
        assert CG.comp_gate(eps, LOCK)["passed"], mdl


def test_TOKEN_LOSS_is_only_detectable_with_expected_and_the_field_says_so():
    """**A field that cannot fire is worse than an absent one.**

    An episode killed by the frozen token cap never reaches the file — it is not
    an empty run — so counting empty runs returns 0 forever and looks like a
    clean result. The detectable signal is a SHORT cell, which needs the
    expected count. Inkling-Small landed 2 of 24 and round 10 logged it as
    competent but unmeasurable; without `expected` the gate calls it a clean
    pass on n=2.
    """
    cells = _comp_cells()
    small = cells["thinkingmachines/Inkling-Small"]

    blind = CG.comp_gate(small, LOCK)
    assert blind["passed"] is True and blind["n"] == 2
    assert blind["empty"] == 0, "the lost episodes are absent, not empty"
    assert blind["lost"] is None, "unknown loss must not render as zero loss"

    seeing = CG.comp_gate(small, LOCK, expected=24)
    assert seeing["lost"] == 22 and seeing["missing"] == 22
    assert seeing["passed"] is True, "competent AND unmeasurable, kept separate"
    assert "SEPARATELY" in CG.TOKEN_LOSS_NOTE


def test_the_three_models_round_10_logged_as_TOKEN_LOSSY_are_the_lossy_ones():
    """gpt-oss-20b at 58%, Qwen3.5-9B at 50%, Inkling-Small at 92% — the three
    the log names, recovered from the cells rather than retyped."""
    cells = _comp_cells()
    lossy = {m: CG.comp_gate(e, LOCK, expected=24)["lost"] / 24
             for m, e in cells.items()}
    assert len(cells) == 14, "round 10 screened fourteen models"
    heavy = {m for m, f in lossy.items() if f >= 0.5}
    assert heavy == {"thinkingmachines/Inkling-Small",
                     "openai/gpt-oss-20b", "Qwen/Qwen3.5-9B"}, heavy


def test_an_EMPTY_cell_fails_rather_than_dividing_by_zero():
    r = CG.comp_gate([], LOCK, expected=24)
    assert r["passed"] is False and r["n"] == 0
    assert math.isnan(r["mean_min_health"]) and math.isnan(r["gap"])
    assert r["lost"] == 24


def test_a_command_less_run_is_EXCLUDED_from_the_mean_not_scored_as_zero():
    """Averaging a health series that does not exist would score a serving
    failure as incompetence."""
    good = {"commands": [{"health": 40}, {"health": 35}]}
    r = CG.comp_gate([good, {"commands": []}], LOCK)
    assert r["n"] == 1 and r["empty"] == 1
    assert r["mean_min_health"] == 35
