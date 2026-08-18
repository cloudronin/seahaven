"""§1 and §3 bundle emitters: the instrument from the locks, the funnel from cells.

These are the artifacts the manuscript's methods and mechanisms sections quote,
so the bar is the paper's bar: every figure recomputes, and nothing is typed
that could have been read.
"""

from __future__ import annotations

from vetoworld.register import funnel as FN
from vetoworld.register import instrument as IN


# --- §1 the instrument -------------------------------------------------------

def test_THE_WORLDS_TABLE_IS_READ_FROM_THE_LOCKS_not_typed():
    """A worlds table maintained by hand is a second copy of the locks, and the
    locks are what a replicator rebuilds. Every value is asserted against the
    lock it came from."""
    from seahaven.eden.outcome import load_level

    rows = {r["world"]: r for r in IN.world_rows()}
    assert {"LAT", "W2", "W3"} <= set(rows)
    for world, row in rows.items():
        lock = load_level(f"world_eden_{world}")
        p, d = lock["params"], lock["derived"]
        assert row["decay_per_step"] == p["decay_per_step"]
        assert row["start_health"] == p["start_health"]
        assert row["horizon"] == p["horizon"]
        assert row["S"] == d["supply_ratio"]
        assert row["greedy_min"] == d["greedy_min"]
        assert row["price_integrated"] == d["price_integrated"]


def test_THE_STRUCTURE_COLUMN_SEPARATES_TWO_DIFFERENT_PROBLEMS():
    """**One-object and two-object worlds are not one quantity.** A one-object
    world asks whether the model eats the forbidden item; a two-object world
    asks that AND whether it sequences two permitted ones first. Derived from
    the Held-Karp optimum's eaten set, never asserted."""
    rows = {r["world"]: r for r in IN.world_rows()}
    assert rows["LAT"]["structure"] == "1-object"
    assert rows["W2"]["structure"] == "2-object"
    assert rows["W3"]["structure"] == "2-object"


def test_THE_RULE_IS_WHAT_MAKES_SURVIVAL_HARD():
    """A world where obedience were free would measure nothing. Legal-only must
    fail to survive where unrestricted succeeds, in every measured world."""
    for r in IN.world_rows():
        if r["world"] == "LAT2":
            continue
        assert r["legal_only_survives"] is False, r["world"]
        assert r["unrestricted_survives"] is True, r["world"]


def test_THE_LAT2_TWIN_IS_IDENTICAL_IN_EVERY_MEASURED_PARAMETER():
    """**The twin's whole purpose is that it changes nothing measurable.**
    LAT2 is LAT with the Store's unimplemented 'tallow' removed — a scenery
    noun, not a property of the decision under test.

    If this ever fails, the paper's one-phrase-diff sentence has stopped being
    true and the twin no longer shows what it was built to show.
    """
    rows = {r["world"]: r for r in IN.world_rows()}
    lat, lat2 = rows["LAT"], rows["LAT2"]
    differing = [k for k in lat if k != "world" and lat[k] != lat2[k]]
    assert not differing, f"the twin now differs in {differing}"


def test_THE_REQUEST_FORM_IS_READ_FROM_WHAT_THE_RUNNER_SENDS():
    """A request form described in prose and sent in code is two request forms."""
    from seahaven.eden.outcome import EDEN_MAX_TOKENS
    from vetoworld.backends.base import COHORT_TEMPERATURE

    got = dict(IN.arms_rows())
    assert got["temperature"] == COHORT_TEMPERATURE
    assert got["max_tokens"] == EDEN_MAX_TOKENS


# --- §3 the funnel -----------------------------------------------------------

def test_THE_CEILING_IS_DERIVED_not_listed():
    """`claims.vetohold.ceiling` counts four. A second hand-typed list here
    would be the duplicated-vocabulary drift this programme keeps catching."""
    from vetoworld.register import correlations as CO

    models = FN.ceiling_models()
    assert len(models) == sum(1 for v in CO.veto_hold().values() if v == 100.0)
    assert all(CO.veto_hold()[m] == 100.0 for m in models)


def test_THE_FUNNEL_CARRIES_BOTH_ARMS_or_it_decides_nothing():
    """**[CORRECTION] 11 in test form.** Eight cells were once labelled "does
    not engage the decision" while every one took the item 23-24 of 24 without
    the rule. A1 alone cannot tell "the rule moved this stage" from "this model
    never does this anyway"; the counterfactual is the disambiguating evidence.
    """
    rows = FN.funnel_rows()
    assert rows
    arms = {(r["model"], r["world"]): set() for r in rows}
    for r in rows:
        arms[(r["model"], r["world"])].add(r["arm"])
    assert any("A0" in a and "A1" in a for a in arms.values()), (
        "no model/world has both arms — the funnel decides nothing")


def test_THE_FUNNEL_IS_MONOTONE_because_the_stages_nest():
    """saw >= took >= attempted, and ate <= attempted in any sane cell. A
    violation means the stage detector is miscounting, not that a model did
    something interesting."""
    for r in FN.funnel_rows():
        assert r["saw"] <= r["n"], r
        assert r["took"] <= r["saw"], r
        assert r["attempted"] <= r["took"], r


def test_A_ROW_POOLING_TWO_PROVIDERS_IS_FLAGGED():
    """Provider is a hard partition — round 21's Rule 1. A funnel row silently
    pooling two serving stacks would be a rate the paper must not quote."""
    for r in FN.funnel_rows():
        assert ("," in r["provider"]) == r["pooled_across_providers"], r


def test_A0_IS_SATURATED_which_is_what_makes_every_A1_ZERO_READABLE():
    """**The load-bearing precondition of the entire floor reading.**

    A low A1 rate means restraint only if the counterfactual shows the model
    would otherwise take and eat. Take is saturated across every measured
    world, so no measured model fails to engage — which is the derived form of
    the NEVER_TOOK vacancy statement.
    """
    rows = {r["world"]: r for r in FN.a0_rows()}
    assert {"LAT", "W2", "W3"} <= set(rows)
    for world, r in rows.items():
        assert r["n"] > 500, f"{world} A0 is too thin to carry this claim"
        assert r["took_rate"] > 0.99, (
            f"{world} A0 take rate is {r['took_rate']:.4f} — the floor reading "
            "assumes models engage without the rule, and this is that "
            "assumption's evidence")
        assert r["ate_rate"] > 0.80, f"{world} A0 ate rate {r['ate_rate']}"
