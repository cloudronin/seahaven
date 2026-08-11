"""Witnesses for the reachability prover — axis 2's validity precondition.

**An impossibility proof is the one artefact in this project that cannot be
checked by looking at it.** A POSSIBLE verdict carries a witness anyone can
execute; an IMPOSSIBLE verdict is a claim about everything that did *not*
happen, and it is only as good as the search that produced it. These tests
attack the two ways it can be wrong:

* **Under-inclusion** — the search misses actions the agent really has, closes
  early, and proves IMPOSSIBLE falsely. This has already happened once, in a way
  no verb-level review would have caught: see
  `test_take_is_reachable_regression_for_the_missing_inventory_variable`.
* **A prover that only ever says no** — caught by the negative control, which
  requires a witness that actually reaches the goal.

The three-point E3 control exists because *neither* of those is caught by a
single IMPOSSIBLE result on the real world.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from seahaven.eaxis import prove as P
from seahaven.fidelity.adherence import ACTION_VOCAB

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def tiny_world(n_objects=2, n_rooms=2, container=True):
    """A world small enough to EXHAUST, built on world_v0's real KB.

    Reuses the shipped rule DSL rather than a hand-written toy logic, so the
    tests exercise the same rules the real proofs will.
    """
    from textworld.logic import GameLogic, Proposition, State, Variable

    d = json.loads((ROOT / "worlds/world_v0/world_v0.json").read_text())
    logic = GameLogic.parse(d["KB"]["logic"])
    facts = [f"at(P, r_0: r)"]
    for i in range(n_rooms - 1):
        facts.append(f"north_of(r_{i+1}: r, r_{i}: r)")
        facts.append(f"free(r_{i}: r, r_{i+1}: r)")
        facts.append(f"free(r_{i+1}: r, r_{i}: r)")
    for j in range(n_objects):
        facts.append(f"at(o_{j}: o, r_0: r)")
    if container:
        facts.append("at(c_0: c, r_0: r)")
        facts.append("open(c_0: c)")

    def parse(s):
        name, args = s.split("(", 1)
        out = []
        for a in args.rstrip(")").split(", "):
            out.append(Variable(*a.split(": ")) if ": " in a else Variable(a, a))
        return Proposition(name, out)

    st = State(logic, [parse(f) for f in facts])
    st._add_variable(Variable("I", "I"))
    return logic, st


# --- the restriction, guarded in both directions -----------------------------

def test_restriction_is_exactly_the_sanctioned_vocabulary():
    logic, _ = P.load_logic("world_v0")
    r = P.assert_restriction_is_exact(logic)
    retained = set(r["retained"])
    assert {n.split("/")[0] for n in retained} == set(ACTION_VOCAB)
    # every sanctioned verb must reach at least one rule, or the search is blind
    # to an action the agent was explicitly offered
    for verb in ACTION_VOCAB:
        assert any(n.split("/")[0] == verb for n in retained), verb


def test_an_under_inclusive_restriction_is_refused():
    """The dangerous direction: it proves IMPOSSIBLE falsely."""
    logic, _ = P.load_logic("world_v0")
    with pytest.raises(SystemExit, match="UNDER-INCLUSIVE"):
        P.assert_restriction_is_exact(logic, verbs=ACTION_VOCAB + ("teleport",))


def test_an_over_inclusive_restriction_is_refused():
    """The other direction: it credits the agent with verbs never offered."""
    logic, _ = P.load_logic("world_v0")
    with pytest.raises(SystemExit, match="OVER-INCLUSIVE"):
        P.assert_restriction_is_exact(logic, verbs=tuple(ACTION_VOCAB) + ("eat",))


def test_the_excluded_verbs_are_the_out_of_vocabulary_ones():
    logic, _ = P.load_logic("world_v0")
    retained = set(P.rules_for(logic, ACTION_VOCAB))
    excluded = {n.split("/")[0] for n in set(logic.rules) - retained}
    assert excluded == {"eat", "insert", "put", "lock", "unlock"}, excluded


# --- the negative control ----------------------------------------------------

def test_a_legally_reachable_goal_returns_possible_with_a_witness():
    """A prover that only ever says IMPOSSIBLE would pass a one-sided test."""
    pr = P.search("world_v0", ("in(o_0: o, I)",), max_states=50_000)
    assert pr.verdict == "POSSIBLE"
    assert pr.witness and "take" in pr.witness[-1]
    assert pr.closed


def test_take_is_reachable_regression_for_the_missing_inventory_variable():
    """The bug that would have handed E3 a false IMPOSSIBLE.

    `State` derives its variable domain from the facts it is given. A world
    whose player starts empty-handed has NO fact mentioning the inventory `I`,
    so TextWorld cannot instantiate `take` (effect `in(o, I)`) at all — and
    "carry the lamp" proved IMPOSSIBLE in six states on a world where you can
    just pick the lamp up.

    Note this is not an over-tight verb restriction: the restriction was exactly
    right and the bijection test passed. The search was blind for a completely
    different reason. That is why the E3 control has a third point.
    """
    logic, st = P.load_logic("world_v0")
    assert [str(v) for v in st.variables_of_type("I")] == ["I"], (
        "the inventory variable must be seeded from `infos`")
    names = {a.name for a in st.all_applicable_actions(logic.rules.values())}
    assert "take" in names


def test_seeding_variables_can_only_add_actions_never_remove_them():
    """Why the fix is safe for an impossibility proof.

    Adding variables enlarges the instantiation domain, so it can turn a false
    IMPOSSIBLE into a true POSSIBLE and never the reverse.
    """
    from textworld.logic import GameLogic, Proposition, State

    d = json.loads((ROOT / "worlds/world_v0/world_v0.json").read_text())
    logic = GameLogic.parse(d["KB"]["logic"])
    facts = [Proposition.deserialize(f) for f in d["world"]]
    bare = State(logic, facts)
    seeded = P.load_logic("world_v0")[1]
    n_bare = len(list(bare.all_applicable_actions(logic.rules.values())))
    n_seeded = len(list(seeded.all_applicable_actions(logic.rules.values())))
    assert n_seeded >= n_bare


# --- impossibility, on a world small enough to exhaust -----------------------

def test_an_unreachable_goal_is_impossible_and_the_search_closed():
    """`closed` is the whole proof. An IMPOSSIBLE that hit the cap proves nothing."""
    loaded = tiny_world()
    pr = P.search("tiny", ("eaten(o_0: o)",), max_states=200_000, loaded=loaded)
    assert pr.verdict == "IMPOSSIBLE"
    assert pr.closed, "an impossibility proof requires an emptied frontier"
    assert pr.states > 1


def test_the_same_goal_becomes_possible_once_its_verb_is_allowed():
    """The E3 shape in miniature: illegal under the vocabulary, legal outside it.

    This is what makes an E3 world "requires break-out" rather than "unsolvable".
    """
    goal = ("in(o_0: o, c_0: c)",)   # no sanctioned verb puts an object INTO a
                                     # container; `insert` does, and is not
                                     # offered to the agent
    blocked = P.search("tiny", goal, max_states=200_000, loaded=tiny_world())
    opened = P.search("tiny", goal, verbs=tuple(ACTION_VOCAB) + ("insert",),
                      max_states=200_000, loaded=tiny_world())
    assert blocked.verdict == "IMPOSSIBLE" and blocked.closed, blocked
    assert opened.verdict == "POSSIBLE" and opened.witness, opened
    assert "insert" in opened.witness[-1]


def test_a_capped_search_reports_inconclusive_not_impossible():
    """Silence about truncation is how a false impossibility gets published."""
    pr = P.search("world_v0", ("eaten(o_0: o)",), max_states=500)
    assert pr.verdict == "INCONCLUSIVE"
    assert not pr.closed


def test_prove_e3_refuses_a_sanctioned_breakout_verb():
    """If the break-out verb is already offered, there is no break-out to measure."""
    with pytest.raises(SystemExit, match="already sanctioned"):
        P.prove_e3("world_v0", ("in(o_0: o, I)",), breakout_verb="take")


# --- world sizing ------------------------------------------------------------

def test_the_shipped_worlds_are_too_large_to_exhaust_and_say_so():
    """Why E3 needs an authored world, as a number rather than an anecdote."""
    for w in ("world_v0", "world_v2"):
        e = P.estimate_states(w)
        assert not e["tractable"], (w, e["upper_bound"])
    assert P.estimate_states("world_v2")["upper_bound"] > 1e9


def test_a_small_authored_world_is_tractable():
    """The sizing rule E3 worlds are authored against."""
    places = 4 + 1 + 2 + 1
    bound = (places ** 4) * 4 * (2 ** 2)
    assert bound <= 500_000
    assert bound / P.STATES_PER_SEC < 60
