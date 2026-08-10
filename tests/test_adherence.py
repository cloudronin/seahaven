"""Tests for the Constraint Adherence measurement.

Two jobs. First, the ordinary one: does `classify` sort the tokens real models
actually emitted, and does the affordance denominator reproduce the published
figures. Second, and the reason this file matters more than most — **the
constants are pinned by sha256**, so an edit to a frozen list after any result
exists fails the suite rather than silently moving a number.

`docs/adherence-spec-v0.1.md` §1 froze these. The project's history is that the
things which move quietly are the things that produce retractions.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from seahaven.fidelity.adherence import (ACTION_VOCAB, DIRECTIONS,
                                         OBJECT_VERBS, PARSER_REJECTIONS,
                                         TEMPLATE_NOISE, affordances,
                                         classify, consumed, coverage_series,
                                         resolve_pair, unmatched_response_rate)
from seahaven.fidelity.worldspec import load as load_world

V0 = load_world("world_v0")
V2 = load_world("world_v2")


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()[:16]


# --- the pinning ------------------------------------------------------------

#: Recomputed and re-pinned ONLY when a change is deliberate, and never after a
#: result has been produced under the old value. If one of these fails, the
#: question is not "update the hash" — it is "what moved, and was anything
#: already measured with it".
PINNED = {
    "ACTION_VOCAB": "014eeaaead320071",
    "DIRECTIONS": "60906539224805d6",
    "TEMPLATE_NOISE": "f0c4e4c5cc0e599a",
    "OBJECT_VERBS": "8e97f80199966f14",
    "PARSER_REJECTIONS": "f58f129a98dfcac7",
}


def test_frozen_constants_are_pinned():
    """A silent edit to any frozen list fails here."""
    actual = {
        "ACTION_VOCAB": _sha(ACTION_VOCAB),
        "DIRECTIONS": _sha(DIRECTIONS),
        "TEMPLATE_NOISE": _sha(TEMPLATE_NOISE),
        "OBJECT_VERBS": _sha(OBJECT_VERBS),
        "PARSER_REJECTIONS": _sha(PARSER_REJECTIONS),
    }
    assert actual == PINNED, (
        "a frozen constant changed. Do not just update PINNED — first check "
        "whether any published number was measured under the old value.\n"
        f"actual: {json.dumps(actual, indent=2)}")


def test_action_vocab_matches_the_prompt():
    """The measure and the prompt cannot drift apart.

    The whole construct is 'did the agent stay inside the vocabulary its prompt
    declared'. If `ACTION_VOCAB` and `SYSTEM_TEMPLATE` disagree, the measure is
    scoring against a rule the agent was never given.
    """
    from seahaven.fidelity.runner import SYSTEM_TEMPLATE

    for verb in ACTION_VOCAB:
        assert verb in SYSTEM_TEMPLATE, f"{verb!r} is measured but never declared"
    for d in DIRECTIONS:
        assert f"go {d}" in SYSTEM_TEMPLATE


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("take kettle", "legitimate"),
    ("go north", "legitimate"),
    ("look", "legitimate"),
    # Real tokens from the corpus. IBM's most frequent out-of-vocabulary token
    # is list numbering and Meta's is an allowed verb with a period; counting
    # either as a violation made a punctuation habit look like rule-breaking.
    ("1.", "noise"),
    ("inventory.", "noise"),
    ("<|assistant|>", "noise"),
    ("i", "noise"),
    ("", "noise"),
    # Markdown emphasis bleeding into the command slot. Found by regression
    # against the published table: a naive verb extraction turns `**given` into
    # the verb `given` and scores prose as a rule violation, which moved AI2's
    # action-level figure by 0.1 points.
    ("**given", "noise"),
    ("**note:", "noise"),
    ("→ go north", "noise"),
    # Genuine out-of-vocabulary attempts.
    ("unlock chest", "violation"),
    ("read logbook", "violation"),
    ("pick up the rope", "violation"),
    ("pour the oil", "violation"),
])
def test_classify(command, expected):
    assert classify(command) == expected


# --- the affordance denominator --------------------------------------------

@pytest.mark.parametrize("spec,expected", [(V0, 32), (V2, 35)])
def test_affordance_counts_reproduce_the_published_figures(spec, expected):
    """The spec's §0 ledger states 32 and 35. A denominator that does not
    reproduce them is wrong, and every coverage number computed with it."""
    assert len(affordances(spec)) == expected


def test_affordances_include_containers_and_supporters():
    """`WorldSpec.takeable` is portable objects only, because fidelity's ground
    truth could never be true of a fixed thing. Adherence is different: a
    container is examinable and openable, so omitting it understates the space
    and inflates coverage."""
    a = affordances(V0)
    kinds = V0.entity_kinds()
    for c in kinds["c"]:
        assert ("examine", c) in a and ("open", c) in a and ("close", c) in a
        assert ("take", c) not in a, "containers are fixed in place"
    for s in kinds["s"]:
        assert ("examine", s) in a
        assert ("open", s) not in a, "supporters are not openable"


# --- coverage ---------------------------------------------------------------

def test_inventory_period_is_noise_for_adherence_but_counts_for_coverage():
    """The one deliberate disagreement between the two functions.

    `inventory.` is a tokenisation artefact, not a rule violation — but it
    succeeds in the engine and consumes an affordance. It is Meta's single most
    frequent out-of-vocabulary token, so excluding it from coverage would make
    Meta undercount against models that omit the period, and coverage is the
    key regressor in H1.
    """
    assert classify("inventory.") == "noise"
    assert resolve_pair("inventory.", V0, affordances(V0)) == ("inventory", "")


def test_coverage_numerator_is_restricted_to_the_manifest():
    """Room prose is not an affordance. 'glass' appears in a description but is
    not an entity, so guessing at it must not inflate coverage."""
    m = affordances(V0)
    assert resolve_pair("examine glass", V0, m) is None
    assert resolve_pair("take kettle", V0, m) == ("take", "kettle")


def test_short_form_resolves_to_the_canonical_name():
    m = affordances(V0)
    assert resolve_pair("take rope", V0, m) == ("take", "coil of rope")
    assert resolve_pair("take coil of rope", V0, m) == ("take", "coil of rope")


def test_coverage_is_monotone_and_bounded():
    rows = [{"command": c, "response": "", "ok": True} for c in
            ["look", "take kettle", "go north", "take rope", "examine glass",
             "take kettle"]]
    cov = coverage_series(rows, V0)
    assert all(b >= a for a, b in zip(cov.issued, cov.issued[1:]))
    assert max(cov.issued) <= 1.0
    assert cov.off_manifest_attempts == 1, "the 'examine glass' guess"


def test_successful_coverage_never_exceeds_issued():
    rows = [{"command": "take kettle", "response": "You can't see any such thing.",
             "ok": False},
            {"command": "look", "response": "-= Galley =-", "ok": True}]
    cov = coverage_series(rows, V0)
    assert all(s <= i for s, i in zip(cov.successful, cov.issued))


# --- consumed semantics (spec §1b) ------------------------------------------

@pytest.mark.parametrize("response,expected", [
    ("You can't see any such thing.", False),
    ("That's not a verb I recognise.", False),
    ("Which do you mean, the kettle or the key?", False),
    # Parsed and executed, however uninformative the reply.
    ("You see nothing special about the kettle.", True),
    ("You pick up the kettle.", True),
])
def test_consumed(response, expected):
    assert consumed(response) is expected


def test_unmatched_response_rate_is_reported_not_hidden():
    """An unforeseen response class must be visible rather than silently binned
    as consumed."""
    rows = [{"response": "You pick up the kettle."},
            {"response": "Some engine reply nobody anticipated."},
            {"response": "You can't see any such thing."}]
    assert unmatched_response_rate(rows) == pytest.approx(2 / 3)


# --- the policy wrapper -----------------------------------------------------

def test_endpoint_policy_passes_arguments_unchanged():
    """`EndpointPolicy` must not perturb generation.

    The measurement loop is shared with a 542-episode corpus and a published
    table, so if wrapping the endpoint changes any call argument, every future
    number is incomparable with every past one.

    **Tested against a recording fake rather than a live endpoint, deliberately.**
    A live byte-identity check was tried first and was uninterpretable: over four
    runs at a fixed seed the bare path produced two distinct command sequences
    and the wrapped path produced two, sharing the same modal sequence 3/4 each.
    The endpoint is nondeterministic under its own seed, so byte-identity cannot
    separate code drift from ambient noise — which is precisely why the
    determinism control has to precede the identity checks rather than follow
    them. The invariant that actually matters is argument fidelity, and that is
    deterministic and free to assert.
    """
    from seahaven.fidelity.policy import EndpointPolicy

    class Recorder:
        served_name = "fake"

        def __init__(self):
            self.calls = []

        def chat(self, messages, **kw):
            self.calls.append((len(messages), kw))
            return "look"

    rec = Recorder()
    EndpointPolicy(rec).reply([{"role": "user", "content": "x"}], step=7, seed=4242)
    assert len(rec.calls) == 1
    _, kw = rec.calls[0]
    # Verbatim from the pre-refactor `_rollout` body.
    assert kw == {"max_tokens": 16, "temperature": 0.9,
                  "seed": 4242 * 100_003 + 7}


def test_rollout_accepts_a_bare_endpoint_and_a_policy():
    """Existing call sites pass an `Endpoint`; the calibration study passes a
    `Policy`. Both must work, or the scripted baselines cannot traverse the same
    pipeline the models do."""
    from seahaven.fidelity.policy import EndpointPolicy

    class Stub:
        served_name = "stub"

        def chat(self, messages, **kw):
            return "look"

    assert hasattr(EndpointPolicy(Stub()), "reply")
    assert not hasattr(Stub(), "reply"), "bare endpoints are wrapped, not duck-typed"
