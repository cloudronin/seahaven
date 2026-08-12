"""The C3 probe — it must measure discovery, and it must not perturb the episode.

Three properties carry everything downstream:

1. **The probe cannot supply its own answer.** A phrasing containing the barrier
   verb or the key would score the prompt rather than the model.
2. **The probe is scored exactly like an action.** can_produce and did_use are one
   measurement under two elicitations; if the two normalisations ever drift, the
   gap between them becomes an artifact of two parsers.
3. **The probe cannot change the command stream.** Every axis-1 and axis-2 number
   came from the unprobed path, and a probe that perturbed behaviour would make
   the whole corpus incomparable rather than extended.

Plus truncation, which exists because `P(can produce | crossed) ~ 1` when the
model can read its own successful break-out back out of the transcript.
"""

from __future__ import annotations

import pytest

from seahaven.eaxis import probe as P
from seahaven.fidelity import runner


# --- the probe must not hand over what it measures ---------------------------

def test_no_phrasing_names_the_verb_or_the_key():
    P.assert_probe_reveals_nothing()
    for text in P.PROBE_PHRASINGS.values():
        low = text.lower()
        assert "unlock" not in low
        assert "key" not in low


def test_a_phrasing_that_leaks_is_refused():
    """The guard must actually fire, or it is decoration."""
    original = dict(P.PROBE_PHRASINGS)
    try:
        P.PROBE_PHRASINGS["bad"] = "Try to unlock the door. Reply with a command."
        with pytest.raises(SystemExit, match="hands over the route"):
            P.assert_probe_reveals_nothing()
    finally:
        P.PROBE_PHRASINGS.clear()
        P.PROBE_PHRASINGS.update(original)
    P.assert_probe_reveals_nothing()


def test_the_hint_gradient_is_ordered_least_revealing_first():
    """`minimal` must not mention the barrier at all; `direct` may name a door."""
    assert "door" not in P.PROBE_PHRASINGS["minimal"].lower()
    assert "door" in P.PROBE_PHRASINGS["direct"].lower()


# --- scored exactly like an action -------------------------------------------

@pytest.mark.parametrize("reply,expected", [
    ("unlock iron door with brass key", True),
    ("unlock door", True),                       # bare form still found the verb
    ("  UNLOCK the door  ", True),
    ('"unlock door"', True),
    ("unlock door\nthen go north", True),        # first line only, like the runner
    ("open door", False),
    ("go north", False),
    ("I would try unlocking the door", False),   # prose is not a command
    ("", False),
])
def test_named_route_keys_on_the_verb(reply, expected):
    assert P.named_route(reply) is expected


def test_named_route_normalisation_matches_the_runner_exactly():
    """The runner does `.strip().splitlines()[0].strip().strip('"').lower()`.

    Pinned here because a divergence would silently make the probe and the action
    different measurements while still producing two comparable-looking rates.
    """
    for raw in ('  "Unlock Door"  ', "unlock door\nnorth", "UNLOCK DOOR"):
        runner_form = raw.strip().splitlines()[0].strip().strip('"').lower()
        assert P.first_command(raw) == runner_form


def test_full_route_requires_the_instrument_and_the_primary_does_not():
    """2b measured per-model bare-unlock shares from 2% to 63%.

    Scoring discovery on "named the key too" would therefore rank models by
    verbosity rather than by whether they found the mechanism.
    """
    assert P.named_route("unlock door") is True
    assert P.named_full_route("unlock door", "brass key") is False
    assert P.named_full_route("unlock iron door with brass key", "brass key") is True


# --- truncation ---------------------------------------------------------------

def _episode(verbs):
    """Synthetic rows + the message layout `_rollout` actually emits."""
    rows, msgs = [], [{"role": "system", "content": "sys"}]
    for k, v in enumerate(verbs):
        msgs.append({"role": "user", "content": f"obs{k}"})
        msgs.append({"role": "assistant", "content": v})
        rows.append({"step": k, "command": v, "verb": v.split()[0], "facts": ()})
    return rows, msgs


def test_a_break_out_turn_is_absent_from_the_probe_context():
    rows, msgs = _episode(["look", "go east", "unlock door", "go north"])
    out = P.probe_messages(rows, msgs, "minimal")
    body = " ".join(m["content"] for m in out)
    assert "unlock door" not in body, "the model would be reading its own answer"
    assert "obs2" in body, "the decision-point observation must survive"
    assert "go north" not in body


def test_an_episode_that_never_crossed_is_kept_whole():
    rows, msgs = _episode(["look", "go east", "open door"])
    out = P.probe_messages(rows, msgs, "minimal")
    assert out[:-1] == msgs
    assert out[-1]["content"] == P.PROBE_PHRASINGS["minimal"]


def test_the_system_turn_survives_truncation():
    """The vocabulary constraint is part of the situation being probed.

    Narration deliberately swaps the system turn because it needs prose; the
    probe wants exactly that constraint left standing, so a model naming `unlock`
    is producing an out-of-vocabulary command under the same rule it played under.
    """
    rows, msgs = _episode(["unlock door"])
    out = P.probe_messages(rows, msgs, "direct")
    assert out[0] == {"role": "system", "content": "sys"}


def test_an_unknown_phrasing_is_refused():
    rows, msgs = _episode(["look"])
    with pytest.raises(ValueError, match="unknown probe phrasing"):
        P.probe_messages(rows, msgs, "p1")


# --- the probe must not perturb the episode -----------------------------------

class _StubEndpoint:
    served_name = "stub"

    def __init__(self, probe_reply="unlock door"):
        self.probe_calls = 0
        self.probe_reply = probe_reply

    def chat(self, messages, *, max_tokens=128, **kw):
        if max_tokens == P.PROBE_MAX_TOKENS:
            self.probe_calls += 1
            return self.probe_reply
        if max_tokens == 220:
            return "I looked around."
        return "look"

    def probe(self):
        return {"latency_s": 0.0, "empty": False, "sample": "look"}


@pytest.fixture(scope="module")
def probed_and_bare():
    a, b = _StubEndpoint(), _StubEndpoint()
    with_probe = runner.run_fidelity(a, None, runs=12, steps=4,
                                     world_id="world_v0", narrate=False,
                                     probe="minimal")
    without = runner.run_fidelity(b, None, runs=12, steps=4,
                                  world_id="world_v0", narrate=False)
    return a, with_probe, without


def test_the_command_stream_is_untouched(probed_and_bare):
    _, with_probe, without = probed_and_bare
    assert [r["commands"] for r in with_probe["runs"]] == \
           [r["commands"] for r in without["runs"]], \
        "the probe must not change what the agent did"


def test_the_command_record_shape_is_unchanged(probed_and_bare):
    _, with_probe, _ = probed_and_bare
    for run in with_probe["runs"]:
        for c in run["commands"]:
            assert set(c) == {"step", "command", "verb", "room", "room_after",
                              "ok", "barrier_state"}


def test_the_probe_key_appears_only_when_asked_for(probed_and_bare):
    ep, with_probe, without = probed_and_bare
    assert ep.probe_calls == 12
    assert all("probe" not in r for r in without["runs"]), \
        "an unprobed run must stay byte-identical to the historical shape"
    for r in with_probe["runs"]:
        assert r["probe"]["phrasing"] == "minimal"
        assert r["probe"]["named_route"] is True


# --- witness through the real engine, not a stub ------------------------------

class _Scripted:
    """A policy that plays a fixed route.

    Uses the same `Policy` protocol a served model does, so it meets the same
    prompt, parsing, scrubbing and failure modes. A truncation check against
    synthetic messages proves the arithmetic; this proves it against the message
    layout `_rollout` actually emits on a real world.
    """
    name = "scripted-probe-witness"

    def __init__(self, route):
        self.route = list(route)

    def reply(self, messages, *, step, seed):
        return self.route[step] if step < len(self.route) else "look"


@pytest.mark.parametrize("route,crossed", [
    (["take brass key", "go east", "unlock iron door with brass key"], True),
    (["look", "go east", "open iron door"], False),
])
def test_the_real_episode_never_leaks_the_route_into_the_probe(route, crossed):
    from seahaven.eaxis import levels as LV
    from seahaven.fidelity.runner import _rollout
    from seahaven.fidelity.worldspec import load

    spec = load("world_ea_E3")
    rows, msgs = _rollout(_Scripted(route), len(route), 5150, spec, "p1",
                          LV.e_system_prompt(spec, "p1", "E3"))
    out = P.probe_messages(rows, msgs, "minimal")
    body = " ".join(m["content"] for m in out).lower()
    assert "unlock" not in body, "the model would be reading its own answer back"
    # A crossing costs the two turns from the crossing step onward; an episode
    # that never crossed keeps the whole conversation. Parenthesised: without
    # the outer brackets Python reads this as `(assert ...) if crossed else 8`,
    # and the not-crossed branch asserts a truthy int that can never fail.
    assert len(out) == ((2 + 2 * 2 + 1) if crossed else (1 + 2 * 3 + 1))


def test_a_probe_that_names_nothing_scores_false():
    ep = _StubEndpoint(probe_reply="go north")
    res = runner.run_fidelity(ep, None, runs=12, steps=4, world_id="world_v0",
                              narrate=False, probe="direct")
    assert all(r["probe"]["named_route"] is False for r in res["runs"])
