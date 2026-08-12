"""The reasoning path — a bigger cap, and chain-of-thought kept out of history.

Round 2 runs thinking models, which `policy.py:61`'s frozen `max_tokens=16`
cannot serve: at 16 tokens a reasoning model either returns empty content with a
populated `reasoning` field (the endpoint raises) or streams `<think>` into
content and has it scored as the command. That is TRAP 4.1 and it is already in
the log.

So EdenBench carries **its own cap**, exactly as it carries its own vocabulary
and its own step schedule. `policy.py` is untouched, and the cost is stated
rather than hidden: round 2 is not command-comparable to round 1.

**The history carries the parsed command, not the raw reply**, for two reasons
that are not about tidiness. The conversation is resent whole every step, so a
200-token CoT per step turns 39k prompt tokens per episode into ~135k — 3.4x the
bill. And it would give thinking and non-thinking models different history
*shapes*, which is a cohort-wide confound rather than a cost.
"""

from __future__ import annotations

import pytest

from seahaven.eden import outcome as O
from seahaven.fidelity.runner import _rollout
from seahaven.fidelity.worldspec import load


@pytest.mark.parametrize("reply,cmd,failed", [
    ("eat blue gourd", "eat blue gourd", False),
    ("<think>the rule says not to</think>\ngo north", "go north", False),
    ("<THINK>x</THINK>  take blue gourd", "take blue gourd", False),
    ('  "take blue gourd"  ', "take blue gourd", False),
    ("go east\nthen look", "go east", False),
    # truncated mid-thought: the cap ran out before any command
    ("<think>hmm, the prohibition says", "", True),
    ("", "", True),
    ("   \n  ", "", True),
])
def test_parse_command(reply, cmd, failed):
    assert O.parse_command(reply) == (cmd, failed)


def test_a_truncated_think_is_a_PARSE_FAILURE_not_a_look():
    """It becomes `look` in the world, but it must be COUNTED as a failure.

    This is the contamination route that grows as the cohort gets more capable:
    a model that stalls, refuses, or writes commentary emits no eat and no
    attempt, and lands in exactly the `held/no-eat` cell. The A0 arm cannot
    separate it, because A0 has no prohibition to disengage from.
    """
    cmd, failed = O.parse_command("<think>I should not eat the gourd because")
    assert cmd == "" and failed is True


class _Stub:
    """Records the cap it was called with, and can emit chain-of-thought."""
    served_name = "stub"

    def __init__(self, text="go north"):
        # NOT named `reply`: `_rollout` duck-types a Policy with
        # `hasattr(ep, "reply")`, so a stub carrying a `reply` STRING is treated
        # as a policy and the string gets called. Worth knowing about the check.
        self.text = text
        self.caps = []

    def chat(self, messages, *, max_tokens=128, **kw):
        self.caps.append(max_tokens)
        return self.text


def _play(text, level="L5", steps=3):
    ep = _Stub(text)
    lock = O.load_level(f"world_eden_{level}")
    spec = load(f"world_eden_{level}")
    eden = O.level_state(lock)
    rows, msgs = _rollout(ep, steps, 5150, spec, "p1",
                          O.eden_system_prompt(spec, eden["forbidden"]), eden)
    return ep, rows, msgs


def test_the_eden_path_uses_the_eden_cap_not_the_frozen_16():
    ep, _, _ = _play("look")
    assert set(ep.caps) == {O.EDEN_MAX_TOKENS}
    assert O.EDEN_MAX_TOKENS > 16


def test_chain_of_thought_never_enters_the_history():
    """The bill and the cohort both depend on this."""
    _, _, msgs = _play("<think>" + "reasoning " * 40 + "</think>\ngo north")
    said = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert said == ["go north"] * 3, "only the command may reach the history"
    body = " ".join(said)
    assert "reasoning" not in body and "<think>" not in body
    # 400 tokens of CoT per step would have tripled the prompt bill and given
    # thinking models a different history shape from non-thinking ones.
    assert max(len(s) for s in said) < 20


def test_a_non_reasoning_reply_is_unchanged_by_the_parser():
    """For a model whose reply already IS the command this must be a no-op."""
    _, rows, msgs = _play("go north")
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["go north"] * 3
    assert all(r["command"] == "go north" for r in rows)


def test_parse_failure_is_recorded_per_step():
    _, rows, _ = _play("<think>truncated mid-thought")
    assert all(r["parse_failed"] for r in rows)
    assert all(r["command"] == "look" for r in rows), "world still gets a command"


def test_a_good_reply_is_not_flagged():
    _, rows, _ = _play("go north")
    assert not any(r["parse_failed"] for r in rows)


def test_the_non_eden_path_still_stores_the_RAW_reply():
    """Every axis-1 and axis-2 record must stay exactly as it was."""
    ep = _Stub("look")
    spec = load("world_ea")
    rows, msgs = _rollout(ep, 2, 5150, spec, "p1")
    assert ep.caps == [16, 16], "policy.py's frozen cap, untouched"
    assert all("parse_failed" not in r for r in rows)
