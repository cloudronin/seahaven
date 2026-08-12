"""The A0 arm, and the pairing that makes the contrast worth having.

Round 1 ran the prohibition arm alone and every ambiguity in the read traced back
to that. A model that never ate the item might have been respecting the rule or
might have been unable to compose `take` then `eat` under pressure; the funnel's
attempted-vs-ate gap was standing in for a competence control it cannot replace,
because a model that never *reaches* produces the same funnel whether it declined
or never got there.

A0 is that control: same world, same larder, same goal, same vocabulary, one
sentence removed. Whatever P(eat) it produces is what reaching for the nearest
high-value item looks like for that model at that level with no rule to obey.

**And the arms must be PAIRED.** Episode i draws `seed0 + i` in both arms, so the
contrast is within-episode for free — but only if both sweeps were launched with
the same `--seed`, and an unpaired headline is a variance decision made by
accident rather than on purpose. The seed is recorded per episode so a read can
verify it instead of trusting it.
"""

from __future__ import annotations

import pytest

from seahaven.eden import outcome as O
from seahaven.fidelity.worldspec import load


def _prompts(level="L3"):
    spec = load(f"world_eden_{level}")
    item = O.level_state(O.load_level(f"world_eden_{level}"))["forbidden"]
    return (O.eden_system_prompt(spec, item, arm="A1"),
            O.eden_system_prompt(spec, item, arm="A0"), item)


def test_the_arms_differ_by_EXACTLY_the_prohibition_sentence():
    """Anything else that differs is a second manipulation riding along."""
    a1, a0, item = _prompts()
    assert a1 != a0
    assert a0 == a1.replace(" " + O.PROHIBITION.format(item=item), ""), (
        "removing the prohibition sentence from A1 must yield A0 exactly; a "
        "second difference makes the arm contrast uninterpretable")


def test_A0_never_names_the_forbidden_item_in_the_PROMPT():
    """Naming it would make it salient without prohibiting it — a third arm."""
    _, a0, item = _prompts()
    assert item not in a0
    assert "not to be eaten" not in a0


def test_A0_keeps_the_goal_the_vocabulary_and_the_world():
    """A0 is a control, not a different task."""
    a1, a0, _ = _prompts()
    assert O.EDEN_GOAL in a0
    for verb in O.EDEN_VOCAB:
        assert (verb in a0) == (verb in a1)
    assert len(a0) < len(a1), "A0 is A1 minus a sentence"


@pytest.mark.parametrize("level", ("L1", "L5", "NEC"))
def test_the_arm_prompt_is_identical_across_LEVELS(level):
    """A level is its larder. If the text moved too, pressure and prompt would
    be confounded and the ladder would not be a ladder."""
    base_a1, base_a0, _ = _prompts("L3")
    a1, a0, _ = _prompts(level)
    assert (a1, a0) == (base_a1, base_a0)


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    spec = load("world_eden_L3")
    with pytest.raises(ValueError, match="unknown eden arm"):
        O.eden_system_prompt(spec, "blue gourd", arm="A2")


def test_episode_i_draws_the_SAME_seed_in_both_arms():
    """The pairing, at the arithmetic that produces it.

    `_rollout` is handed `seed0 + i`, and `EdenPolicy` derives every per-step
    seed from that. So two sweeps launched with the same `--seed` are paired
    episode-for-episode, and two launched with different ones are not — with
    nothing in the artifacts to say which happened, until now.
    """
    from seahaven.eden.outcome import EdenPolicy

    class _Ep:
        served_name = "stub"

        def __init__(self):
            self.seeds = []

        def chat(self, messages, *, max_tokens=128, temperature=0.0,
                 seed=None, stop=None):
            self.seeds.append(seed)
            return "look"

    def sweep(seed0, n=4, steps=3):
        out = []
        for i in range(n):
            ep = _Ep()
            pol = EdenPolicy(ep)
            for step in range(steps):
                pol.reply([], step=step, seed=seed0 + i)
            out.append(ep.seeds)
        return out

    assert sweep(5150) == sweep(5150), "same --seed must pair the arms"
    assert sweep(5150) != sweep(9999), "and a different one must not"


class _StubEndpoint:
    """Not named `reply`: `_rollout` duck-types a Policy with `hasattr(ep,
    "reply")`, so anything carrying that name is treated as a policy."""
    served_name = "stub"

    def __init__(self):
        self.systems = []

    def chat(self, messages, *, max_tokens=128, temperature=0.0,
             seed=None, stop=None):
        for m in messages:
            if m["role"] == "system":
                self.systems.append(m["content"])
        return "look"


def _sweep(arm, seed0=5150, runs=12):
    # runs MUST equal the schedule length -- the runner refuses an uneven mix,
    # and EDEN_STEP_SCHEDULE is 12 flat entries.
    from seahaven.fidelity.runner import run_fidelity
    ep = _StubEndpoint()
    res = run_fidelity(ep, None, runs=runs, steps=30, seed0=seed0,
                       world_id="world_eden_L5", narrate=False,
                       eden_level="L5", eden_arm=arm)
    return ep, res


def test_the_arms_are_PAIRED_end_to_end_through_the_real_runner():
    """Not a source-text check: both arms are driven through `run_fidelity` and
    the artifacts are compared. Episode i must carry the same seed in both."""
    _, a1 = _sweep("A1")
    _, a0 = _sweep("A0")
    s1 = [r["seed"] for r in sorted(a1["runs"], key=lambda r: r["run"])]
    s0 = [r["seed"] for r in sorted(a0["runs"], key=lambda r: r["run"])]
    assert s1 == s0 == list(range(5150, 5162))

    # ...and a mismatched launch must be VISIBLE in the artifacts, which is the
    # whole reason the seed is recorded.
    _, bad = _sweep("A0", seed0=9999)
    assert [r["seed"] for r in sorted(bad["runs"], key=lambda r: r["run"])] != s1


def test_the_prompt_the_ENDPOINT_ACTUALLY_SAW_differs_by_arm():
    """The arm has to survive the whole plumbing, not just the prompt builder."""
    ep1, _ = _sweep("A1")
    ep0, _ = _sweep("A0")
    item = O.level_state(O.load_level("world_eden_L5"))["forbidden"]
    assert ep1.systems and ep0.systems
    assert all(item in s for s in ep1.systems)
    assert all(item not in s for s in ep0.systems)
    assert len(set(ep1.systems)) == 1, "one prompt per arm, across every step"
    assert len(set(ep0.systems)) == 1
