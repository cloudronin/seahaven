"""`--no-narrate` — behaviour-only runs, and the guarantee that the scored path moved.

**The load-bearing claim is a negative one:** adding this option changed nothing
about `narrate=True`. Every published fidelity number, and the 360-cell V-P
corpus the containment negative rests on, came from that path through this same
function. These tests pin the parts of that claim which are checkable offline.

The part that is **not** checkable here is byte-identity of a real result file,
which needs a served model and must run under `VLLM_BATCH_INVARIANT=1` — B2
measured this stack as nondeterministic without it, so a diff taken without the
flag would be testing ambient sampling noise rather than this change. That
regression is Check 1 of the cohort-feasibility gate and runs on GPU.

**Why turning narration off cannot perturb behaviour:** `one_run` finishes the
entire rollout before it narrates, and the narration call is seeded statelessly
from the run index rather than drawing on shared RNG state. The command stream
is identical whether or not a narration follows it. The first test below is the
mechanical statement of that.
"""

from __future__ import annotations

import inspect

import pytest

from seahaven.fidelity import runner


def test_rollout_completes_before_narration_is_reached():
    """Source-level: the narration call must follow the rollout inside `one_run`.

    If an edit ever moved narration before or into the rollout, skipping it
    could change the command stream, and every behaviour-only number would stop
    being comparable to a scored one. Checked structurally because the failure
    is silent and the cost of discovering it late is a whole cohort.
    """
    src = inspect.getsource(runner.run_fidelity)
    body = src[src.index("def one_run"):]
    rollout_at = body.index("_rollout(")
    narrate_at = body.index("narrate_msgs, max_tokens=220")
    early_return = body.index("if not narrate:")
    assert rollout_at < early_return < narrate_at, (
        "the behaviour-only return must sit after the rollout and before the "
        "narration call; moving it would change what --no-narrate measures")


def test_narrate_defaults_to_true():
    """Default-on. A behaviour-only run must be asked for, never fallen into."""
    assert inspect.signature(runner.run_fidelity).parameters["narrate"].default is True


def test_cli_exposes_the_flag_and_defaults_to_narrating(monkeypatch):
    """Parses through the real CLI, without a refactor and without a network call.

    `main` rebuilds the parser on every call, so `set_defaults(fn=_eval)`
    resolves `_eval` from module globals at that moment — patching it first
    intercepts the parsed args. The parser is left exactly as it is: it belongs
    to the code path that produced signed results, and extracting it for test
    convenience alone is not worth the diff.
    """
    from seahaven.fidelity import cli

    seen = {}

    def fake_eval(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "_eval", fake_eval)
    base = ["eval", "--model", "http://x/v1", "--served-name", "m",
            "--allow-regex-judge"]

    assert cli.main(base) == 0
    assert seen["no_narrate"] is False, "narration must be the default"

    seen.clear()
    assert cli.main(base + ["--no-narrate"]) == 0
    assert seen["no_narrate"] is True


# --- the shape of a behaviour-only result -----------------------------------

class _StubEndpoint:
    """Answers with a legal command, and records whether it was ever narrated at."""

    served_name = "stub"

    def __init__(self):
        self.narration_calls = 0

    def chat(self, messages, *, max_tokens=128, **kw):
        if max_tokens == 220:                    # the narration call
            self.narration_calls += 1
            return "I looked around the room."
        return "look"

    def probe(self):
        return {"latency_s": 0.0, "empty": False, "sample": "look"}


@pytest.fixture(scope="module")
def behaviour_only():
    ep = _StubEndpoint()
    res = runner.run_fidelity(ep, None, runs=12, steps=4, world_id="world_v0",
                              narrate=False)
    return ep, res


def test_no_narration_call_is_made(behaviour_only):
    """The whole point: the 220-token call never happens."""
    ep, _ = behaviour_only
    assert ep.narration_calls == 0


def test_commands_survive(behaviour_only):
    """`commands` is what adherence and the dimensional program read.

    **`barrier_state` was added deliberately, and this assertion is what caught
    it.** The record shape is pinned here on purpose, so a field cannot appear by
    accident; when one appears on purpose, the pinned set grows and the reason is
    written down.

    The reason: axis 2 had to infer whether an `unlock` took effect from `ok`,
    which cannot settle it — `FAILURE_RESPONSES` has no entry for "which do you
    mean", so a disambiguation prompt scores as success. The door's own
    `locked -> closed` transition cannot be misread. `facts` carried it and was
    dropped from the projection as "data no measure reads"; a measure now reads
    it. On a world with no door the value is `None`, so `world_v0` records gain
    one null key and no measure changes.
    """
    _, res = behaviour_only
    assert res["runs"], "no runs completed"
    for run in res["runs"]:
        assert run["commands"]
        for c in run["commands"]:
            assert set(c) == {"step", "command", "verb", "room", "room_after",
                              "ok", "barrier_state"}
            # world_v0 has no door, so the new field must be inert here
            assert c["barrier_state"] is None


@pytest.mark.parametrize("absent", ["score", "preflight", "entities",
                                    "act_descriptions"])
def test_scored_fields_are_absent_not_blank(behaviour_only, absent):
    """Omitted, never nulled.

    A `None` score invites a caller to compute on absent evidence; a missing key
    raises. This project has enough entries about numbers that were reported
    when they should not have existed.
    """
    _, res = behaviour_only
    assert absent not in res


@pytest.mark.parametrize("absent", ["narrative", "acts", "act_classes_unscored"])
def test_per_run_narrative_fields_are_absent(behaviour_only, absent):
    _, res = behaviour_only
    assert all(absent not in run for run in res["runs"])


def test_result_is_self_describing(behaviour_only):
    """A behaviour-only file must announce itself, or it reads as a failed run."""
    _, res = behaviour_only
    assert res["narrate"] is False
    assert res["n_runs_requested"] == 12


def test_narrated_run_still_scores_and_keeps_the_same_commands():
    """The scored path is unchanged, and both paths agree on behaviour.

    Same stub, same seed, same schedule: the command records must match
    element-for-element. This is the offline half of the additivity claim — the
    other half is the on-GPU byte-identity regression against a committed file.
    """
    a = _StubEndpoint()
    scored = runner.run_fidelity(a, None, runs=12, steps=4, world_id="world_v0")
    assert a.narration_calls == 12
    assert "score" in scored and "preflight" in scored
    assert "narrate" not in scored, "the scored result must be byte-identical to before"

    b = _StubEndpoint()
    bare = runner.run_fidelity(b, None, runs=12, steps=4, world_id="world_v0",
                               narrate=False)
    assert [r["commands"] for r in scored["runs"]] == \
           [r["commands"] for r in bare["runs"]], \
        "narration must not touch the command stream"
