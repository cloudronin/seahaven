"""Did the cell get served by the model it names?

**161 committed cells did not.** `Backend` fixes the request's `"model"` field
at construction from `spec.model` (`fidelity/endpoint.py`), and `_round_cells`
built ONE Backend from `endpoints.toml`, whose `together` entry names a single
model. So the grid's model tuple selected a filename and a price, never a served
model: every cell in rounds 15-19 except the cogito ones was served by
`deepcogito/cogito-v2-1-671b`.

**The tell was in the tokens.** Round 18's eight "different models" agreed to
within 1.09x on prompt tokens and 1.05x on completion, while billing differed
7x because `R.COHORT[model]` charged each at its intended price. Eight genuinely
different models cannot be that tight.

This file does three things: bounds the damage, prevents recurrence, and refuses
to let the affected set grow quietly.
"""

from __future__ import annotations

import pytest

from seahaven.eden._shared import corpus as C

#: **The known-bad set, by round, frozen at discovery.** Not a target to fix by
#: editing — these cells record a model that did not serve them, and the only
#: honest repairs are re-serving or retraction. Frozen so the number cannot grow
#: without a test failing.
MISLABELLED = {"15": 98, "16": 28, "17": 21, "18": 7, "19": 7}
MISLABELLED_TOTAL = 161


def _mismatches():
    out = {}
    for path, cell in C.iter_cells():
        meta = cell.get("meta", {})
        got = C.parse_cell_name(path.name)
        if not got:
            continue
        resolved = meta.get("resolved_model_string")
        if resolved is None:                     # pre-dates the field
            continue
        if resolved != meta.get("served_name"):
            out.setdefault(got["round"], []).append(path.name)
    return out


def test_THE_MISLABELLED_SET_IS_EXACTLY_WHAT_WAS_FOUND_AND_HAS_NOT_GROWN():
    """The damage, bounded. If this fails upward, a NEW cell has been served
    under the same defect and the fix regressed."""
    found = _mismatches()
    counts = {r: len(v) for r, v in found.items()}
    assert counts == MISLABELLED, (
        f"the mislabelled set changed: {counts} vs frozen {MISLABELLED}. "
        "Growing means the per-model backend fix regressed; shrinking means "
        "cells were re-served and this literal should be updated deliberately.")
    assert sum(counts.values()) == MISLABELLED_TOTAL


def test_NO_ROUND_BEFORE_15_IS_AFFECTED():
    """Rounds up to 14 either predate `resolved_model_string` or served one
    model per invocation, where `spec.model` and the intent agreed. The corpus
    before round 15 is not implicated by this defect."""
    found = _mismatches()
    early = [r for r in found if r.rstrip("abtcocmn0123456789") == ""
             and r.isdigit() and int(r) < 15]
    assert not early, f"rounds before 15 are affected too: {early}"


def test_ROUND_18s_TOKEN_COUNTS_SHOW_ONE_MODEL_SERVED_EIGHT_CELLS():
    """**The evidence, kept executable.** This is what made the diagnosis
    certain rather than suspected, and it is the check to run first if anyone
    ever doubts a cohort really was a cohort."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cells = sorted(root.glob("results/eden_e18_*.json"))
    assert len(cells) == 8

    metas = [json.loads(p.read_text())["meta"] for p in cells]
    assert len({m["resolved_model_string"] for m in metas}) == 1, (
        "if this ever becomes 8, round 18 has been re-served correctly and "
        "this test should be retired with the re-serve")

    prompt = [m["usage"]["prompt_tokens"] for m in metas]
    completion = [m["usage"]["completion_tokens"] for m in metas]
    assert max(prompt) / min(prompt) < 1.2, "one model's spread"
    assert max(completion) / min(completion) < 1.2

    billed = [m["billed_usd"] for m in metas]
    assert max(billed) / min(billed) > 5, (
        "billing spread came from the intended models' PRICES, not from what was "
        "served — which is precisely how the defect hid")


def test_THE_RUNNER_NOW_BUILDS_ONE_BACKEND_PER_MODEL():
    """Prevention. Both serving paths must construct the backend inside the
    per-cell loop, and both must refuse when the endpoint resolves something
    other than what was asked for."""
    import inspect

    from vetoworld.commands import probe, run

    for mod, name in ((run, "run"), (probe, "probe")):
        src = inspect.getsource(mod)
        assert "def backend_for(" in src, f"{name} has no per-model backend"
        #: **The comparison must exist; its exact form may move.** It was
        #: `resolved != model`. Round 21 routes through a proxy that echoes the
        #: model with its `:provider` suffix stripped, so the raw strings differ
        #: on every routed cell while naming the same model, and the comparison
        #: became `bare_model(resolved) != bare_model(model)`.
        #:
        #: Asserting the literal string would have made this test fail for a
        #: correct change — and the temptation then is to delete the assertion.
        #: What must hold is that BOTH sides are compared and that any
        #: normalisation goes through the one shared helper, never an inline
        #: `startswith` that would be a second, looser identity rule.
        compared = ("resolved != model" in src
                    or "bare_model(resolved) != ID.bare_model(model)" in src)
        assert compared, (
            f"{name} does not compare the resolved model to the intended one — "
            "that comparison is the guard that would have caught 161 cells")
        assert "startswith" not in src.split("SERVED THE WRONG MODEL")[0][-400:], (
            f"{name} appears to normalise model ids inline; stripping belongs "
            "in `_shared.identity.bare_model` and nowhere else")

    #: **EVERY comparison of a requested id to a resolved one, not just the
    #: one at serve time.**
    #:
    #: The serve-loop check was moved to `bare_model` and the PRE-FLIGHT check
    #: was left comparing raw strings — so the pre-flight refused all seven of
    #: round 21's models, every one of them correctly resolved. Two copies of
    #: one identity comparison diverging, inside the guard built to stop
    #: exactly that.
    #:
    #: It cost nothing because the pre-flight refuses before spending. This
    #: asserts the count instead of the site: if a third comparison appears, it
    #: must go through the helper too.
    src = inspect.getsource(run)
    resolved_cmps = src.count("resolve_model()")
    normalised = src.count("ID.bare_model(")
    assert normalised >= resolved_cmps, (
        f"run.py resolves a model {resolved_cmps} time(s) but normalises "
        f"{normalised} — a comparison somewhere is using raw strings")


def test_A_ROUTED_CELL_IS_VERIFIED_not_MISLABELLED():
    """**The classifier compares bare ids, like every other check. Six sites.**

    A routed cell records the wire id `org/model:provider` as resolved — that
    is genuinely what was sent, and the provider directive is worth keeping —
    against a bare requested name. Compared raw it classifies MISLABELLED, and
    `assert_identity` then refuses to MEASURE it: round 21's cells would have
    been unreadable while being perfectly good data.

    One comparison lived at six sites — serve loop, pre-flight, catalogue
    check, classifier — and each was fixed as it was found, which is the slow
    way. They diverged because a normalisation was introduced without sweeping
    every place the comparison happens. This asserts the end state directly on
    committed cells rather than on any one site's source.
    """
    from seahaven.eden._shared import identity as ID

    routed = [(p, cell["meta"]) for p, cell in C.iter_cells()
              if ":" in (cell.get("meta", {}).get("resolved_model_string") or "")]
    if not routed:
        pytest.skip("no routed cells committed yet")

    for p, m in routed:
        ident = ID.model_identity(m)
        assert ident.status == "VERIFIED", (
            f"{p.name}: routed cell classified {ident.status} — "
            f"requested {ident.requested!r} vs served {ident.served!r}")
        #: The provider directive is PRESERVED in the record, not stripped on
        #: write. Normalisation is a read-time comparison concern only.
        assert ":" in ident.served
        #: And the attestation is present: a routed cell must say who answered.
        assert m.get("served_provider"), f"{p.name} has no provider attestation"
        assert m["served_provider"] == m.get("pinned_provider")


def test_A_SINGLE_BACKEND_CANNOT_SERVE_TWO_MODELS():
    """The property underneath, asserted on the class rather than inferred:
    `Backend` binds one model string for its lifetime."""
    import dataclasses

    from vetoworld.backends.base import EndpointSpec

    a = EndpointSpec(name="x", base_url="https://example.invalid/v1",
                     key_env="NOPE", model="org/model-a")
    b = dataclasses.replace(a, model="org/model-b")
    assert a.model != b.model
    #: The request payload takes the served name from the spec, so two models
    #: require two specs — there is no per-request override.
    import inspect

    import seahaven.fidelity.endpoint as EP
    assert 'payload = {"model": self.served_name' in inspect.getsource(EP), (
        "the request model is no longer bound to the spec — if this changed, "
        "a per-request override may now exist and the fix above can be simpler")
