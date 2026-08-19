"""THE SERVING-PATH REGISTRY — every path that spends money, one checklist.

**Why this exists, stated once because it is the whole point.**

Blockers 5 and 6 of the matched-pair build were not bugs in a path. They were
bugs in the SPACE BETWEEN paths. The probe path was built separate from the
round path deliberately — that separation IS the corpus-exclusion feature, and
`probe_channel`'s docstring argues for it at length — and the cost of building
something to be separate is that structural fixes do not propagate into it.

So the provider boundary was endorsed as complete at round 21 without anyone
enumerating the paths it had to reach. `run.py` got the attestation guards;
`probe._daily` never imported a round and got none of them; and the test that
was supposed to cover it grepped `run.py` and passed. Three separate
consequences of one missing list.

**This is the list.** Every path that can serve an episode and write a cell,
each asserted against the same invariants. A new path added to `PATHS` inherits
the checklist; a new path NOT added fails `test_EVERY_SERVING_PATH_IS_REGISTERED`,
which discovers them from the source rather than from memory.

**The invariants, and what each one is protecting against:**

  IDENTITY-FROM-EVIDENCE   the cell records what actually served it — a
                           `base_url` or a `served_provider` — never the
                           provider the caller intended. #113.
  PROVIDER-PARTITION       `identity.provider_of` returns that column, so two
                           providers' rates never pool. Round 21's Rule 1.
  ANCHOR-SCOPING           nothing compares a cell to another column's anchor.
                           `probe.epoch_for` / `envelope_for`, blocker 1.
  BUDGET-SOURCE            spend is priced from the serving column's own cohort,
                           not from whichever cohort was in scope. Blocker 4.

A path may legitimately not implement an invariant — `replicate` has no anchors
to scope — and says so with NOT-APPLICABLE and a reason. What it may not do is
be absent.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

NA = "NOT-APPLICABLE"

#: Every path that can serve an episode and write a cell.
PATHS = {
    "round": {
        "module": "vetoworld.commands.run",
        "what": "the round path — `vworld run`, the corpus's own measurements",
        "identity-from-evidence":
            "writes served_provider from the wire (`last_provider`) and both "
            "base_url and endpoint; REFUSES on mismatch or absence",
        "provider-partition": "served_provider is read directly by provider_of",
        "anchor-scoping":
            NA + ": a round is judged against its own pinned cohort, not "
                 "against a daily anchor; there is no cross-column comparison "
                 "to scope",
        "budget-source": "prices from the round module's own COHORT",
    },
    "probe": {
        "module": "vetoworld.commands.probe",
        "what": "the daily probe path — `vworld probe --daily`, the seismograph",
        "identity-from-evidence":
            "REFUSES to serve unless the pinned base_url maps to the pinned "
            "column (probe.served_provider_for), then writes served_provider "
            "and base_url into every cell",
        "provider-partition": "served_provider written explicitly per cell",
        "anchor-scoping":
            "epoch_for(provider, level, arm) / envelope_for(...) — a column "
            "with no earned anchor gets None, never another column's",
        "budget-source": "PB.cohort_for(provider), not PB.COHORT",
    },
    "replicate": {
        "module": "vetoworld.commands.replicate",
        "what": "third-party replication — `vworld replicate`, run by strangers "
                "against endpoints this repo has never seen",
        "identity-from-evidence": "writes base_url and endpoint from the spec",
        "provider-partition":
            "provider_of derives the column from base_url; an unknown host "
            "forms its OWN partition rather than joining together",
        "anchor-scoping":
            NA + ": compares against the replicated round's published band, "
                 "which is supplied with the claim and is not a column anchor",
        "budget-source":
            NA + ": refuses to start without an explicit --budget and prices "
                 "nothing from a cohort",
    },
}

INVARIANTS = ("identity-from-evidence", "provider-partition",
              "anchor-scoping", "budget-source")


def _source(dotted):
    mod = __import__(dotted, fromlist=["_"])
    return inspect.getsource(mod)


def test_EVERY_REGISTERED_PATH_DECLARES_EVERY_INVARIANT():
    """The checklist is complete for each path, or it is not a checklist."""
    for name, spec in PATHS.items():
        assert spec["module"] and spec["what"]
        for inv in INVARIANTS:
            assert inv in spec, f"{name} does not answer {inv!r}"
            answer = spec[inv]
            assert answer and len(answer) > 20, (
                f"{name}.{inv} is not a real answer: {answer!r}")
            if answer.startswith(NA):
                assert ":" in answer, (
                    f"{name}.{inv} claims NOT-APPLICABLE with no reason — "
                    "which is how a gap disguises itself as a decision")


def test_EVERY_SERVING_PATH_IS_REGISTERED():
    """**Discovered from the source, not from memory.**

    This is the test that would have caught blocker 5 before it was written: it
    finds every module that calls `run_fidelity` and requires it to be in
    `PATHS`. A new serving path cannot be added quietly — which is exactly how
    the probe path came to exist without any of the round path's guards.
    """
    root = Path(__file__).resolve().parents[1]
    found = set()
    for path in sorted((root / "vetoworld").rglob("*.py")):
        src = path.read_text()
        #: The definition, and the dry-run recorder in run.py, are not serving.
        if not re.search(r"\brun_fidelity\(", src):
            continue
        if "def run_fidelity" in src:
            continue
        found.add(".".join(path.relative_to(root).with_suffix("").parts))

    registered = {spec["module"] for spec in PATHS.values()}
    missing = found - registered
    assert not missing, (
        f"serving paths that spend money and are not in the registry: "
        f"{sorted(missing)}. Add each to PATHS with its answer to all four "
        "invariants — a path built separate from the others does not inherit "
        "their structural fixes, which is what this registry is for.")


@pytest.mark.parametrize("name", sorted(PATHS))
def test_EVERY_SERVING_PATH_REFUSES_A_BROKEN_INSTALL(name):
    """**The fifth instance, and the registry is what makes it cheap.**

    0.3.0's scheduled job served zero cells because `textworld` is an extra
    and nothing checked for it — 22 identical failures, each caught by a
    per-cell `except` and reported as a provider SERVE_FAIL. A missing
    dependency is a broken install: identical for every cell, knowable before
    the first request, and fixable only by an instruction.

    `run` and `replicate` had the same gap. Enumerating the paths is what
    turned one fix into three, which is the AGENTS.md rule this registry
    exists to serve.
    """
    src = _source(PATHS[name]["module"])
    assert "_serving_deps" in src, (
        f"{name} can serve without checking it CAN serve — it will discover a "
        "missing textworld once per cell and report it as provider trouble")
    assert "require" in src


@pytest.mark.parametrize("name", sorted(PATHS))
def test_EACH_PATH_ACTUALLY_WRITES_IDENTITY_FROM_EVIDENCE(name):
    """**Asserted against the source, on the path that runs.**

    The predecessor of this test grepped `vetoworld.commands.run` for the
    round path's guard strings while claiming to protect the probe column. It
    passed for as long as the probe path was unguarded. Every assertion here
    names the module it is about.
    """
    src = _source(PATHS[name]["module"])
    assert "base_url" in src, (
        f"{name} writes no base_url, so identity.provider_of falls through to "
        "DIRECT_PROVIDER and every cell claims Together")
    assert "served_provider" in src or "provider_of" in src


def test_THE_PROBE_PATH_REFUSES_BEFORE_IT_SPENDS():
    """The probe path's guard has to run before any cell is served and paid
    for. An unattributable cell that already cost money is worse than one that
    was never served."""
    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert "REFUSING TO SERVE" in src
    assert src.index("REFUSING TO SERVE") < src.index("for model, arm, level in todo")


def test_THE_PROBE_PATH_PRICES_FROM_ITS_OWN_COLUMN():
    """Blocker 4. Today DeepInfra models miss `PB.COHORT` and bill $0, so the
    ceiling can never bite; with a MATCHED PAIR the failure inverts and gets
    quieter — a model in both cohorts hits and bills the second column's cells
    at the FIRST column's rate card, producing a plausible wrong number."""
    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert "cohort_for(provider)" in src
    assert not re.search(r"PB\.COHORT\.get|PB\.COHORT\[", src), (
        "the serving path reads the Together cohort directly again")


# --- what 0.3.0's empty day taught, asserted so it cannot recur -------------

def test_A_DAY_THAT_SERVED_NOTHING_IS_SERVE_FAIL_not_PARTIAL():
    """`SERVE_FAIL` was in the status enum and UNREACHABLE from `_daily`: the
    chain fell through to PARTIAL whether one cell failed or all 22 did.

    0.3.0's first scheduled fire served zero cells and reported PARTIAL, which
    reads as "most of a day" and was none of one.
    """
    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert '"SERVE_FAIL" if served == 0' in src, (
        "total failure no longer maps to SERVE_FAIL")
    assert "served = len(todo) - len(failed)" in src


def test_A_DAY_WITH_NO_ROWS_IS_NOT_PUSHED():
    """**An empty push is a claim that nothing happened.** 0.3.0 pushed two
    zero-row files because `read_cells` found nothing, the channel loop never
    ran, and the empty list went up anyway.

    An empty day in the log is indistinguishable from a quiet one, and the
    whole instrument exists to keep those apart.
    """
    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert "REFUSING TO PUSH" in src
    assert src.index("if not rows:") < src.index("url = push_day("), (
        "the refusal must come BEFORE the upload")


def test_THE_SUCCESS_SIGNAL_IS_THE_SERVING_not_the_push():
    """`_daily` returned 0 once the upload succeeded, so a day that served
    nothing still reported success and the job read DAY_RC=0. That is the
    tail-exit-code lesson, in the module whose own docstring cites it."""
    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert 'if status in ("SERVE_FAIL", "BUDGET_REFUSED"):' in src
    assert src.count("return 1") >= 2


def test_THE_SERVING_EXTRA_IS_DECLARED_and_probe_includes_it():
    """`textworld` and `jericho` were undeclared runtime dependencies — the
    scipy defect of 0.2.1, one dependency over. The scheduled job installs
    `vetoworld[probe]`, so `probe` must carry them."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    extras = cfg["project"]["optional-dependencies"]
    serve = " ".join(extras["serve"])
    assert "textworld" in serve and "jericho" in serve
    probe = " ".join(extras["probe"])
    assert "vetoworld[serve]" in probe, (
        "the scheduled job installs vetoworld[probe] and must get the serving "
        "deps with it, or it serves zero cells again")


def test_THE_SERVING_PATH_CREATES_ITS_OWN_OUTPUT_DIRECTORY(tmp_path, monkeypatch):
    """**0.3.2's day served two cells and threw them away.**

    The container is ephemeral and has no `results/`; `write_text` does not
    make parents, so `_daily` crashed at the write — AFTER serving and paying
    for the first cell of each column. The write sits outside the per-cell
    `except`, so it was not even a SERVE_FAIL: an unhandled crash that never
    reached the status logic, the push refusal, or the exit-code branch.

    A serving verb that requires its caller to have made a directory is a verb
    with an unwritten precondition, and unwritten preconditions are what
    ephemeral containers find.
    """
    import inspect

    from vetoworld.commands import probe as CMD

    src = inspect.getsource(CMD._daily)
    assert "root.mkdir(parents=True, exist_ok=True)" in src
    #: Before the first write, not lazily beside it. Compare against the CALL,
    #: not the bare name — the comment above the mkdir explains the failure and
    #: names `write_text`, so a substring test would match the prose.
    assert (src.index("root.mkdir(parents=True")
            < src.index("path_for(model, arm, level).write_text("))


def test_THE_PREFLIGHT_DOES_NOT_PREPARE_WHAT_THE_JOB_DOES_NOT():
    """**The pre-flight hid the defect it existed to catch.**

    It ran `mkdir -p .../results` before serving; the scheduled job does not.
    So it served into a directory that existed only because the pre-flight made
    it, and the real day died on the write.

    Same shape as the dry run that could not see `textworld`: a check that
    diverges from the real path verifies the divergence, not the path.
    """
    from pathlib import Path

    src = Path("scripts/preflight_probe_serving.py").read_text()
    assert "mkdir -p /tmp/preflight/results" not in src, (
        "the pre-flight prepares `results/` again — the job does not, so this "
        "re-hides exactly the failure it was written to expose")
    assert "NOTHING IS PREPARED HERE" in src
