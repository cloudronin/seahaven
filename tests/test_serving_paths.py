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
