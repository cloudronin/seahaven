"""The emitted artifacts, the prompt fixtures, and the corpus manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from expedientbench.commands import corpus as CORPUS
from expedientbench.commands import emit
from expedientbench.register import artifacts as A

_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = _ROOT / "expedientbench/fixtures/prompts.json"


class _A:
    def __init__(self, **kw):
        self.artifact = None
        self.action = "status"
        self.results = "results"
        self.__dict__.update(kw)


@pytest.mark.parametrize("art", ["matrix", "floor-mechanisms", "generations",
                                 "limitations", "disclosures", "occasions",
                                 "spend", "seeds"])
def test_every_artifact_renders(art, capsys):
    assert emit.main(_A(artifact=art)) == 0
    assert capsys.readouterr().out.strip()


def test_the_matrix_picks_the_CANONICAL_cell_not_a_diagnostic_block(capsys):
    """**The bug this caught.** Round tags were ranked by STRING LENGTH, so
    "11occ" beat "12" and cogito's LAT row showed its 24-episode occasion probe
    instead of its 48-episode round-12 measurement."""
    emit.main(_A(artifact="matrix"))
    row = [ln for ln in capsys.readouterr().out.splitlines()
           if "cogito" in ln and " LAT " in ln]
    assert row, "cogito must appear at LAT"
    assert "e12" in row[0], f"canonical cell is round 12, got: {row[0]}"
    assert "e11occ" not in row[0]


def test_diagnostic_stages_are_named_and_excluded():
    assert "occasion_probe" in A.DIAGNOSTIC_STAGES
    assert "timing_probe" in A.DIAGNOSTIC_STAGES
    cells = A._cells("LAT")
    assert cells["deepcogito/cogito-v2-1-671b"][0] == "12"


def test_floor_mechanisms_shows_THREE_routes_to_one_number(capsys):
    emit.main(_A(artifact="floor-mechanisms"))
    out = capsys.readouterr().out
    assert "NEVER_TOOK" in out and "BINDS_AT_EAT" in out
    assert "DISTINCT ROUTES" in out
    assert "opposite" in out


def test_limitations_states_the_TEMPERATURE_deviation_and_the_occasion_component(
        capsys):
    emit.main(_A(artifact="limitations"))
    out = capsys.readouterr().out
    assert "temperature 1.0" in out.lower()
    assert "0.319" in out


# --- prompt fixtures -------------------------------------------------------

def test_prompt_fixtures_still_REPRODUCE_from_assembly():
    """Committed assembled bytes per world per generation. If assembly drifts,
    every measured cell was served under a different prompt than the fixtures
    record, and this is what notices."""
    from seahaven.fidelity.runner import run_fidelity
    want = json.loads(FIXTURES.read_text())
    assert want, "no fixtures committed"
    for key, rec in sorted(want.items()):
        level, arm, gen = key.split("|")
        seen = []

        class _R:
            usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

            def chat(self, messages, **kw):
                return "ready"

            def reply(self, messages, *, step, seed):
                seen.append([dict(m) for m in messages])
                return "look"

        run_fidelity(_R(), None, runs=1, steps=30, seed0=5150,
                     world_id=f"world_eden_{level}", narrate=False,
                     eden_level=level, eden_arm=arm,
                     terminal_at_zero=(gen == "gen3"))
        got = hashlib.sha256(
            json.dumps(seen[0], sort_keys=True).encode()).hexdigest()
        assert got == rec["sha256"], f"{key}: assembled prompt changed"


def test_gen1_and_gen3_FIRST_REQUESTS_are_identical():
    """**Round 9's argument, as an artifact.** Terminal death changes what the
    WORLD does at health zero, not what is served — which is why a generation-1
    cell's pre-crossing behaviour carries to generation 3."""
    want = json.loads(FIXTURES.read_text())
    pairs = [(k, k.replace("gen1", "gen3")) for k in want if k.endswith("gen1")]
    assert pairs
    for a, b in pairs:
        assert want[a]["sha256"] == want[b]["sha256"], (a, b)


# --- corpus ----------------------------------------------------------------

def test_corpus_status_matches_the_committed_manifest(capsys):
    assert CORPUS.main(_A(action="status")) == 0
    assert "manifest MATCHES" in capsys.readouterr().out


def test_corpus_digest_is_ORDER_INDEPENDENT(tmp_path):
    """A re-fetch in any order must verify, so the digest cannot depend on
    filesystem ordering."""
    for n, body in (("eden_e9_a__A1__LAT.json", '{"runs":[]}'),
                    ("eden_e9_b__A1__LAT.json", '{"runs":[1]}')):
        (tmp_path / n).write_text(body)
    d1, n1, _ = CORPUS._digest_corpus(tmp_path)
    (tmp_path / "eden_e9_a__A1__LAT.json").touch()      # change mtime only
    d2, n2, _ = CORPUS._digest_corpus(tmp_path)
    assert d1 == d2 and n1 == n2 == 2


def test_corpus_status_REFUSES_when_absent(tmp_path, capsys):
    assert CORPUS.main(_A(action="status", results=str(tmp_path / "no"))) == 2
    assert "NO CORPUS" in capsys.readouterr().out


# --- predictions / corrections / related-work -------------------------------

@pytest.mark.parametrize("art", ["predictions", "corrections", "related-work"])
def test_the_remaining_artifacts_render(art, capsys):
    assert emit.main(_A(artifact=art)) == 0
    assert capsys.readouterr().out.strip()


def test_predictions_reads_the_PINNED_literal_not_a_retyped_copy():
    """A prediction typed into the artifact could be retrofitted to its result.
    Both sides come from elsewhere: the claim from the round module's frozen
    literal, the outcome recomputed from cells."""
    import inspect

    from seahaven.eden import round13 as R13
    src = inspect.getsource(A.predictions)
    assert "R13.PREDICTION" in src
    assert R13.PREDICTION not in src, (
        "the prediction text is inlined here — it must be read from the pin")


def test_predictions_reports_the_derivation_as_FIVE_of_SIX(capsys):
    emit.main(_A(artifact="predictions"))
    out = capsys.readouterr().out
    assert "5 of 6 consistent" in out
    assert "DEVIATES" in out, "the one miss must not be dropped"
    assert "HELD" in out


def test_every_correction_cites_a_REAL_commit(capsys):
    """**The ledger's evidence is the commit.** A hand-written corrections table
    can claim anything; this one fails if a SHA does not resolve."""
    assert emit.main(_A(artifact="corrections")) == 0
    out = capsys.readouterr().out
    assert "COMMIT NOT FOUND" not in out
    assert "0 unverifiable" in out


def test_the_corrections_ledger_includes_the_RETRACTED_HEADLINE():
    claims = [c[0] for c in A.CORRECTIONS]
    assert any("break" in c for c in claims), (
        "round 10's retracted break is the most consequential correction and "
        "must not be missing from its own ledger")


def test_related_work_DECLARES_that_it_is_not_computed(capsys):
    """Every other artifact is derived. This one is asserted, and saying so is
    the difference between a table and a claim about a table."""
    emit.main(_A(artifact="related-work"))
    out = capsys.readouterr().out
    assert "NOT COMPUTED" in out
    assert "ENFORCED" in out and "ASSERTED" in out
