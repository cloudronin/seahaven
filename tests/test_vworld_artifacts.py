"""The emitted artifacts, the prompt fixtures, and the corpus manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vetoworld.commands import corpus as CORPUS
from vetoworld.commands import emit
from vetoworld.register import artifacts as A

_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = _ROOT / "vetoworld/fixtures/prompts.json"


class _A:
    def __init__(self, **kw):
        self.artifact = None
        self.action = "status"
        self.results = "results"
        self.__dict__.update(kw)


def test_ARTIFACTS_and_the_DISPATCH_TABLE_cannot_fork():
    """**The single-source assertion, added at the third instance.**

    `ARTIFACTS` (what `--help` lists) and the dispatch dict were separate hand-
    maintained literals. An artifact in the dict but not the tuple never got
    rendered by a test; one in the tuple but not the dict would advertise a verb
    that exits 1. Neither is detectable by reading either list alone.
    """
    assert set(emit.ARTIFACTS) == set(emit.registry())


@pytest.mark.parametrize("art", sorted(emit.ARTIFACTS))
def test_every_artifact_renders(art, capsys):
    """Parametrized over `ARTIFACTS` itself rather than a retyped subset. The
    retyped version covered 11 of 14 — `correlations`, `convergence` and
    `occasion-health` all escaped it."""
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


# --- corpus fetch ----------------------------------------------------------

def test_fetch_REFUSES_to_clobber_a_corpus_that_is_already_there(tmp_path,
                                                                 capsys):
    """A replicator who already has cells and re-runs fetch out of habit should
    not silently lose a corpus they may have verified."""
    from vetoworld.commands import corpus as CO

    (tmp_path / "eden_e14_x__A1__LAT2.json").write_text('{"runs":[]}')

    class _A:
        action, results, force = "fetch", str(tmp_path), False
        repo = CO.DATASET
    assert CO.main(_A()) == 1
    assert "Refusing to overwrite" in capsys.readouterr().out


@pytest.mark.parametrize("code", [401, 403, 404])
def test_fetch_EXPLAINS_an_unreadable_dataset_instead_of_leaking_the_status(
        code, tmp_path, capsys, monkeypatch):
    """**HuggingFace answers 401, not 404, for a dataset that does not exist.**

    Checked against the live API: a nonsense repo and (before it was published)
    the real one both returned 401, because HF will not leak whether a private
    dataset exists. The first version of this command handled 404 alone, so the
    state it was actually in produced a bare "Unauthorized" — accurate and
    useless. All three codes are one case, and the message must not claim to
    know WHICH of the two conditions it hit, because HF does not say.
    """
    import urllib.error

    from vetoworld.commands import corpus as CO

    def _raise(_repo):
        raise urllib.error.HTTPError(_repo, code, "nope", {}, None)
    monkeypatch.setattr(CO, "_listing", _raise)

    class _A:
        action, results, force = "fetch", str(tmp_path / "none"), False
        repo = CO.DATASET
    assert CO.main(_A()) == 2
    out = capsys.readouterr().out
    assert f"HTTP {code}" in out
    assert "does not exist or it is private" in out
    assert "not evidence" in out, "must not claim to know which of the two"
    assert "--repo" in out, "and must name the way out"


def test_fetch_INSTALLS_NOTHING_when_the_digest_does_not_match(tmp_path,
                                                              capsys,
                                                              monkeypatch):
    """**The check the whole command exists for.** A corpus that fetched cleanly
    but hashes differently is not the one the manuscript was computed from, and
    installing it would make `verify` report drift caused by the download."""
    from vetoworld.commands import corpus as CO

    monkeypatch.setattr(CO, "_listing", lambda repo: ["eden_e14_x__A1__LAT2.json"])
    monkeypatch.setattr(CO, "_get", lambda url, timeout=60: b'{"runs": []}')
    monkeypatch.setattr(CO, "MANIFEST", tmp_path / "m.json")
    (tmp_path / "m.json").write_text(json.dumps({"digest": "0" * 64, "cells": 1}))

    dest = tmp_path / "results"

    class _A:
        action, results, force = "fetch", str(dest), False
        repo = "someone/mirror"
    assert CO.main(_A()) == 1
    out = capsys.readouterr().out
    assert "DIGEST MISMATCH" in out
    assert not dest.exists(), "a mismatched corpus must not be installed"
    assert (tmp_path / "results.partial").exists(), "and must be left for inspection"


def test_fetch_INSTALLS_when_the_digest_matches(tmp_path, capsys, monkeypatch):
    """The other half: the check has to pass on a good fetch, or it is just a
    command that always fails."""
    from vetoworld.commands import corpus as CO

    body = b'{"runs": []}'
    monkeypatch.setattr(CO, "_listing", lambda repo: ["eden_e14_x__A1__LAT2.json"])
    monkeypatch.setattr(CO, "_get", lambda url, timeout=60: body)
    monkeypatch.setattr(CO, "MANIFEST", tmp_path / "m.json")

    h = hashlib.sha256()
    h.update(b"eden_e14_x__A1__LAT2.json")
    h.update(hashlib.sha256(body).digest())
    (tmp_path / "m.json").write_text(json.dumps({"digest": h.hexdigest()}))

    dest = tmp_path / "results"

    class _A:
        action, results, force = "fetch", str(dest), False
        repo = "someone/mirror"
    assert CO.main(_A()) == 0
    assert (dest / "eden_e14_x__A1__LAT2.json").read_bytes() == body
    assert not (tmp_path / "results.partial").exists(), "staging must be gone"
    assert "MATCHES" in capsys.readouterr().out


def test_fetch_pulls_NO_provider_sdk_into_the_import_path():
    """The package declares zero runtime dependencies so it installs anywhere a
    model is being served. The verb a replicator runs FIRST must not be the one
    that breaks that."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "vetoworld/commands/corpus.py"
    names = set()
    for n in ast.walk(ast.parse(src.read_text())):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module.split(".")[0])
    assert "huggingface_hub" not in names and "requests" not in names, names


def test_get_RETRIES_transient_failures_instead_of_dying_mid_download(
        monkeypatch):
    """**The bug the first live fetch found.** 259 sequential requests make a
    transient failure the normal case, and the first version had no retry: a real
    download died on `RemoteDisconnected` partway through, with a urllib
    traceback where a replicator needed a progress line."""
    import http.client

    from vetoworld.commands import corpus as CO

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.RemoteDisconnected("closed")
        class _R:
            def read(self): return b"ok"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _R()

    monkeypatch.setattr(CO.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(CO.time, "sleep", lambda s: None)
    assert CO._get("https://example/x") == b"ok"
    assert calls["n"] == 3


def test_get_HONOURS_retry_after_rather_than_its_own_backoff(monkeypatch):
    """The far end saying how long to wait is better information than any curve
    computed here — the same rule the serving client follows on 429."""
    from vetoworld.commands import corpus as CO

    waited, calls = [], {"n": 0}

    def limited(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CO.urllib.error.HTTPError(
                "u", 429, "slow down", {"Retry-After": "7"}, None)
        class _R:
            def read(self): return b"ok"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _R()

    monkeypatch.setattr(CO.urllib.request, "urlopen", limited)
    monkeypatch.setattr(CO.time, "sleep", waited.append)
    assert CO._get("https://example/x") == b"ok"
    assert waited == [7.0], waited


def test_a_failed_fetch_KEEPS_the_partial_and_names_the_cell(tmp_path, capsys,
                                                             monkeypatch):
    """A re-run must resume, not restart, on a connection already shown to be
    unreliable — and the message has to name a cell, not a socket."""
    from vetoworld.commands import corpus as CO

    monkeypatch.setattr(CO, "_listing", lambda repo: ["eden_e1_a.json",
                                                      "eden_e2_b.json"])
    n = {"i": 0}

    def one_then_fail(url, timeout=60):
        n["i"] += 1
        if n["i"] > 1:
            raise CO.TransientFetchError("boom — after 5 attempts")
        return b'{"runs": []}'
    monkeypatch.setattr(CO, "_get", one_then_fail)
    monkeypatch.setattr(CO, "MANIFEST", tmp_path / "absent.json")

    dest = tmp_path / "results"

    class _A:
        action, results, force = "fetch", str(dest), False
        repo = "someone/mirror"
    assert CO.main(_A()) == 2
    out = capsys.readouterr().out
    assert "eden_e2_b.json" in out and "Re-run to resume" in out
    assert not dest.exists(), "nothing may be installed on a failed fetch"
    assert (tmp_path / "results.partial" / "eden_e1_a.json").exists()


def test_a_resumed_fetch_SKIPS_what_is_already_staged(tmp_path, capsys,
                                                      monkeypatch):
    from vetoworld.commands import corpus as CO

    stage = tmp_path / "results.partial"
    stage.mkdir()
    (stage / "eden_e1_a.json").write_text('{"runs": []}')
    monkeypatch.setattr(CO, "_listing", lambda repo: ["eden_e1_a.json",
                                                      "eden_e2_b.json"])
    pulled = []

    def track(url, timeout=60):
        pulled.append(url.rsplit("/", 1)[-1])
        return b'{"runs": []}'
    monkeypatch.setattr(CO, "_get", track)
    monkeypatch.setattr(CO, "MANIFEST", tmp_path / "absent.json")

    class _A:
        action, results, force = "fetch", str(tmp_path / "results"), False
        repo = "someone/mirror"
    assert CO.main(_A()) == 0
    assert pulled == ["eden_e2_b.json"], pulled
    assert "resuming: 1 cell(s) already staged" in capsys.readouterr().out
