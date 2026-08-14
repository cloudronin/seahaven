"""`vworld verify` — and the drift test that proves it can fail.

A verifier that has never failed is not known to work. This perturbs one cell in
a scratch copy of the corpus and asserts `verify` notices AND names the figure.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vetoworld.commands import verify as V
from vetoworld.register import CLAIMS, recompute_all

_ROOT = Path(__file__).resolve().parents[1]


def test_verify_is_green_on_the_committed_corpus(capsys):
    assert V.main() == 0
    out = capsys.readouterr().out
    assert f"all {len(CLAIMS)} figures recompute" in out


def test_every_claim_recomputes_and_none_ERRORS():
    """A claim that raises is as bad as one that drifts — it means the figure is
    not recomputable, and an unrecomputable figure does not belong in a paper."""
    for c, got, ok in recompute_all():
        assert not str(got).startswith("ERROR"), f"{c.fid}: {got}"
        assert ok, f"{c.fid}: published {c.value!r}, recomputed {got!r}"


def test_SYNTHETIC_DRIFT_makes_verify_fail_and_NAME_the_figure(tmp_path, capsys,
                                                              monkeypatch):
    """**The check that the verifier can fail.**

    Copy the corpus, flip one episode's `ate_forbidden` in Terra's LAT cell, and
    `verify` must exit nonzero and print `terra.floor.total` — not merely "some
    figure changed".
    """
    scratch = tmp_path / "results"
    shutil.copytree(_ROOT / "results", scratch)
    target = scratch / "eden_e13_gpt-5.6-terra__A1__LAT.json"
    d = json.loads(target.read_text())
    ep = d["runs"][0]
    for c in ep["commands"]:
        if c.get("fb_held"):
            c["ate_forbidden"] = True     # one fabricated eat
            break
    target.write_text(json.dumps(d))

    monkeypatch.chdir(tmp_path)
    from seahaven.eden._shared import corpus as C
    monkeypatch.setattr(C, "RESULTS", Path("results"))

    rc = V.main()
    out = capsys.readouterr().out
    assert rc == 1, "verify passed on a corpus with a fabricated eat"

    # **Assert on the FAILURE line, not on the presence of the id.** `verify`
    # prints every row, so `"terra.route.LAT" in out` is true whether or not it
    # drifted — the first version of this test asserted exactly that and would
    # have passed against a verifier that never fails.
    failures = [ln for ln in out.splitlines() if "manuscript says" in ln]
    assert failures, "a drift must print both values, not just a marker"
    assert any("terra.route.LAT" in ln for ln in failures), failures

    # And it must be TARGETED: flipping one episode moves one figure, not all.
    # If the scratch corpus were unreadable every claim would 'drift' and this
    # test would pass for entirely the wrong reason.
    from vetoworld.register import recompute_all
    rows = recompute_all()
    drifting = [c.fid for c, _g, ok in rows if not ok]
    erroring = [c.fid for c, g, _ok in rows if str(g).startswith("ERROR")]
    assert erroring == [], f"scratch corpus unreadable: {erroring}"
    assert len(drifting) == 1, f"expected one targeted drift, got {drifting}"


def test_the_register_records_MEASURED_vs_DERIVED(capsys):
    """A derived figure rests on an identity checked six times with one miss.
    Quoting it without that provenance is the thing the register prevents."""
    derived = [c for c in CLAIMS if not c.measured]
    assert derived, "the derivation tally is a derived figure"
    V.main()
    out = capsys.readouterr().out
    assert "DERIVED" in out
    assert "five consistent, one off by 0.29" in out


def test_every_claim_names_the_cells_it_consumes():
    """A figure whose inputs are unstated cannot be audited."""
    for c in CLAIMS:
        assert c.cells, f"{c.fid} does not say which cells it uses"
        assert c.generation, f"{c.fid} does not say which generation"
