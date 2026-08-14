"""Packaging: the pins must survive an install, and an absent corpus must not
look like a drifting manuscript.

Both of these were found by building a wheel and running it outside the repo,
not by reading the config.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seahaven.eden._shared import corpus as C

_ROOT = Path(__file__).resolve().parents[1]


def test_worlds_ships_as_package_data():
    """**The pins read repo-root-relative paths.** From an installed wheel that
    root is site-packages, so the locks must land beside the code or every pin
    raises FileNotFoundError and `vworld verify` cannot run at all."""
    cfg = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    inc = cfg["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "worlds*" in inc
    pdata = cfg["tool"]["setuptools"]["package-data"]["worlds"]
    assert any("BUILD.lock.json" in g for g in pdata)
    assert any(g.endswith("*.z8") for g in pdata), (
        "the .z8 binaries ship too: splitting locks from the worlds they "
        "describe would let the two drift")
    assert (_ROOT / "worlds/__init__.py").exists()


def test_repo_root_resolves_from_the_LIBRARY_not_the_cwd():
    """A command run from another directory must still find the worlds."""
    root = C.repo_root()
    assert (root / "seahaven" / "eden").is_dir()
    assert (root / "worlds").is_dir()


def test_corpus_present_distinguishes_ABSENT_from_EMPTY_RESULT(tmp_path):
    assert C.corpus_present() > 200
    assert C.corpus_present(tmp_path) == 0


def test_verify_REFUSES_on_an_absent_corpus_rather_than_claiming_drift(
        tmp_path, monkeypatch, capsys):
    """**The serious bug the wheel exposed.**

    Computing over zero cells reported every figure as changed — telling a
    replicator the manuscript is wrong when the truth is they have no data.
    Exit 2 is a distinct state from exit 1 (drift), and the message says so.
    """
    from vetoworld.commands import verify as V
    monkeypatch.setattr(C, "RESULTS", tmp_path / "nothing")
    rc = V.main()
    out = capsys.readouterr().out
    assert rc == 2, "an absent corpus must not share an exit code with drift"
    assert "NO CORPUS" in out
    assert "NOT a claim that any figure drifted" in out
    assert "DRIFT" not in out


def test_worlds_resolves_its_json_against_the_library_root(monkeypatch, capsys,
                                                          tmp_path):
    """It passed a relative path, which found nothing outside the repo."""
    from vetoworld.commands import worlds as W

    class _A:
        level = ["LAT", "W2"]

    monkeypatch.chdir(tmp_path)
    assert W.main(_A()) == 0
    assert "LOCK FAILED" not in capsys.readouterr().out


def test_the_necessity_control_is_asserted_ONLY_where_S_is_below_one(capsys):
    """**A designed property is not a defect.**

    Above S=1 legal-only survival is possible BY CONSTRUCTION — that is what the
    L1-L5 ladder and the upper Z bracket exist to vary. Asserting the necessity
    control on them reported six correctly-built worlds as failing, which is the
    kind of noise that teaches a reader to ignore the column.
    """
    from vetoworld.commands import worlds as W

    class _All:
        level = None

    assert W.main(_All()) == 0, "the full sweep must be clean"
    out = capsys.readouterr().out
    assert "necessity N/A by design" in out
    assert "NECESSITY FAILED" not in out
    # and it is still ENFORCED where it applies
    assert "LAT " in out and "necessity ok" in out


def test_every_FREE_verb_exits_zero_with_no_provider_key(monkeypatch):
    """A command that costs nothing must not need credentials — including the
    full-sweep form of `worlds`, which is what a stranger runs first."""
    from vetoworld.commands import doctor, seeds, verify, worlds
    for env in ("TOGETHER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(env, raising=False)

    class _A:
        level = None
        check = None
        model = None
        results = "results"

    assert doctor.main(_A()) == 0
    assert seeds.main(_A()) == 0
    assert worlds.main(_A()) == 0
    assert verify.main(_A()) == 0


def test_the_MANIFEST_ships_in_the_wheel_or_fetch_cannot_check_anything():
    """**The gap the first PyPI install exposed.**

    `pip install vetoworld && vworld corpus fetch` printed `manifest not on disk
    — UNCHECKED`: the manifest lived at the repo root and shipped in neither the
    wheel nor the dataset, so the digest check that is the whole point of the
    command silently did not run.

    It must ship with the CODE and not with the corpus. A manifest downloaded
    beside the cells would have the corpus vouching for itself.
    """
    import json

    from vetoworld.commands import corpus as CO

    assert CO.MANIFEST.exists(), CO.MANIFEST
    assert CO.MANIFEST.is_absolute(), "must not depend on the working directory"
    assert "vetoworld" in CO.MANIFEST.parts, "must live inside the package"
    d = json.loads(CO.MANIFEST.read_text())
    assert len(d["digest"]) == 64 and d["cells"] > 0


def test_the_manifest_is_declared_as_PACKAGE_DATA():
    """Being inside the package directory is not enough — setuptools ships only
    what `package-data` names, and a .json is not code."""
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    pkg = cfg["tool"]["setuptools"]["package-data"]["vetoworld"]
    assert "corpus.manifest.json" in pkg, pkg


def _project():
    import tomllib
    from pathlib import Path
    return tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]


def test_the_package_DECLARES_a_readme_or_PyPI_renders_nothing():
    """**0.1.0 and 0.1.1 both shipped with a blank PyPI page.** The README
    existed and was simply never referenced, so the metadata carried no
    description at all and nobody landing on the project page learned what it
    was."""
    from pathlib import Path
    d = _project()
    assert d.get("readme"), "no readme declared; the PyPI page will be empty"
    assert (Path(__file__).resolve().parents[1] / d["readme"]).exists()


def test_the_README_has_NO_RELATIVE_LINKS_because_PyPI_cannot_resolve_them():
    """The same file renders on GitHub and on PyPI, and PyPI has no repository
    to resolve a relative path against — every one of them 404s there."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / _project()["readme"]).read_text()
    rel = [m for m in re.findall(r"\]\(([^)]+)\)", src)
           if not m.startswith(("http", "#", "mailto:"))]
    assert not rel, f"relative links break on PyPI: {rel}"


def test_the_metadata_points_at_the_repo_and_the_dataset():
    """A benchmark whose page does not link its corpus is not replicable from
    the page."""
    urls = _project()["urls"]
    assert "huggingface.co/datasets" in urls["Dataset"]
    assert "github.com" in urls["Repository"]


def test_no_license_CLASSIFIER_beside_the_license_EXPRESSION():
    """setuptools rejects the two together under PEP 639 — the build fails
    outright, which is how this was found."""
    d = _project()
    assert d["license"] == "MIT"
    assert not [c for c in d.get("classifiers", []) if c.startswith("License ::")]
