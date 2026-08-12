"""The round-2 grid, its pin, and the corpus boundary the read must not cross.

Two failure modes here are silent and expensive, and neither is caught by
anything else in the suite.

**A moved pin.** A constant edited after the numbers arrive recomputes perfectly
from the edited constant, so every derived check still passes. The digest is the
only thing that notices.

**A crossed corpus boundary.** The 36 round-1 cells are named
`eden_e0_<level>.json` and match any reasonable glob over `results/`. Pooling
them with round 2 would merge a self-served 3-10B cohort at a 16-token cap with a
hosted frontier cohort at 2048, across different worlds — and the merged table
would look entirely normal. Axis 2b was bitten by this exact class of bug
(attribution by filename index after a model was dropped), and the fix is the
same: match on what the artifact says it is.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from seahaven.eden import round2 as R

_ROOT = Path(__file__).resolve().parents[1]


def test_the_pin_verifies_against_the_committed_constants():
    R.assert_pinned()
    assert R.current_hash() == R.PINNED_ROUND2_HASH


def test_the_pin_notices_a_moved_constant():
    """The whole point: derived checks cannot see this, the digest can."""
    old = R.EPISODES_PER_CELL
    R.EPISODES_PER_CELL = old + 1
    try:
        with pytest.raises(SystemExit, match="PIN BROKEN"):
            R.assert_pinned()
    finally:
        R.EPISODES_PER_CELL = old
    R.assert_pinned()


def test_the_pin_notices_a_moved_ARTIFACT_not_just_a_constant():
    """Hashing only constants leaves the reasoning free to drift."""
    assert set(R.ARTIFACTS) >= {"seahaven/eden/outcome.py",
                                "worlds/build_eden_worlds.py"}
    for a in R.ARTIFACTS:
        assert (_ROOT / a).exists(), f"hashed artifact missing: {a}"
    body = json.loads(R.payload())
    assert len(body["artifacts"]) == len(R.ARTIFACTS)


def test_the_grid_is_what_the_budget_was_approved_for():
    full = R.cells()
    assert len(full) == 84, "6 models x 2 arms x 6 levels, plus 6 x 2 SAL"
    assert len(full) * R.EPISODES_PER_CELL == 2016
    # SAL is the prohibition arm only: it measures salience, not the rule.
    sal = [c for c in full if c[2] in R.SALIENCE]
    assert len(sal) == 12 and {c[1] for c in sal} == {"A1"}
    # every ladder level appears in BOTH arms for every model, or the headline
    # has a hole in it
    for m in R.COHORT:
        for lv in R.LADDER:
            assert (m, "A1", lv) in full and (m, "A0", lv) in full


def test_gate0_is_a_strict_subset_run_first():
    g = R.cells(gate0=True)
    assert len(g) == 12 and {c[2] for c in g} == {R.GATE0_LEVEL}
    assert set(g) <= set(R.cells()), \
        "Gate 0 cells must be reusable by the full run, not re-billed"


def test_one_seed0_for_both_arms_is_what_makes_the_contrast_paired():
    assert isinstance(R.SEED0, int)
    a1 = [c for c in R.cells() if c[1] == "A1"]
    a0 = [c for c in R.cells() if c[1] == "A0"]
    assert {(m, lv) for m, _, lv in a0} <= {(m, lv) for m, _, lv in a1}


def test_the_read_EXCLUDES_round_one_even_though_the_filenames_match(tmp_path):
    """The corpus boundary, exercised against real round-1 artifacts on disk."""
    results = _ROOT / "results"
    r1 = sorted(results.glob("eden_e0_*.json"))
    assert r1, "round-1 corpus missing; this test needs it to mean anything"
    for f in r1:
        d = json.loads(f.read_text())
        assert not d.get("meta", {}).get("round2_pin"), \
            f"{f.name} carries a round-2 pin; the boundary is not a boundary"

    out = subprocess.run([sys.executable, "scripts/eden_read.py"],
                         cwd=_ROOT, capture_output=True, text=True)
    if "nothing to read" in out.stdout:
        pytest.skip("no round-2 cells on disk yet")
    # Round 1's cohort must not appear anywhere in a round-2 read.
    for name in ("Qwen2.5-3B", "gemma-2-9b-it", "Mistral-7B", "Falcon3-10B",
                 "Llama-3.1-8B"):
        assert name not in out.stdout, \
            f"round-1 model {name} leaked into the round-2 read"


def test_every_round2_cell_on_disk_carries_the_CURRENT_pin():
    """A cell produced under an older pin is a different experiment.

    Cheap to check and impossible to notice by eye: the file is well-formed, the
    numbers recompute, and only the digest says it came from a different world.
    """
    # Selected on RECORDED IDENTITY, not on the glob -- which is the whole
    # lesson of this file. `eden_e2_` is already round 1's prefix for E-axis
    # level 2, so `eden_e2_*.json` matches six round-1 cells that have nothing
    # to do with round 2. Globbing for them here would have been the same
    # mistake one function up.
    pinned = []
    for p in (_ROOT / "results").glob("eden_*.json"):
        try:
            pin = json.loads(p.read_text()).get("meta", {}).get("round2_pin")
        except Exception:
            continue
        if pin:
            pinned.append((p.name, pin))
    if not pinned:
        pytest.skip("no round-2 cells on disk yet")
    stale = [(n, pin[:12]) for n, pin in pinned
             if pin != R.PINNED_ROUND2_HASH]
    assert not stale, f"cells from a different freeze: {stale}"


def test_round1_already_owns_the_eden_e2_prefix():
    """A near-miss worth a regression: `eden_e2_<level>.json` is round 1's
    E-axis level 2, in the same directory. Round-2 cell paths must never
    collide, because the sweep's resume check keys on path existence and would
    skip a cell it had never run."""
    from scripts.eden_round2_sweep import cell_path
    r1 = {p.name for p in (_ROOT / "results").glob("eden_e2_*.json")
          if "__" not in p.name}
    assert r1, "round-1 E2 cells missing; this regression needs them"
    r2 = {cell_path(*c).name for c in R.cells()}
    assert not (r1 & r2), f"round-2 cell path collides with round 1: {r1 & r2}"


# --------------------------------------------------------------------------
# The round-3 top-up's licence: the pin moved, the generative path did not.
# --------------------------------------------------------------------------

def test_the_LAT_GENERATIVE_PATH_is_unchanged_since_the_original_cells():
    """**A pin that moved is not the same as a world that moved.**

    Round 3's LAT cells were played at `d50dc45`. Since then `simulate.py`,
    `outcome.py` and `build_eden_worlds.py` have all changed, so round 3's pin
    no longer matches what those cells recorded — and the top-up rule says stop
    if a hashed file moved.

    It moved. But `simulate.py` changed the arithmetic reported ABOUT a world,
    not the world served: the runner reads `params` and `larder` from the lock
    during a rollout and never calls `best_trajectory`. This asserts the things
    a model actually experiences are byte-identical, which is what makes the new
    72 episodes poolable with the committed 24.

    If this ever fails, the top-up is genuinely not poolable and the rule bites.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    REF = "d50dc45"          # the commit the LAT cells were played at

    old = json.loads(subprocess.run(
        ["git", "show", f"{REF}:worlds/world_eden_LAT/BUILD.lock.json"],
        cwd=root, capture_output=True, text=True).stdout)
    new = json.loads((root / "worlds/world_eden_LAT/BUILD.lock.json").read_text())
    assert old["params"] == new["params"], "the served world's parameters moved"
    assert old["larder"] == new["larder"], "the larder moved"
    assert old["sha256"] == new["sha256"], "the compiled .z8 moved"
    assert old["derived"] != new["derived"], (
        "derived is expected to have changed — if it has not, the optimum fix "
        "did not land and this test is checking nothing")

    # Every constant that reaches the served prompt.
    src = subprocess.run(["git", "show", f"{REF}:seahaven/eden/outcome.py"],
                         cwd=root, capture_output=True, text=True).stdout
    cur = (root / "seahaven/eden/outcome.py").read_text()
    import re
    for name in ("PROHIBITION", "EDEN_GOAL", "EDEN_VOCAB", "EDEN_RESTRICTION",
                 "EDEN_MAX_TOKENS"):
        a = re.search(rf"^{name} = (.+)$", src, re.M)
        b = re.search(rf"^{name} = (.+)$", cur, re.M)
        assert a and b and a.group(1) == b.group(1), \
            f"{name} changed; the served prompt is not what round 3 served"


def test_the_LAT_corpus_already_pools_across_pins_and_that_is_recorded():
    """gemma and Llama's n=96 — quoted throughout rounds 3 and 4 — pools two
    pins. No hashed artifact moved between them, only `round3.py`'s own
    constants, so it was substantively safe. It was never stated, and this is
    where it is stated."""
    import glob

    root = Path(__file__).resolve().parents[1]
    pins = set()
    for f in glob.glob(str(root / "results" / "eden_e3_*LAT*.json")):
        pins.add(json.loads(Path(f).read_text())["meta"]["round3_pin"])
    assert len(pins) >= 2, (
        "expected the LAT corpus to span more than one pin; if it no longer "
        "does, this note is stale and should be removed rather than kept")
