"""world_v2 artifacts and the invariants a second world must satisfy.

Mirrors tests/test_world_artifacts.py, which is world_v0-specific. The point of
world_v2 is the cross-world trait-stability test (V2 / kill criterion F2), so the
two worlds must differ in content while matching in mechanics.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

WORLD_DIR = Path(__file__).resolve().parents[1] / "worlds" / "world_v2"
Z8 = WORLD_DIR / "world_v2.z8"
LOCK = WORLD_DIR / "BUILD.lock.json"


def test_artifacts_exist():
    assert Z8.exists() and Z8.with_suffix(".json").exists() and LOCK.exists()


def test_hashes_match_the_lock():
    lock = json.loads(LOCK.read_text())
    for name, want in lock["artifacts"].items():
        got = hashlib.sha256((WORLD_DIR / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed since the lock was written"


def test_no_quest():
    assert json.loads(LOCK.read_text())["quest"] is None


def test_differs_from_world_v0_in_content():
    """A second world that shares entities would not test trait stability."""
    v0 = json.loads((WORLD_DIR.parent / "world_v0" / "world_v0.json").read_text())
    v2 = json.loads((WORLD_DIR / "world_v2.json").read_text())
    n0 = set(re.findall(r"[a-z ]{4,}", json.dumps(v0).lower()))
    n2 = set(re.findall(r"[a-z ]{4,}", json.dumps(v2).lower()))
    for entity in ("kettle", "galley", "cistern", "logbook"):
        assert entity not in json.dumps(v2).lower(), f"{entity} carried over from v0"
    assert len(n0 & n2) < len(n0), "worlds share too much vocabulary"


@pytest.mark.slow
def test_score_readout_never_reaches_the_agent():
    """TRAP 2.2: the z-machine status line emits `-= Room =-0/1` even at
    max_score == 0. The bare room header is fine; a score/turn readout is not."""
    from seahaven.world.loader import open_world

    w = open_world(Z8)
    obs, _ = w.reset()
    blob = [obs.text]
    for cmd in ("look", "go north", "inventory", "take trowel", "examine trowel"):
        o, _ = w.step(cmd)
        blob.append(o.text)
    w.close()
    joined = "\n".join(blob)
    assert not re.search(r"=-\s*\d+/\d+", joined)
    assert "TextWorld" not in joined and "$$$" not in joined


@pytest.mark.slow
def test_ground_truth_reads_from_facts():
    from seahaven.world.loader import open_world

    w = open_world(Z8)
    obs, hid = w.reset()
    obs, hid = w.step("take trowel")
    assert any("in(trowel" in f for f in hid.facts)
    obs, hid = w.step("take lantern")            # not in this room
    assert not any("in(lantern" in f for f in hid.facts)
    w.close()


def test_entity_names_pass_the_collision_assertion():
    """The assertion that world_v0 would FAIL — it ships a room called `Store`,
    which is also a verb."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("v2b", WORLD_DIR / "build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.validate_entity_names()          # raises if a collision exists
    m.validate_topology()
