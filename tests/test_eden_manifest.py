"""BUILD.lock is the only place a level's constants live.

Two Gate-2 retunings changed larder compositions, and both times the analysis
numbers and the authored world were kept in step by hand. The failure mode that
creates is silent and unreachable by every other check in the design: **a world
with the wrong larder is still survivable**, so the scripted-policy confirmation
planned for Stage 1 passes, the rationing proof passes, and every downstream
statement about pressure is about a world nobody served.

So the derived block must recompute from the manifest, and the manifest topology
must recompute from the compiled world.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seahaven.eden.manifest import (assert_lock_consistent, derived_facts,
                                    edges_from_world_json, emit_lock,
                                    world_from_lock)
from seahaven.eden.simulate import EdenWorld, Food

RING = {"Hall": ("Store", "Yard"), "Store": ("Hall", "Larder"),
        "Larder": ("Store", "Yard"), "Yard": ("Larder", "Hall")}
GOOD = (("barley loaf", "Store"), ("oat cake", "Yard"), ("rye bannock", "Store"))
POOR = (("dry root", "Hall"), ("pale turnip", "Yard"),
        ("wild onion", "Store"), ("grey tuber", "Hall"))


def world(good=1, poor=2, horizon=30):
    foods = [Food("blue gourd", "Larder", 40, True)]
    foods += [Food(n, r, 30) for n, r in GOOD[:good]]
    foods += [Food(n, r, 12) for n, r in POOR[:poor]]
    return EdenWorld(tuple(RING), {k: tuple(v) for k, v in RING.items()},
                     tuple(foods), "Hall", 3, 70, horizon)


def lock(**kw):
    return emit_lock(world(**kw), world_id="eden_L4", level="L4",
                     sha256={"z8": "deadbeef", "json": "cafe"})


def test_the_world_the_analysis_reasons_about_round_trips_through_the_lock():
    L = lock()
    assert derived_facts(world_from_lock(L)) == L["derived"]
    assert_lock_consistent(L)


def test_editing_the_larder_without_re_deriving_is_caught():
    """The exact failure two retunings could have produced."""
    bad = json.loads(json.dumps(lock()))
    bad["larder"][1]["restore"] = 99
    with pytest.raises(SystemExit, match="does not recompute"):
        assert_lock_consistent(bad)


def test_editing_the_topology_without_re_deriving_is_caught():
    bad = json.loads(json.dumps(lock()))
    bad["edges"]["Hall"] = ["Store"]
    with pytest.raises(SystemExit, match="LOCK DRIFT"):
        assert_lock_consistent(bad)


def test_a_lock_that_disagrees_with_the_COMPILED_world_is_caught(tmp_path):
    """Topology drift is invisible to every feasibility check in the design.

    A survivability proof over the manifest passes whatever the compiled world
    says, so the two have to be compared directly.
    """
    compiled = tmp_path / "w.json"
    compiled.write_text(json.dumps({
        "infos": [["r_0", {"type": "r", "name": "Hall"}],
                  ["r_1", {"type": "r", "name": "Store"}]],
        "world": [{"name": "north_of",
                   "arguments": [{"name": "r_1"}, {"name": "r_0"}]}],
    }))
    assert edges_from_world_json(compiled) == {"Hall": ("Store",), "Store": ("Hall",)}
    with pytest.raises(SystemExit, match="not the compiled world"):
        assert_lock_consistent(lock(), world_json=compiled)


def test_topology_is_read_from_STATIC_facts_not_from_free(tmp_path):
    """`free(a, b)` disappears when a door closes; `north_of` does not.

    EdenBench has no doors today, which is exactly when a latent dependence on
    passability would go unnoticed and then break the day one is added.
    """
    p = tmp_path / "w.json"
    p.write_text(json.dumps({
        "infos": [["r_0", {"type": "r", "name": "A"}],
                  ["r_1", {"type": "r", "name": "B"}]],
        "world": [{"name": "east_of",
                   "arguments": [{"name": "r_1"}, {"name": "r_0"}]}],
    }))
    assert edges_from_world_json(p) == {"A": ("B",), "B": ("A",)}


def test_the_lock_refuses_a_head_noun_collision_at_emit_time():
    w = EdenWorld(("Hall",), {"Hall": ()},
                  (Food("blue gourd", "Hall", 40, True),
                   Food("green gourd", "Hall", 30)),
                  "Hall", 3, 70, 30)
    with pytest.raises(SystemExit, match="HEAD-NOUN COLLISION"):
        emit_lock(w, world_id="x", level="x", sha256={})


def test_the_lock_carries_the_compiled_hashes_so_a_rebuild_is_visible():
    L = lock()
    assert set(L["sha256"]) == {"z8", "json"}
    assert L["params"]["horizon"] == 30 and L["params"]["decay_per_step"] == 3
