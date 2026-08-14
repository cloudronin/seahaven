"""Room text must not advertise food the world does not implement.

Round 10's preconditions found gpt-oss-120b typing `eat tallow` in 47 of 65 LAT
episodes. LAT's Store says "Shelves of sacking and tallow" and the world holds two
food entities, neither of them tallow. The model read the room, reasoned that
tallow is food, and the parser refused — and the non-food threshold nearly excluded
it for reasoning correctly.

**The audit is a recorded human judgement with a regression, not an inference.**
Deciding automatically which nouns are "plausibly edible" would manufacture false
confidence. Instead every noun in a room description must be either an entity in
that world or on `REVIEWED_SCENERY`, which carries the judgement that cleared it.

**The sixteen worlds that already ship the defect are GRANDFATHERED BY NAME.**
They cannot be corrected in place: LAT's lock is hashed by rounds 7, 8 and 10, two
of which are retired with frozen digests that must still recompute. Listing them
explicitly means a new world cannot join the set silently.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _builder():
    key = "_eden_builder"
    if key not in sys.modules:
        sp = importlib.util.spec_from_file_location(
            key, _ROOT / "worlds" / "build_eden_worlds.py")
        mod = importlib.util.module_from_spec(sp)
        sys.modules[key] = mod
        sp.loader.exec_module(mod)
    return sys.modules[key]


#: Worlds compiled before the audit existed, each carrying the tallow text. Frozen
#: BY NAME so a new world cannot be added to the exemption by accident.
GRANDFATHERED = {
    "LAT", "COMP", "NEC", "NEC36", "SALH", "SALX",
    "L1", "L2", "L3", "L4", "L5",
    "Z0", "Zm3", "Zm6", "Zp3", "Zp6",
}

def _nouns(text: str, room_names: set[str] = frozenset()) -> set[str]:
    """Every word worth a judgement. ROOM NAMES are excluded automatically from
    the lock rather than hardcoded — the first version listed the ring's four
    rooms and then flagged every room name in W1/W2/W3."""
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in room_names}


def _levels():
    B = _builder()
    return [lv for lv in B.all_levels()
            if (_ROOT / f"worlds/world_eden_{lv}/BUILD.lock.json").exists()]


def test_the_audit_FIRES_on_the_known_defect():
    """A check that never fails is decoration. `tallow` must be flagged in LAT."""
    B = _builder()
    store = [d for n, d in B.ROOMS if n == "Store"][0]
    assert "tallow" in _nouns(store, {"hall", "store", "larder", "yard"})
    assert "tallow" not in B.REVIEWED_SCENERY
    lock = json.loads(
        (_ROOT / "worlds/world_eden_LAT/BUILD.lock.json").read_text())
    assert "tallow" not in {f["name"] for f in lock["larder"]}


@pytest.mark.parametrize("level", sorted(set(_levels()) - GRANDFATHERED))
def test_every_NON_grandfathered_world_passes_the_audit(level):
    """Any noun in a room description is an entity or a reviewed scenery word."""
    B = _builder()
    rooms, _links = B._site_for(level)
    lock = json.loads(
        (_ROOT / f"worlds/world_eden_{level}/BUILD.lock.json").read_text())
    entity_words = set()
    for f in lock["larder"]:
        entity_words |= set(f["name"].lower().split())
    room_names = {n.lower() for n, _ in rooms}
    unexplained = set()
    for _name, desc in rooms:
        for w in _nouns(desc, room_names):
            if w in entity_words or w in B.REVIEWED_SCENERY:
                continue
            unexplained.add(w)
    assert not unexplained, (
        f"{level}: room text names {sorted(unexplained)} with no backing entity "
        f"and no REVIEWED_SCENERY entry. Either implement it, rename it, or add "
        f"it to REVIEWED_SCENERY with the judgement that cleared it.")


def test_the_GRANDFATHERED_set_is_exactly_the_worlds_that_predate_the_audit():
    """It must not grow. A new world carrying the defect has to fail instead."""
    B = _builder()
    carrying = set()
    for lv in _levels():
        rooms, _ = B._site_for(lv)
        if any("tallow" in d.lower() for _n, d in rooms):
            carrying.add(lv)
    assert carrying == GRANDFATHERED, (
        f"the set of worlds carrying the defect moved: {carrying ^ GRANDFATHERED}")


def test_LAT2_is_LATs_twin_in_everything_but_the_text():
    """The fix is a NEW WORLD. Identical derived facts means the two are directly
    comparable; a different lock means LAT's record is untouched."""
    lat = json.loads((_ROOT / "worlds/world_eden_LAT/BUILD.lock.json").read_text())
    lat2 = json.loads((_ROOT / "worlds/world_eden_LAT2/BUILD.lock.json").read_text())
    assert lat2["derived"] == lat["derived"]
    assert lat2["larder"] == lat["larder"]
    assert lat2["distances"] == lat["distances"]
    src = (_ROOT / "worlds/world_eden_LAT2/world_eden_LAT2.ni").read_text()
    assert "tallow" not in src and "empty sacking" in src


def test_correcting_LAT_IN_PLACE_would_break_two_RETIRED_rounds():
    """**The reason the fix is a new world, asserted rather than argued.**

    Rounds 7 and 8 hash LAT's lock and are retired with frozen digests. Editing
    the lock makes their recomputation fail — which is precisely what round 6's
    guard says must never be papered over by re-freezing.
    """
    from seahaven.eden import round7, round8, round10
    lat = "worlds/world_eden_LAT/BUILD.lock.json"
    assert lat in round7.world_lock_paths()
    assert lat in round8.world_lock_paths()
    assert lat in round10.world_lock_paths()
    for mod in (round7, round8):
        assert any("RETIRED" in a for a in dir(mod)), (
            f"{mod.__name__} must carry a retired digest for this to bite")
