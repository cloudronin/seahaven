"""Build world_v0 — throwaway scaffolding for the A1b gate.

This is NOT the Seahaven world. It exists to produce one realistic 320-step
trajectory so that A1b can ask "does a LoRA pass on realistically-shaped,
agent-selected data move behavior?" before the real harness is built.

Design constraints, all in service of that one question:

  - Six rooms in a loop with one spur, so a wandering agent keeps finding new
    text for a few hundred steps without the affordances running dry.
  - No quest, so `max_score == 0` and the engine has no goal to leak. Verified
    to compile: TextWorld 1.7.0 accepts a questless build.
  - Stock logic only. No custom `.twl`. Anything needing a new verb belongs in
    the real world build, not here.
  - Enough takeable objects and containers that the action distribution is
    genuinely action-token dominated rather than navigation-only.

Compiling requires Rosetta 2 (Inform 7's `ni` is x86_64). Outputs are committed
so nothing downstream needs it.

Usage:
    python worlds/world_v0/build.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from textworld import GameMaker

WORLD_ID = "world_v2"

# (name, description) — flat, unatmospheric on purpose. Prose craft belongs to
# the real world; this one only has to be navigable and non-repetitive.
#
# EVERY DESCRIPTION MUST NAME ITS EXITS, and this is not a stylistic choice.
# GameMaker does not append exit text, so the first version of this file produced
# rooms like "A cramped galley. Salt has got into everything." The agent had no
# way to learn that north and south led anywhere: `admissible_commands` knows,
# but that is hidden from the agent by design, and correctly so. The result was a
# trajectory that never left the starting room — 40 steps, 2 unique commands.
#
# The exits are described in prose rather than exposed as a command list, which
# is how a text adventure has always done it: the room tells you where the doors
# are, and reading it is the agent's job. `test_world_artifacts.py` asserts the
# invariant so a future room cannot quietly omit them.
ROOMS = {
    "nursery": (
        "Nursery",
        "Trays of dry compost under cracked glass. A door leads north to the "
        "furnace, and the colonnade runs east.",
    ),
    "furnace": (
        "Furnace",
        "Cold iron, long dead. The nursery is south, and a low arch opens east "
        "into the fernery.",
    ),
    "fernery": (
        "Fernery",
        "Fronds gone brown, the air still damp. The furnace lies west, and the "
        "vault is east.",
    ),
    "colonnade": (
        "Colonnade",
        "Pillars under a broken roof. The nursery is west, the cloche lies "
        "east, and the vault is north.",
    ),
    "vault": (
        "Vault",
        "Drawers in rows, most pulled open and empty. The fernery is west, and "
        "the colonnade runs south.",
    ),
    "cloche": (
        "Cloche",
        "A sunken bed under a hinged pane. The colonnade is west, and a stair "
        "goes down to the north.",
    ),
    "sump": (
        "Sump",
        "Brickwork, half full and very quiet. The only way out is south.",
    ),
}

# GameMaker exposes only the four compass directions — there is no up/down, so
# vertical connections have to be expressed horizontally. Laid out on a grid so
# the map is mentally consistent for anything navigating it:
#
#   boiler_house --- fern_house --- vault
#        |                              |
#   potting_shed --- gravel_path --------
#                         |
#                    orchid_case --- sump
#
# Seven rooms against world_v0's six, a different topology, and a different
# object inventory. Mechanics, action vocabulary and prompts are identical --
# only the world changes, which is what a trait-stability test requires.
EDGES = [
    ("nursery", "north", "furnace", "south"),
    ("nursery", "east", "colonnade", "west"),
    ("furnace", "east", "fernery", "west"),
    ("fernery", "east", "vault", "west"),
    ("colonnade", "north", "vault", "south"),
    ("colonnade", "east", "cloche", "west"),
    ("cloche", "north", "sump", "south"),
]

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

# (type, name, description, room). Types: o portable, c container, s supporter.
OBJECTS = [
    ("o", "trowel", "Worn bright along one edge.", "nursery"),
    ("s", "workbench", "Scarred, and dusted with dry soil.", "nursery"),
    ("o", "twine", "A ball of it, furred with age.", "nursery"),
    ("o", "lantern", "Glass sooted almost opaque.", "furnace"),
    ("c", "chest", "Wooden, with many drawers.", "vault"),
    ("o", "ledger", "Ruled columns, entries trailing off.", "vault"),
    ("o", "shears", "Stiff, and spotted with rust.", "fernery"),
    ("s", "slab", "Cool to the touch, and very flat.", "fernery"),
    ("o", "basket", "Split along one side.", "colonnade"),
    ("c", "hamper", "Wicker, with a sagging lid.", "cloche"),
    ("o", "dipper", "Tin, with a long handle.", "sump"),
]


def validate_topology() -> None:
    """Check the edge table before handing it to GameMaker.

    GameMaker raises ExitAlreadyUsedError from deep inside `connect()`, naming
    the rooms by `<object at 0x...>`, which is useless for finding the offending
    row. Checking here fails with the room and direction that actually clash.
    """
    used: dict[tuple[str, str], str] = {}
    for a, a_dir, b, b_dir in EDGES:
        for room, direction, other in ((a, a_dir, b), (b, b_dir, a)):
            if room not in ROOMS:
                raise ValueError(f"unknown room {room!r} in edge table")
            if direction not in OPPOSITE:
                raise ValueError(
                    f"{room}.{direction}: GameMaker supports only "
                    f"{sorted(OPPOSITE)} — there is no up/down"
                )
            key = (room, direction)
            if key in used:
                raise ValueError(
                    f"{room}.{direction} is claimed twice: "
                    f"by {used[key]} and by {other}"
                )
            used[key] = other

        if OPPOSITE[a_dir] != b_dir:
            raise ValueError(
                f"{a}.{a_dir} <-> {b}.{b_dir} is geometrically inconsistent; "
                f"expected {b}.{OPPOSITE[a_dir]}"
            )

    reachable, frontier = {"nursery"}, ["nursery"]
    while frontier:
        current = frontier.pop()
        for a, _, b, _ in EDGES:
            for src, dst in ((a, b), (b, a)):
                if src == current and dst not in reachable:
                    reachable.add(dst)
                    frontier.append(dst)
    if reachable != set(ROOMS):
        raise ValueError(f"unreachable from potting_shed: {sorted(set(ROOMS) - reachable)}")


#: Words common enough in first-person narration that an entity sharing one
#: cannot be detected by name. `store` is a verb, `can` a modal, `frame` and
#: `walk` verbs. world_v0 shipped a room called "Store" and got away with it
#: only because rooms are usually named in a locational context — luck, not
#: design.
NARRATIVE_STOPLIST = {
    "store", "well", "can", "walk", "frame", "light", "lamp", "still", "left",
    "right", "back", "long", "cold", "seed", "rain", "spring", "fall", "run",
    "set", "place", "room", "house", "open", "close", "look", "take", "drop",
    "path", "case", "shed", "bed", "pot", "point", "second", "part", "order", "match",
}


def validate_entity_names() -> None:
    """Names must be findable in prose. Runs BEFORE compiling, deliberately:
    discovering a collision on the finished world costs a rebuild.

    Two failures, both silent at runtime:

    1. **Substring collision.** The mention detector is containment, so if one
       name contains another, a mention of the longer is a mention of the
       shorter. "Lamp Room" and "lamp" would be indistinguishable.
    2. **Stopword collision.** A name that is also a common narrative word gets
       matched by ordinary prose about anything.
    """
    names = [n for n, _ in ROOMS.values()] + [o[1] for o in OBJECTS]
    low = [n.lower() for n in names]

    for i, a in enumerate(low):
        for j, b in enumerate(low):
            if i != j and a in b:
                raise ValueError(
                    f"entity name {names[i]!r} is a substring of {names[j]!r}; "
                    "a mention of one cannot be told from the other")

    for name, n in zip(names, low):
        for token in n.split():
            if token in NARRATIVE_STOPLIST:
                raise ValueError(
                    f"entity name {name!r} contains {token!r}, which is common "
                    "in first-person narration and matches ordinary prose")


def build(out_dir: Path) -> dict:
    validate_topology()
    validate_entity_names()
    out_dir.mkdir(parents=True, exist_ok=True)
    z8 = out_dir / f"{WORLD_ID}.z8"

    M = GameMaker()
    rooms = {key: M.new_room(name, desc) for key, (name, desc) in ROOMS.items()}

    for a, a_dir, b, b_dir in EDGES:
        M.connect(getattr(rooms[a], a_dir), getattr(rooms[b], b_dir))

    for kind, name, desc, room in OBJECTS:
        entity = M.new(type=kind, name=name, desc=desc)
        rooms[room].add(entity)

    M.set_player(rooms["nursery"])

    # No quest. Deliberate: a quest gives the engine a score, and score is the
    # one signal the design forbids reaching the agent.
    M.build(validate=True)
    M.compile(str(z8))

    sidecar = z8.with_suffix(".json")
    if not sidecar.exists():
        raise RuntimeError(
            f"sidecar {sidecar.name} was not written — facts/entities will be "
            "empty at runtime because JerichoEnv does not populate them"
        )

    lock = {
        "world_id": WORLD_ID,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rooms": len(ROOMS),
        "objects": len(OBJECTS),
        "quest": None,
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.iterdir())
            if p.suffix in {".z8", ".json"} and p.name != "BUILD.lock.json"
        },
        "tools": _tool_versions(),
    }
    (out_dir / "BUILD.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def _tool_versions() -> dict[str, str]:
    import jericho
    import textworld

    versions = {
        "textworld": getattr(textworld, "__version__", "unknown"),
        "jericho": getattr(jericho, "__version__", "unknown"),
        "python": sys.version.split()[0],
    }
    try:
        ni = (
            Path(textworld.__file__).parent
            / "thirdparty/inform7-6M62/share/inform7/Compilers/ni"
        )
        out = subprocess.run(["file", str(ni)], capture_output=True, text=True).stdout
        versions["inform7_ni_arch"] = "arm64" if "arm64" in out else "x86_64"
    except Exception:  # pragma: no cover - diagnostic only
        pass
    return versions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    lock = build(args.out)
    print(f"built {WORLD_ID}: {lock['rooms']} rooms, {lock['objects']} objects, no quest")
    for name, digest in lock["artifacts"].items():
        print(f"  {name:24s} {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
