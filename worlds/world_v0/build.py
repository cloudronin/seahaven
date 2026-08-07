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

WORLD_ID = "world_v0"

# (name, description) — flat, unatmospheric on purpose. Prose craft belongs to
# the real world; this one only has to be navigable and non-repetitive.
ROOMS = {
    "galley": ("Galley", "A cramped galley. Salt has got into everything."),
    "store": ("Store", "Shelves, mostly empty. It smells of paraffin."),
    "landing": ("Landing", "A half-turn of stair with a window that does not open."),
    "lamp_room": ("Lamp Room", "Glass on every side. The mechanism is still."),
    "workshop": ("Workshop", "A bench under a rack of tools, most of them gone."),
    "cistern": ("Cistern", "Low and cold. Water somewhere, not visible."),
}

# GameMaker exposes only the four compass directions — there is no up/down, so
# vertical connections have to be expressed horizontally. Laid out on a grid so
# the map is mentally consistent for anything navigating it:
#
#       workshop(0,2) --- lamp_room(1,2)
#            |                  |
#       store(0,1)   ---   landing(1,1)
#            |
#       galley(0,0)
#            |
#       cistern(0,-1)
#
# The four-room loop lets an agent wander indefinitely; galley and cistern are
# spurs, so there is also somewhere to deliberately go and come back from.
EDGES = [
    ("galley", "north", "store", "south"),
    ("galley", "south", "cistern", "north"),
    ("store", "north", "workshop", "south"),
    ("store", "east", "landing", "west"),
    ("landing", "north", "lamp_room", "south"),
    ("lamp_room", "west", "workshop", "east"),
]

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

# (type, name, description, room). Types: o portable, c container, s supporter.
OBJECTS = [
    ("o", "kettle", "Tin, dented on one side.", "galley"),
    ("s", "scrubbed table", "Scoured pale by years of it.", "galley"),
    ("c", "crate", "A wooden crate.", "store"),
    ("o", "coil of rope", "Stiff with salt.", "store"),
    ("o", "brass key", "Small, and green where it is handled.", "landing"),
    ("o", "logbook", "Ruled columns, most of them empty.", "lamp_room"),
    ("s", "bench", "Scarred, and burnt at one end.", "workshop"),
    ("o", "oil can", "Half full, by the weight of it.", "workshop"),
    ("c", "locker", "Steel, and it sticks.", "workshop"),
    ("o", "tin cup", "Enamel, chipped to the metal.", "cistern"),
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

    reachable, frontier = {"galley"}, ["galley"]
    while frontier:
        current = frontier.pop()
        for a, _, b, _ in EDGES:
            for src, dst in ((a, b), (b, a)):
                if src == current and dst not in reachable:
                    reachable.add(dst)
                    frontier.append(dst)
    if reachable != set(ROOMS):
        raise ValueError(f"unreachable from galley: {sorted(set(ROOMS) - reachable)}")


def build(out_dir: Path) -> dict:
    validate_topology()
    out_dir.mkdir(parents=True, exist_ok=True)
    z8 = out_dir / f"{WORLD_ID}.z8"

    M = GameMaker()
    rooms = {key: M.new_room(name, desc) for key, (name, desc) in ROOMS.items()}

    for a, a_dir, b, b_dir in EDGES:
        M.connect(getattr(rooms[a], a_dir), getattr(rooms[b], b_dir))

    for kind, name, desc, room in OBJECTS:
        entity = M.new(type=kind, name=name, desc=desc)
        rooms[room].add(entity)

    M.set_player(rooms["galley"])

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
