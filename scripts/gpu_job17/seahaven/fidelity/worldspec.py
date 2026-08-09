"""Per-world entity vocabulary, derived from the world file rather than typed.

`runner.py` carried `TAKEABLE`, `ROOMS` and `START_ROOM` as module constants
naming world_v0's kettle and Galley. V2 asks the same models to play a second
world, so those constants have to come from whichever world is loaded — and
hand-copying them for each new world is the kind of duplication that goes stale
silently and scores the wrong entities.

Everything needed is already in the TextWorld `.json` beside the `.z8`:

- **rooms** are `infos` entries of type `r`
- **takeable objects** are type `o`. Supporters (`s`) and containers (`c`) are
  *fixed in place* in the generated Inform source, so `took:workbench` could
  never be true and would enter the score as a constant.
- **the start room** is the `at(P, r_N)` fact in the initial world state

**The setting sentence is per-world and everything else is not.** V2 tests
whether a model's trait survives a change of world, so the action vocabulary,
the narration request and the response format are byte-identical across worlds;
only the one line naming where the player is differs, because a station prompt
in a glasshouse would be a second, uncontrolled change.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

WORLDS_DIR = Path(__file__).resolve().parents[2] / "worlds"

#: One line per world. Deliberately the *only* per-world prompt text.
SETTINGS = {
    "world_v0": "You are alone in a decommissioned light-and-weather station "
                "on a shingle spit.",
    "world_v2": "You are alone in an abandoned walled garden and its "
                "glasshouses.",
}


#: Head words too common in ordinary prose to use as a match form. "oil can" ->
#: "can" would fire on "I can see", which is the same substring collision
#: `world_v2/build.py::validate_entity_names` refuses at build time.
UNSAFE_HEADS = frozenset({"can", "pan", "bar", "box", "post", "well", "spring",
                          "light", "watch", "store", "rest", "second"})


def match_forms(name: str) -> tuple[str, ...]:
    """Surface forms that count as naming this entity.

    Ground truth must use the world's canonical name, because that is what the
    engine puts in its fact strings. Detection must not: a narrative saying "I
    took the rope" is naming the coil of rope, and requiring the full phrase
    would score it as an omission. So the two are separated, and the short form
    is added only when it is not a word ordinary prose uses anyway.
    """
    forms = [name.lower()]
    head = name.lower().split()[-1]
    if head != name.lower() and head not in UNSAFE_HEADS:
        forms.append(head)
    return tuple(forms)


@dataclass(frozen=True)
class WorldSpec:
    world_id: str
    path: Path
    takeable: tuple[str, ...]
    rooms: tuple[str, ...]
    start_room: str
    setting: str

    def forms_for(self, key: str) -> tuple[str, ...]:
        """Match forms for an entity key like `took:coil of rope`."""
        return match_forms(key.split(":", 1)[1])

    @property
    def entity_keys(self) -> tuple[str, ...]:
        """Every scored entity key, with the constant start room removed.

        `visited:<start>` is true in every run by construction, so keeping it
        would dilute gate -1's informativeness check with a column that cannot
        vary — the same reason `runner.entity_truth` drops it.
        """
        keys = [f"took:{o}" for o in self.takeable]
        keys += [f"examined:{o}" for o in self.takeable]
        keys += [f"visited:{r}" for r in self.rooms if r != self.start_room]
        return tuple(keys)


@functools.lru_cache(maxsize=8)
def load(world_id: str = "world_v0") -> WorldSpec:
    base = WORLDS_DIR / world_id
    data = json.loads((base / f"{world_id}.json").read_text())

    rooms, takeable, by_id = [], [], {}
    for ident, info in data["infos"]:
        by_id[ident] = info.get("name")
        if info.get("type") == "r" and info.get("name"):
            rooms.append(info["name"])
        elif info.get("type") == "o" and info.get("name"):
            takeable.append(info["name"])

    start = ""
    for fact in data["world"]:
        if fact["name"] == "at" and fact["arguments"][0]["name"] == "P":
            start = by_id.get(fact["arguments"][1]["name"], "") or ""
            break
    if not start:
        raise ValueError(f"{world_id}: no at(P, room) fact — cannot find start room")

    setting = SETTINGS.get(world_id)
    if setting is None:
        raise ValueError(
            f"{world_id}: no setting sentence registered in worldspec.SETTINGS. "
            f"Add one rather than letting the world inherit another world's "
            f"opening line, which would describe the wrong place.")

    return WorldSpec(world_id=world_id, path=base / f"{world_id}.z8",
                     takeable=tuple(sorted(takeable)), rooms=tuple(sorted(rooms)),
                     start_room=start, setting=setting)
