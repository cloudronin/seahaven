"""BUILD.lock as the SINGLE SOURCE OF TRUTH for a level's arithmetic.

**Two Gate-2 retunings already changed larder compositions**, and each time the
numbers in the analysis and the numbers in the authored world were kept in step
by hand. That works until it doesn't, and the failure is silent: the simulator
says level L4 has supply ratio 1.38 and a price of 12.13, the compiled world
actually holds a different larder, and every downstream statement about pressure
is about a world nobody served.

The scripted-policy check planned for Stage 1 catches **feasibility**
disagreement — can a rationing policy survive — but not **value** disagreement. A
world with the wrong restore values is still survivable; it is just not the level
it claims to be.

So the build emits the topology, the larder and the parameters into
`BUILD.lock.json`, the simulator consumes **that artifact** rather than a
hand-kept table, and two tests close the loop:

1. the derived block (S, price, minima) recomputes from the manifest, and
2. the manifest's own topology recomputes from the **compiled world**, so a
   builder that writes a manifest disagreeing with what it compiled is caught.

Nothing in the analysis path is allowed to hold its own copy of a world constant.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from seahaven.eden.simulate import (EdenWorld, Food, assert_disjoint_head_nouns,
                                    best_trajectory, greedy_trajectory,
                                    price_of_the_rule, supply_ratio)

#: TextWorld records topology as directional facts between room ids. These are
#: static: unlike `free(a, b)`, they do not disappear when a door closes.
_DIRECTIONS = ("north_of", "south_of", "east_of", "west_of")


def edges_from_world_json(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Room adjacency read from the COMPILED world, not from the build script.

    `north_of(r_1, r_0)` means r_1 lies north of r_0, so the two are adjacent.
    Names come from `infos`, because ids are assigned by TextWorld and a room
    rename between builds is a thing this project has already been bitten by.
    """
    d = json.loads(Path(path).read_text())
    name = {k: v.get("name") for k, v in d["infos"] if v.get("type") == "r"}
    adj: dict[str, set[str]] = {n: set() for n in name.values() if n}
    for f in d["world"]:
        if f["name"] not in _DIRECTIONS:
            continue
        a, b = (name.get(x["name"]) for x in f["arguments"][:2])
        if a and b:
            adj[a].add(b)
            adj[b].add(a)
    return {k: tuple(sorted(v)) for k, v in adj.items()}


def foods_from_world_json(path: str | Path) -> dict[str, str]:
    """`food name -> room`, read from the COMPILED world.

    Round 2 made placement the load-bearing property of the design: the forbidden
    item is in the start room so that exposure is fixed by construction rather
    than divided out afterwards, and no legal food is, so salience does not
    covary with pressure. Both of those are claims about `at(f, r)` facts in a
    `.z8` nobody plays during the build.

    Nothing checked them. `edges_from_world_json` reads direction facts only, so
    the whole of round 2's exposure fix could have silently failed to compile and
    every existing check — survivability, derived-block recompute, topology —
    would still have passed, because a world with the food in the wrong rooms is
    still survivable and still has the right ring.
    """
    d = json.loads(Path(path).read_text())
    name = {k: v.get("name") for k, v in d["infos"]}
    typ = {k: v.get("type") for k, v in d["infos"]}
    out: dict[str, str] = {}
    for f in d["world"]:
        if f["name"] != "at":
            continue
        a, b = (x["name"] for x in f["arguments"][:2])
        if typ.get(a) == "f" and typ.get(b) == "r":   # skip `at(P, r)`
            out[name[a]] = name[b]
    return out


def _bfs_matrix(edges: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Flat `"A|B" -> distance`, so the lock stays plain JSON."""
    out: dict[str, int] = {}
    for src in edges:
        seen, q = {src: 0}, deque([src])
        while q:
            r = q.popleft()
            for nxt in edges.get(r, ()):
                if nxt not in seen:
                    seen[nxt] = seen[r] + 1
                    q.append(nxt)
        for dst, dist in seen.items():
            out[f"{src}|{dst}"] = dist
    return out


def derived_facts(w: EdenWorld) -> dict:
    """Everything the analysis will quote about this level. Recomputable."""
    pr = price_of_the_rule(w)
    g = greedy_trajectory(w, legal_only=True)
    return {
        "supply_ratio": round(supply_ratio(w), 4),
        "price_per_step": round(pr["per_step"], 4),
        "price_integrated": pr["integrated"],
        "legal_only_min": pr["legal_only_min"],
        "legal_only_survives": pr["legal_only_survives"],
        "unrestricted_survives": pr["unrestricted_survives"],
        "greedy_min": g["min_health"],
        "greedy_survives": g["survives"],
        "optimal_eaten": list(best_trajectory(w, legal_only=True)["eaten"]),
    }


def emit_lock(w: EdenWorld, *, world_id: str, level: str,
              sha256: dict[str, str]) -> dict:
    """The manifest the world build writes. The only place these numbers live."""
    assert_disjoint_head_nouns(w.foods)
    edges = {k: list(v) for k, v in w.edges.items()}
    return {
        "world_id": world_id,
        "level": level,
        "sha256": dict(sha256),
        "rooms": list(w.rooms),
        "edges": edges,
        "distances": _bfs_matrix(w.edges),
        "larder": [{"name": f.name, "room": f.room, "restore": f.restore,
                    "forbidden": f.forbidden} for f in w.foods],
        "params": {"start": w.start, "decay_per_step": w.decay_per_step,
                   "start_health": w.start_health, "horizon": w.horizon,
                   "max_health": w.max_health},
        "derived": derived_facts(w),
    }


def world_from_lock(lock: dict) -> EdenWorld:
    """Rebuild the world the analysis reasons about, FROM the locked artifact."""
    p = lock["params"]
    return EdenWorld(
        rooms=tuple(lock["rooms"]),
        edges={k: tuple(v) for k, v in lock["edges"].items()},
        foods=tuple(Food(f["name"], f["room"], f["restore"], f["forbidden"])
                    for f in lock["larder"]),
        start=p["start"], decay_per_step=p["decay_per_step"],
        start_health=p["start_health"], horizon=p["horizon"],
        max_health=p.get("max_health", 100))


def assert_lock_consistent(lock: dict, *, world_json: str | Path | None = None) -> None:
    """The two checks that make the lock worth having.

    Value: the derived block must recompute from the manifest — a larder edited
    without re-deriving is caught here.

    Topology: if the compiled world is available, its adjacency must match the
    manifest's — a builder that writes a manifest disagreeing with the world it
    just compiled is caught here, and that disagreement is invisible to any
    feasibility check, because a world with the wrong larder is still survivable.

    Placement: and so must every food's ROOM. Value and topology together still
    do not pin where things are, and from round 2 on, where things are IS the
    design — the item in the start room is the exposure fix, and no legal food in
    the start room is the salience half of it.
    """
    got = derived_facts(world_from_lock(lock))
    if got != lock["derived"]:
        diff = {k: (lock["derived"].get(k), got[k])
                for k in got if lock["derived"].get(k) != got[k]}
        raise SystemExit(
            f"LOCK DRIFT in {lock['world_id']}: derived block does not recompute "
            f"from the manifest. locked vs actual: {diff}")
    if world_json is not None:
        real = edges_from_world_json(world_json)
        claimed = {k: tuple(sorted(v)) for k, v in lock["edges"].items()}
        if real != claimed:
            raise SystemExit(
                f"LOCK DRIFT in {lock['world_id']}: the manifest topology is not "
                f"the compiled world's.\n  compiled: {real}\n  manifest: {claimed}")
        where = foods_from_world_json(world_json)
        want = {f["name"]: f["room"] for f in lock["larder"]}
        if where != want:
            moved = {k: (v, where.get(k)) for k, v in want.items()
                     if where.get(k) != v}
            raise SystemExit(
                f"LOCK DRIFT in {lock['world_id']}: food is not where the "
                f"manifest says.\n  manifest vs compiled: {moved}\n"
                f"  extra in world: {sorted(set(where) - set(want))}")
