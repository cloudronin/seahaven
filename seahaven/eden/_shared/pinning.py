"""The pin machinery, once.

`payload`, `current_hash`, `assert_pinned`, `world_lock_paths` and
`episodes_for` were copy-pasted into all eleven round modules — `current_hash`
byte-identically in every one.

**A round's own source is not hashed**, so a round may delegate here freely:
`payload()` covers `ARTIFACTS` (measurement modules), the world locks, and the
*values* of the round's constants. Nothing in `_shared/` may ever be added to a
round's `ARTIFACTS`; doing so would make the pins depend on the refactor.

`tests/test_pin_invariance.py` holds 11 live hashes, 6 retired recompute
functions and 7 frozen-artifact digests, all captured before this file existed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["digest_files", "hash_payload", "assert_hash", "world_lock_paths",
           "episodes_for", "worldspec_digest"]


def digest_files(root: Path, rels: tuple[str, ...]) -> dict[str, str]:
    """sha256 per path, relative to the repo root."""
    return {r: hashlib.sha256((root / r).read_bytes()).hexdigest() for r in rels}


def worldspec_digest(levels: tuple[str, ...]) -> dict[str, str]:
    """The DERIVED spec per world, hashed one world at a time.

    **This is round 6's fix, finally applied to the second registry.**

    Rounds hashed `seahaven/fidelity/worldspec.py` whole, and `SETTINGS` inside
    it is a registry keyed by world. So registering a world that nobody had
    served changed the bytes and broke every live pin — five of them when LAT2
    was added in round 14, which recorded the defect, named this fix, and
    predicted the recurrence: "world 21 will break round 14 too". Round 6 had
    already hit the identical shape with `build_eden_worlds.py` and fixed it by
    hashing each world's own `BUILD.lock.json` instead of the builder.

    Hashing the derived spec rather than the setting sentence alone is what
    keeps the coverage the file-hash had: the sentence is the prompt-affecting
    part of `SETTINGS`, but `load()` also decides rooms, start room, takeables
    and kinds, and a change there would move a served prompt without moving any
    sentence. Both now travel, per world, and neither is disturbed by a
    stranger being registered beside them.

    **`forms` closes a gap the file-hash covered and the structure alone does
    not.** `match_forms` and `UNSAFE_HEADS` are not only used inside `load()` —
    `fidelity/detectors.py` and `fidelity/adherence.py` call `match_forms` at
    measurement time, and neither of those is in any round's `ARTIFACTS`. So
    their behaviour reached the pins only transitively, through hashing
    `worldspec.py` whole. Hashing the resolved forms for THIS WORLD'S OWN nouns
    keeps that coverage while staying immune to a stranger being registered.

    `path` is deliberately excluded — it is absolute, so including it would
    make every pin depend on where the repo is checked out.
    """
    import json

    from ...fidelity import worldspec as WS

    out: dict[str, str] = {}
    for lv in levels:
        wid = f"world_eden_{lv}"
        spec = WS.load(wid)
        nouns = sorted(set(spec.takeable) | set(spec.rooms))
        body = json.dumps({
            "setting": spec.setting,
            "start_room": spec.start_room,
            "rooms": list(spec.rooms),
            "takeable": list(spec.takeable),
            "kinds": [[k, list(v)] for k, v in spec.kinds],
            "forms": {n: list(WS.match_forms(n)) for n in nouns},
        }, sort_keys=True)
        out[wid] = hashlib.sha256(body.encode()).hexdigest()
    return out


def hash_payload(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def assert_hash(got: str, pinned: str, label: str, *, empty_hint: str) -> None:
    """The standard refusal. Wording preserved from the round modules because
    the phrasing is what tells a reader the fix is to revert, not to re-pin."""
    if not pinned:
        raise SystemExit(empty_hint)
    if got != pinned:
        raise SystemExit(
            f"{label} PIN BROKEN\n  pinned {pinned}\n  actual {got}\n"
            "  A constant, a measurement module, or a world lock changed after "
            "the freeze. Either revert it, or re-pin DELIBERATELY and say so in "
            "the commit.")


def world_lock_paths(levels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"worlds/world_eden_{lv}/BUILD.lock.json" for lv in levels)


def episodes_for(arm: str, a1: int, a0: int) -> int:
    return a1 if arm == "A1" else a0
