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
           "episodes_for"]


def digest_files(root: Path, rels: tuple[str, ...]) -> dict[str, str]:
    """sha256 per path, relative to the repo root."""
    return {r: hashlib.sha256((root / r).read_bytes()).hexdigest() for r in rels}


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
