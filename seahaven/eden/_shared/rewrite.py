"""Rewriting committed cells: name the target set, or nothing is written.

**This exists because a one-field bookkeeping stamp rewrote 481 cells and
destroyed 249 occasion labels.**

27 cells were short. 10 carried a header defect. The loop ran over all 481
because that was easier to write, and `occasion_of` falls back to file mtime —
so every cell from rounds 0 through 12, none of which carried a serving
timestamp, had its occasion label reset to the day of the rewrite. The
measurements were untouched. The labels, including round 10's LAT anchors, were
gone, and they were only recoverable because git history happened to hold the
commit dates.

The fault was not the repair. It was scope: a blast radius eighteen times the
defect, in a corpus where a file's mtime is load-bearing.

So a rewrite must SAY what it intends to touch, and this refuses anything else.
The guard is deliberately not clever — it takes an explicit set of paths and a
reason, and it will not widen. A caller that discovers more work mid-pass must
come back with a new, larger, stated scope rather than growing silently.

    with rewrite_scope(paths, reason="stamp X on the 10 cells missing it") as w:
        for p in paths:
            w.write(p, payload)          # a path outside `paths` raises

**Every write is content-compared first.** A rewrite that produces identical
bytes still costs an mtime, which is exactly the resource that was destroyed, so
a no-op write is skipped rather than performed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

__all__ = ["rewrite_scope", "ScopeViolation"]


class ScopeViolation(RuntimeError):
    """A write was attempted outside the declared target set."""


class _ScopedWriter:
    def __init__(self, allowed: set[Path], reason: str):
        self._allowed = allowed
        self.reason = reason
        self.written: list[Path] = []
        self.skipped: list[Path] = []

    def write(self, path, payload: dict) -> bool:
        """Write one cell. Returns whether bytes actually changed."""
        p = Path(path).resolve()
        if p not in self._allowed:
            raise ScopeViolation(
                f"REFUSED: {p.name} is outside this rewrite's declared scope.\n"
                f"  scope  : {len(self._allowed)} cell(s) — {self.reason}\n"
                "  A rewrite that touches cells it did not name is how 249\n"
                "  occasion labels were destroyed by a pass meant for 10.\n"
                "  Declare the wider scope deliberately, or narrow the loop.")

        body = json.dumps(payload, indent=2) + "\n"
        #: **mtime is a resource.** An identical write still burns one, and
        #: mtime is what the corpus falls back to for occasion labels on cells
        #: with no serving timestamp. A no-op write is not free.
        if p.exists() and p.read_text() == body:
            self.skipped.append(p)
            return False
        p.write_text(body)
        self.written.append(p)
        return True


@contextmanager
def rewrite_scope(paths, *, reason: str):
    """Declare exactly which cells a rewrite may touch.

    `reason` is required and is not decoration: it is what the refusal message
    quotes back, so a scope violation reads as "this pass was for X and tried to
    touch Y" rather than as a permissions error.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "a rewrite must state its reason. The scope guard exists because a "
            "pass nobody had to justify grew to eighteen times its defect.")
    allowed = {Path(p).resolve() for p in paths}
    if not allowed:
        raise ValueError("empty rewrite scope: nothing to do, so do nothing")
    w = _ScopedWriter(allowed, reason.strip())
    yield w
    print(f"  rewrite [{w.reason}]: {len(w.written)} written, "
          f"{len(w.skipped)} unchanged, {len(allowed)} in scope")
