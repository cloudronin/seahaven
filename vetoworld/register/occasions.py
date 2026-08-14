"""**The occasion audit.** Which figures in the manuscript compare cells that
were served on different days.

The programme measured a **0.319 between-occasion level shift** on one model
between two days, ruled out batch composition and prefix cache, and could not
establish a mechanism. A deployment change on the provider's side is consistent
and untestable from here. So any figure whose two sides were served on different
occasions has a second explanation for its difference, and the manuscript has to
say so **in place** rather than once in a limitations section.

Two design commitments, both of which exist because the obvious version of this
file would be quietly wrong.

**1. The read set is RECORDED, not declared.** `Claim.cells` is prose for a
human; if this module trusted it, a figure could start reading a different cell
and the audit would keep describing the old one. Instead each claim is run with
`corpus.load_cell` wrapped, and what it actually opened is what gets audited. A
figure cannot lie about its own inputs.

**2. `mtime` is not a serving date, so it can never produce a "same occasion"
verdict.** Real serving timestamps exist on **10 of 259** cells — rounds 13 and
14 and the two timing probes. Everything else falls back to file
last-written, which for a gap-filled cell is its *last* attempt, not its first.
Two mtimes that happen to agree are not evidence that two cells were served
together, so the verdict is `unknown-mtime` and the flag says the corpus cannot
settle it.

**And mtime does not merely fail to prove sameness — it actively under-detects
difference.** The corpus spans three mtime days, but rounds 11, 12 and 13 were
three separate sweeps that all land on one of them. A timestamps-only audit would
call those comparisons same-day and be wrong about the thing it exists to catch.

So a second signal is reported beside it, and this one is **recorded provenance
rather than a filesystem artifact**: the round tag in the cell's own name. Cells
from different rounds came from different sweeps, and a sweep is at minimum a
distinct serving session. `break.gone` reads cells from rounds 10, 11, 12 and 13;
no timestamp is needed to know that figure spans sittings.

The honest summary this produces is uncomfortable and correct: **almost no
multi-cell figure in the manuscript can be shown to be occasion-clean.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seahaven.eden._shared import corpus as C

from .bands import OCCASION_COMPONENT

#: A serving occasion is a **calendar day**. The measured effect is a
#: between-DAY shift, and the two timing-probe cells that deliberately share an
#: occasion were served 79 seconds apart, so a finer grain would split a pair the
#: design intended to be together.
_DAY = 10


@dataclass(frozen=True)
class Row:
    fid: str
    cells: tuple[Path, ...]
    labels: tuple[tuple[str, str], ...]   # (value, source) per cell, aligned
    verdict: str                          # yes | no | unknown-mtime | single-cell
    flag: str                             # what the manuscript carries

    @property
    def days(self) -> tuple[str, ...]:
        return tuple(sorted({v[:_DAY] for v, _s in self.labels}))

    @property
    def sweeps(self) -> tuple[str, ...]:
        """The distinct round tags the figure reads across.

        Recorded, not inferred: the tag is in the cell's filename because a
        sweep wrote it there. More than one means the figure spans sittings even
        where the timestamps cannot show it.
        """
        got = (C.parse_cell_name(p.name) for p in self.cells)
        return tuple(sorted({g["round"] for g in got if g}))

    @property
    def needs_flag(self) -> bool:
        """A figure reading one cell cannot span occasions, and a figure proven
        same-day within a single sweep does not need the caveat. Everything else
        does — including a figure whose timestamps agree but whose cells came
        from different sweeps, which is the case mtime alone would miss."""
        return self.verdict in ("no", "unknown-mtime") or len(self.sweeps) > 1


def record_reads(fn):
    """`(value, paths)` — run `fn` and capture every cell it actually opened.

    Wrapping the loader rather than reading a declared list is the whole point:
    the audit describes the figure's real inputs even when its docstring is
    stale.
    """
    seen: list[Path] = []
    original = C.load_cell

    def spy(path):
        seen.append(Path(path))
        return original(path)

    C.load_cell = spy
    try:
        value = fn()
    finally:
        C.load_cell = original
    return value, seen


def _label(path: Path) -> tuple[str, str]:
    meta = C.load_cell(path).get("meta", {})
    return C.occasion_of(path, meta)


_SHIFT = (f"the programme measured a {OCCASION_COMPONENT} between-occasion shift "
          "on one model with the mechanism unresolved, so a difference here has "
          "a second explanation")


def _flag_for(verdict: str, days: tuple[str, ...], sweeps: tuple[str, ...],
              n: int) -> str:
    across = (f" ACROSS SWEEPS e{', e'.join(sweeps)}" if len(sweeps) > 1 else "")
    if verdict == "single-cell":
        return ("" if len(sweeps) < 2 else
                f"reads {n} cell(s) from sweeps e{', e'.join(sweeps)}; {_SHIFT}.")
    if verdict == "yes":
        base = f"all {n} cells served on {days[0]}"
        return (f"{base}; occasion-clean." if len(sweeps) < 2 else
                f"{base}, but{across} — treat as cross-occasion; {_SHIFT}.")
    if verdict == "no":
        return (f"SPANS {len(days)} SERVING OCCASIONS ({', '.join(days)})"
                f"{across}. {_SHIFT[0].upper() + _SHIFT[1:]}.")
    spread = ("written on one day" if len(days) == 1
              else f"written across {len(days)} days ({', '.join(days)})")
    tail = (f" It DOES read{across}, which is recorded provenance rather than a "
            "timestamp, so the comparison spans sittings whatever the mtimes "
            "say." if len(sweeps) > 1 else "")
    return (f"OCCASION UNKNOWN. {n} cells, {spread}, but the label is file "
            "mtime — when the file was last WRITTEN, which for a gap-filled "
            "cell is its last attempt. The corpus cannot establish that these "
            f"were served together, and cannot establish that they were not.{tail}")


def audit_claim(claim) -> Row:
    _value, read = record_reads(claim.emit)
    cells = tuple(sorted(set(read)))
    labels = tuple(_label(p) for p in cells)
    days = sorted({v[:_DAY] for v, _s in labels})
    if len(cells) < 2:
        verdict = "single-cell"
    elif any(s != "wall_start_epoch" for _v, s in labels):
        verdict = "unknown-mtime"
    elif len(days) == 1:
        verdict = "yes"
    else:
        verdict = "no"
    row = Row(claim.fid, cells, labels, verdict, "")
    return Row(claim.fid, cells, labels, verdict,
               claim.occasion
               or _flag_for(verdict, tuple(days), row.sweeps, len(cells)))


def audit() -> list[Row]:
    from .claims import CLAIMS
    return [audit_claim(c) for c in CLAIMS]


def unflagged(rows: list[Row] | None = None) -> list[str]:
    """**The register regression.** Figure ids that span (or may span) occasions
    and carry no flag of their own.

    `verify` fails on a non-empty result. A new figure that compares cells across
    days is the exact thing this programme cannot afford to publish unmarked, and
    the check is mechanical so it does not depend on anyone remembering.
    """
    from .claims import CLAIMS
    by_fid = {c.fid: c for c in CLAIMS}
    return [r.fid for r in (audit() if rows is None else rows)
            if r.needs_flag and not by_fid[r.fid].occasion]
