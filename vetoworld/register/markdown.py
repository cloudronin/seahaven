"""One markdown table renderer, so `tables/` cannot grow fifteen of them.

The bundle spec asks for tables "ready to drop into LaTeX/md". Every register
emitter currently prints fixed-width ASCII sized for a terminal, which is a
different artifact from a table a paper consumes — and re-rendering by hand at
paste time is precisely the transcription step the bundle exists to remove.

**This renders from DATA, never from captured stdout.** Parsing an emitter's
printed columns back into cells would make the table a function of the print
format, so a cosmetic width change would silently alter the paper's input. Every
emitter feeding `tables/` therefore exposes a data function beside its report
function — the split `register/c5.py` already uses (`rows()` beside `report()`).
"""

from __future__ import annotations

__all__ = ["table", "kv_table"]


def _cell(v: object) -> str:
    """A value as one table cell. Pipes are escaped; nothing else is mangled."""
    if v is None:
        return "—"
    if isinstance(v, float):
        #: Trailing-zero-stable, because a column of 0.5 / 0.50 / 0.500 reads as
        #: three different precisions and none of them is the measured one.
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s or "0"
    if isinstance(v, (tuple, list)):
        #: **An empty sequence is a VALUE, not a missing cell.** `break.gone`
        #: publishes `[]` — "no model breaks any more" — and rendering it as
        #: blank made a finding look like an omission.
        if not v:
            return "`[]` (empty)"
        return " / ".join(_cell(x) for x in v)
    return str(v).replace("|", r"\|").replace("\n", " ")


def table(headers, rows) -> str:
    """A GitHub/pandoc pipe table. `rows` is an iterable of iterables.

    Alignment is left for text and right for numbers, decided per COLUMN from
    the data rather than per cell, so a column does not wobble when one row
    happens to hold a string.
    """
    headers = [str(h) for h in headers]
    body = [[_cell(v) for v in row] for row in rows]

    numeric = []
    for i in range(len(headers)):
        vals = [r[i] for r in body if i < len(r) and r[i] not in ("—", "")]
        numeric.append(bool(vals) and all(
            v.replace(".", "", 1).replace("-", "", 1).replace("/", "").replace(
                " ", "").isdigit() for v in vals))

    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---:" if n else ":---" for n in numeric) + "|"]
    for r in body:
        cells = list(r) + [""] * (len(headers) - len(r))
        out.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return "\n".join(out)


def kv_table(pairs, *, key="field", value="value") -> str:
    """Two columns, for constant blocks — arms, metric definitions, gates."""
    return table([key, value], [[k, v] for k, v in pairs])
