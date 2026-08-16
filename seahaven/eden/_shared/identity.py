"""Was the cell served by the model it claims? The per-cell gate.

**This module exists because the answer was "no" for 166 cells and nothing
asked.** `Backend` fixes the request's `"model"` field at construction from
`spec.model`, and `_round_cells` built ONE Backend for a whole grid, so the
grid's model tuple selected a filename and a price while one model served
every cell. Rounds 15-19 are one model wearing twenty-two name tags.

The rule that would have caught it existed already — `resolved_model_string`
was recorded from round 13 onward, written to check one specific model's
identity — and was never generalised into a standing precondition. So it sat
in the metadata of every affected cell, correct and unread, for weeks.

**Four states, because the corpus cannot honestly support two.**

    VERIFIED     the endpoint reported what it served and it matched
    CORRECTED    it did not match; the record now carries what ACTUALLY
                 served, with the request retained beside it (issue #113)
    UNVERIFIED   no report on record. The cell predates the field.
    MISLABELLED  it did not match and the record has NOT been corrected.
                 Reading one is an error, not a warning.

`UNVERIFIED` is the state that matters most and is easiest to get wrong. It is
**not** a synonym for VERIFIED. 251 of 425 cells are UNVERIFIED, and the belief
that they are fine anyway — because those rounds served one model per
invocation — is the same species of reasoning that let this bug live: an
argument about how the code *ought* to behave, standing in for a record of what
it did. The argument is probably right. It is still not evidence, and this
module refuses to render it as evidence.

**Measurement keys on `served`, never on `requested` and never on the
filename.** The filenames of the corrected cells still spell the model that was
asked for, because correcting them would collide — all eight of round 18's LAT
cells would become the same name. So a filename is a record of a request, and
`parse_cell_name(...)["model"]` must not be used as a model identity.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "VERIFIED", "CORRECTED", "UNVERIFIED", "MISLABELLED", "READABLE",
    "Identity", "model_identity", "assert_identity", "IdentityViolation",
    "CORRECTION_KEY", "ISSUE",
]

VERIFIED = "VERIFIED"
CORRECTED = "CORRECTED"
UNVERIFIED = "UNVERIFIED"
MISLABELLED = "MISLABELLED"

#: States a measurement may be computed from. `MISLABELLED` is excluded by
#: construction: the cell's own metadata says it was served by someone else, so
#: there is no defensible way to attribute it.
READABLE = frozenset({VERIFIED, CORRECTED, UNVERIFIED})

#: Where the correction records itself on a cell. Distinct from the flat
#: `requested_model`/`served_model` fields so that the EDIT has provenance too —
#: a reader can tell a cell that was born correct from one that was repaired.
CORRECTION_KEY = "model_identity_correction"
ISSUE = 113

#: **Sweeps whose per-model attribution was destroyed, and which therefore
#: supply no MEASUREMENT of anyone.**
#:
#: Every cell in these five rounds was served by one model. They are kept, and
#: they are real data about that model — they contain the only identical-input
#: repeatability study the programme has — but they are not measurements of the
#: cohorts their filenames name, and the register must not draw a canonical
#: cell from them.
#:
#: Without this, the damage compounds instead of stopping: cogito acquires 14
#: tied candidates for "canonical cell" at (LAT, A1, e16), the picker takes one
#: arbitrarily, and it displaces cogito's genuine round-12 measurement. The
#: pre-registered prediction check silently went from "5 of 6 consistent" to
#: "6 of 6" that way — a retracted replicate quietly improving a pinned result,
#: which is the worst possible direction for an error to move.
#:
#: The reference channel still READS them, and correctly reports them as one
#: model going quiet. Judging a sweep and crediting a model are different acts.
RETRACTED_SWEEPS = frozenset({"15", "16", "17", "18", "19"})


class IdentityViolation(RuntimeError):
    """A cell was read for measurement whose served model is not established.

    Separate from a plain assertion because the remedy is specific and is not
    "fix the test": either relabel the cell (`vworld corpus relabel`) or drop
    it from the read.
    """


class Identity(NamedTuple):
    requested: str | None
    served: str | None
    status: str

    @property
    def trustworthy(self) -> bool:
        """The endpoint itself confirmed what it served."""
        return self.status in (VERIFIED, CORRECTED)


def model_identity(meta: dict) -> Identity:
    """`(requested, served, status)` for one cell's `meta`.

    Never raises — classification must work on the broken cells too, or the
    verb that repairs them could not find them.
    """
    corrected = meta.get(CORRECTION_KEY)
    if corrected or ("served_model" in meta and "requested_model" in meta):
        return Identity(meta.get("requested_model"), meta.get("served_model"),
                        CORRECTED if corrected else VERIFIED)

    requested = meta.get("served_name")
    resolved = meta.get("resolved_model_string")
    if resolved is None:
        return Identity(requested, requested, UNVERIFIED)
    if resolved == requested:
        return Identity(requested, resolved, VERIFIED)
    return Identity(requested, resolved, MISLABELLED)


def assert_identity(meta: dict, *, where: str = "read",
                    allow_unverified: bool = True) -> Identity:
    """The precondition. Assert this wherever `saw == n` is asserted.

    `allow_unverified` is True by default because 251 committed cells predate
    the field and refusing them would retract the whole programme rather than
    the part that is wrong. Serve-time callers pass False: a cell written today
    has no excuse for lacking the record.
    """
    ident = model_identity(meta)
    if ident.status == MISLABELLED:
        raise IdentityViolation(
            f"{where}: this cell was served by {ident.served!r} but claims "
            f"{ident.requested!r}. It has not been corrected. See issue "
            f"#{ISSUE}; run `vworld corpus relabel` to record what actually "
            "served, or exclude the cell from this read.")
    if ident.status == UNVERIFIED and not allow_unverified:
        raise IdentityViolation(
            f"{where}: no served-model record on this cell. A cell written "
            "now must carry one — that is the whole of the fix for issue "
            f"#{ISSUE}.")
    return ident


def tally(metas) -> dict[str, int]:
    """Status counts, for the surfaces that report corpus health."""
    out = {VERIFIED: 0, CORRECTED: 0, UNVERIFIED: 0, MISLABELLED: 0}
    for m in metas:
        out[model_identity(m).status] += 1
    return out
