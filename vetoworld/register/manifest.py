"""MANIFEST.md — every number the skeleton quotes, bound to what recomputes it.

**Generated from `CLAIMS`, never hand-written.** `register/claims.py` already
holds `Claim(fid, value, emit, cells, ...)` and documents `fid` as "figure id,
cited by the manuscript" — the binding layer this file needs was built before it
was asked for. Maintaining a second copy by hand is the duplicated-vocabulary
drift this programme has now caught in `doctor`'s ROUNDS, in `emit`'s ARTIFACTS,
and in the registry-disjointness lists, so the manifest is derived instead.

Each row is RECOMPUTED at build time, not read from `Claim.value`. A manifest
that quoted the published literal back to itself would agree with the paper
whatever the corpus said, which is the one thing it must not do.

**Supersessions are content, not errata.** Where a number quoted in a past
session differs from the emitted one, the difference is recorded here with both
values. Spec rule 4: those become corrections-section material rather than
silent fixes.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["rows", "SUPERSEDED", "write_manifest"]


#: **Numbers quoted in planning that the register supersedes.**
#:
#: Each entry: (what, quoted, emitted-by, note). These are kept because the
#: bundle spec's own premise is that conversation-quoted figures go stale, and
#: the sharpest instance is the spec itself — it asked for "all 13" corrections
#: while the register held five and the log numbered to eleven.
SUPERSEDED = [
    ("corrections count", "13",
     "vworld emit corrections",
     "The spec asked for 'all 13'. Three sources disagreed: the spec's 13, "
     "the register's rows, and research-log's [CORRECTION] numbering. A "
     "hand-maintained number living in conversation prose — the family this "
     "programme keeps catching in code. The register is being backfilled from "
     "the log with every row verified against its commit; the paper quotes the "
     "EMITTED count."),
    ("Exhibit 1 availability", "$0 and available now",
     "vworld emit exhibit-1",
     "Built, and reports ZERO. Money does not unblock it at any price: all 28 "
     "cross-provider pairs are seven models, and all seven are unservable on "
     "Together — six non-serverless, Kimi-K3 empty-content under the pinned "
     "request form. See SUBSTRATE_PENDING for the route that is open."),
    ("E1 power window", "bounds computed at n=17",
     "vworld emit correlations",
     "Recompute at the live n. The n=17 closed-arms bounds predate the "
     "retraction and are stale."),
]


def rows():
    """`[(fid, value_now, matches_published, function, cells, note)]`.

    `matches_published` is the check `vworld verify` performs, carried into the
    manifest so a reader sees per-figure whether the published literal still
    recomputes — rather than having to trust that verify was run.
    """
    from .claims import CLAIMS

    out = []
    for c in CLAIMS:
        try:
            got = c.emit()
            ok = (got == c.value)
        except Exception as e:                                # noqa: BLE001
            got, ok = f"** RAISED: {e!r} **", False
        out.append({
            "fid": c.fid,
            "published": c.value,
            "recomputed": got,
            "agrees": ok,
            #: **Cite the entry, not the closure.** Every `Claim.emit` is a
            #: lambda, so `__qualname__` renders `<lambda>` for all 22 — which
            #: is not a citable function and not locatable by a reader. The
            #: stable, greppable address of a figure is its fid in the claims
            #: register, which is exactly what the paper is meant to cite.
            "function": f"register/claims.py::{c.fid}",
            "cells": c.cells,
            "measured": c.measured,
            "note": c.note,
        })
    return out


def write_manifest(out: Path, prov: dict) -> int:
    """Write `MANIFEST.md`. Returns the count of figures that do NOT recompute."""
    from .bundle import header
    from .markdown import table

    data = rows()
    bad = [r for r in data if not r["agrees"]]

    body = [
        "# MANIFEST — figure → emitting function → file",
        "",
        header("vworld bundle", prov, comment=True),
        "",
        "Every figure the skeleton may quote, the function that recomputes it "
        "from committed cells, and whether it still agrees with the published "
        "literal. **Cite the `fid`, never the number.** A claim absent from "
        "this table is not citable: the bundle's bar is the paper's bar.",
        "",
        f"Generated from `register/claims.py` — {len(data)} registered figures, "
        f"{len(bad)} not recomputing.",
        "",
        "## Figures",
        "",
        table(
            ["fid", "published", "recomputed", "agrees", "function", "cells"],
            [[r["fid"], r["published"], r["recomputed"],
              "yes" if r["agrees"] else "**NO**", f"`{r['function']}`",
              r["cells"]] for r in data]),
        "",
    ]

    derived = [r for r in data if not r["measured"]]
    if derived:
        body += [
            "## Derived, not measured",
            "",
            "These come from round 9's pre/post-crossing identity rather than "
            "from measured episodes. The identity has been checked six times — "
            "five consistent, one off by 0.29 with an occasion effect as an "
            "unresolvable alternative — and a derived figure inherits both "
            "facts. Label them in prose.",
            "",
            table(["fid", "note"], [[r["fid"], r["note"]] for r in derived]),
            "",
        ]

    body += [
        "## Superseded numbers",
        "",
        "Figures quoted in planning or conversation that the register "
        "supersedes. Per the bundle's rule 4 these are **corrections-section "
        "material, not errata** — each one is an instance of why the bundle "
        "exists.",
        "",
        table(["what", "quoted", "emitted by", "why it moved"],
              [[s[0], s[1], f"`{s[2]}`", s[3]] for s in SUPERSEDED]),
        "",
        "## Artifacts in this bundle",
        "",
        "`register/` holds the verbatim stdout of each emitter, with its own "
        "provenance header. `tables/` holds the same data rendered as markdown "
        "for paste. `log-extracts/` holds the structured ledgers. `exhibits/` "
        "holds Exhibit 1 and any figures.",
        "",
    ]

    if bad:
        body += [
            "## ⚠ Figures that do not recompute",
            "",
            "**These are not citable.** A published literal that no longer "
            "recomputes is either a corpus change or a defect, and either way "
            "the paper must not quote it until resolved.",
            "",
            table(["fid", "published", "recomputed"],
                  [[r["fid"], r["published"], r["recomputed"]] for r in bad]),
            "",
        ]

    (Path(out) / "MANIFEST.md").write_text("\n".join(body))
    return len(bad)
