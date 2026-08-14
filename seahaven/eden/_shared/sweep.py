"""The one cell runner: grid, resume guard, gap-fill, spend.

`run_cell` existed ten times at up to 99% line-similarity within chains, sharing
a fifteen-line docstring verbatim. This is that function, once.

**The dependency arrow points one way.** `run_cell` takes the policy as a
PARAMETER and this module imports nothing from the public `vetoworld`
package — backends live there, sweeping lives here, and a circular import is
prevented by construction rather than by discipline. `tests/test_layering.py`
asserts it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .corpus import episodes, load_cell, missing_seeds

__all__ = ["run_cell", "resume_plan"]


def resume_plan(grid, want_for, path_for, pin_key, pin_value):
    """`(todo, gaps)` — which cells to run and which seeds each is short.

    **A partial cell is not a completed cell.** An HTTP 402 mid-sweep once left
    cells with 12 and 0 surviving episodes carrying the right pin; keyed on the
    pin alone they would have been skipped forever and a rate quoted over a
    denominator of 12 beside rates over 24.
    """
    todo, gaps = [], {}
    for c in grid:
        want = want_for(c)
        p = path_for(c)
        if not p.exists():
            todo.append(c)
            continue
        meta = load_cell(p).get("meta", {})
        if meta.get(pin_key) != pin_value:
            raise SystemExit(
                f"{p} exists but carries {pin_key}={meta.get(pin_key)!r}, not "
                f"{pin_value}. That is a cell from a different freeze; refusing "
                "to treat it as done. Move it aside deliberately.")
        if len(episodes(load_cell(p))) < want:
            gaps[c] = missing_seeds(p, want_for.seed0, want)
            todo.append(c)
    return todo, gaps


def run_cell(policy, *, runs, seed0, level, arm, terminal_at_zero,
             serve, path=None, seeds=None):
    """Run a whole cell, or — with `seeds` — only the episodes it is missing.

    **A retry used to re-draw the whole cell, and that is expensive arithmetic.**
    A sporadic failure at roughly 1 episode in 24 leaves a clean 24-episode sweep
    only ~36% likely, so expected cost to complete is about triple the nominal.
    Round 6's W3 A1 cell cost $3.89 across three attempts to buy one episode,
    twice discarding 23 good ones. Filling the gap makes it near-certain in one
    retry, and the completeness guard is untouched — the fix belongs on this side.

    Safe because the step schedule is FLAT: a one-episode call at
    `seed0=<missing>` is construction-identical to the episode it replaces. That
    is asserted below rather than trusted, because an index-dependent schedule is
    exactly what made the H=36 gate measure nothing.

    `serve(runs, seed0)` is injected — this module never constructs a backend.
    """
    if seeds is None:
        return serve(runs, seed0), None

    res = load_cell(Path(path))
    keep = episodes(res)
    want_steps = {len(r["commands"]) for r in keep}
    for sd in seeds:
        got = serve(1, sd)
        new = episodes(got)
        for r in new:
            if want_steps and len(r["commands"]) not in want_steps:
                raise SystemExit(
                    f"{level} {arm} seed {sd}: filled episode ran "
                    f"{len(r['commands'])} steps, the cell's others ran "
                    f"{sorted(want_steps)}. The schedule is not flat and a "
                    "gap-fill is NOT construction-identical. Refusing.")
        keep += new
    res["runs"] = sorted(keep, key=lambda r: r["seed"])
    return res, list(seeds)
