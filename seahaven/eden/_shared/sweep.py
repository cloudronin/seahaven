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

    # **THE HEADER MUST NOT CONTRADICT THE RUNS IT SUMMARISES.**
    #
    # This wrote `runs` and left `n_runs_completed` and `failed_runs` at their
    # first-attempt values, so a gap-filled cell claimed a completeness it had
    # since fixed: round 21's Kimi-K3 A1 LAT carried 47 real episodes across 47
    # distinct seeds under a header reading `n_runs_completed: 10` and
    # `failed_runs: 38`.
    #
    # Harmless only because every measurement path reads `episodes()` and none
    # reads the header — which is the argument that has preceded most of this
    # programme's retractions. A field that is wrong and unread is a field that
    # becomes wrong and read.
    #
    # `failed_runs` is rebuilt rather than cleared: a seed that failed and was
    # then filled is no longer a failure, but one that failed and was never
    # asked for again still is, and losing that would overstate the cell.
    #
    # **Records written before 2026-08-16 carry no seed** — only a `run` index,
    # which is an index within ONE attempt and so means nothing across them.
    # Those cannot be matched to anything, so the only honest reconciliation is
    # completeness: a cell holding every episode it asked for has no
    # outstanding failure, whatever its old records say. A short cell keeps
    # them, because there we genuinely cannot tell which were resolved.
    # `runner.py` now stamps the seed, so this degrades to the precise rule for
    # everything served from here on.
    have = {r["seed"] for r in res["runs"] if "seed" in r}
    res["n_runs_completed"] = len(res["runs"])
    old = res.get("failed_runs", [])
    req = res.get("n_runs_requested", len(res["runs"]))
    complete = len(res["runs"]) >= req
    res["failed_runs"] = [] if complete else [
        f for f in old if f.get("seed") is None or f["seed"] not in have]

    # **How many episodes are actually still missing, stated as a number.**
    #
    # Seedless records cannot say WHICH failures a gap-fill resolved, and
    # guessing would be inventing. But the COUNT is not ambiguous: a cell that
    # asked for 48 and holds 47 is short by exactly one, however many stale
    # records it carries. Round 21's Kimi-K3 kept 38 records beside 47
    # episodes — 37 of them resolved, unidentifiably — and any reader counting
    # records would have put the cell at less than a quarter served.
    #
    # So the identity stays unknown and the quantity stops being.
    res["outstanding_episodes"] = max(0, req - len(res["runs"]))
    return res, list(seeds)
    return res, list(seeds)
