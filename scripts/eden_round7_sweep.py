"""Run the round-7 screen: LAT + W2 + W3 x {A1, A0} x m=24 on one model.

The prompt now carries `EDEN_RECOVERY`, so these cells are a NEW GENERATION
and pool with nothing before them. LAT is back in the grid because the change
invalidates its committed cells as a comparison; the WORLD is untouched.

Carries the round-4 driver's complete resume guard — pin check AND full episode
count — because the round-3 driver's version was missing the second half once and
a partial cell would have been skipped forever.

`--confirm W2 W3` runs A1-only confirmatory cells at offset seeds for worlds whose
screen reading landed near a band edge. See `round7.band_verdict`: the exposure is
asymmetric and the false failures live at the upper edge, so one $1.25 cell is
cheaper than re-authoring a world that was never broken.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round7 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402

OUT = Path("results")


def cell_path(model: str, arm: str, level: str, confirm: bool = False) -> Path:
    # A confirmatory cell writes its OWN file: it is 24 additional episodes at
    # offset seeds, not a replacement for the screen's 24. Overwriting would
    # discard the reading the confirmation exists to check, and the two must be
    # reported separately before they are pooled.
    tag = f"{level}confirm" if confirm else level
    return OUT / f"eden_e7_{model.replace('/', '__')}__{arm}__{tag}.json"


def missing_seeds(p: Path, s0: int, m: int) -> list[int]:
    """Seeds the cell is SHORT, from the file rather than from a count."""
    d = json.loads(p.read_text())
    have = {r["seed"] for r in d.get("runs", []) if r.get("commands") and "seed" in r}
    return sorted(set(range(s0, s0 + m)) - have)


def run_cell(model: str, arm: str, level: str, confirm: bool = False,
             seeds: list[int] | None = None) -> dict:
    """Run a whole cell, or — with `seeds` — only the episodes it is missing.

    **A retry used to re-draw all 24 and that is expensive arithmetic.** A
    sporadic endpoint failure at roughly 1 episode in 24 leaves a clean 24-episode
    sweep only ~36% likely, so expected cost to complete a cell is about triple
    the nominal. Round 6's W3 A1 cell cost $3.89 across three attempts to buy one
    episode, twice discarding 23 good ones.

    Filling the gap makes it near-certain in one retry. **The 24/24 guard is
    untouched** — the fix belongs on this side, not in weakening what counts as a
    complete cell.

    Safe because EdenBench's step schedule is FLAT: `eden_schedule` returns
    `(horizon,) * runs`, so `_steps_for(i, ...)` gives 33 at every index and a
    one-episode call at `seed0=<missing>` is construction-identical to the
    episode it replaces. Asserted below rather than trusted — an index-dependent
    schedule is exactly what made the H=36 gate measure nothing.
    """
    pi, po = R.COHORT[model]
    ep = Endpoint(base_url=R.BASE_URL, served_name=model,
                  api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
    t0 = time.time()
    m_ = R.CONFIRM_EPISODES if confirm else R.EPISODES_PER_CELL
    s_ = R.CONFIRM_SEED0 if confirm else R.SEED0

    if confirm:
        # **A seed collision would silently double-count identical episodes.**
        # Checked against what is on disk rather than against the constants,
        # because the constants are the thing that would be wrong.
        prior = cell_path(model, arm, level)
        if prior.exists():
            have = {r["seed"] for r in json.loads(prior.read_text()).get("runs", [])
                    if "seed" in r}
            clash = have & set(range(s_, s_ + m_))
            if clash:
                raise SystemExit(
                    f"{model} {level}: confirmatory seeds collide with the "
                    f"screen cell ({len(clash)} overlapping, e.g. "
                    f"{sorted(clash)[:5]}). Pooling would double-count.")

    # `steps=30` is a placeholder the runner OVERRIDES from the lock's horizon
    # (33 here). Passing the real number would still be wrong, because
    # `_steps_for` rescales the schedule -- that is the bug that made the H=36
    # gate measure nothing. The lock is the authority.
    def _one(runs, seed0):
        return run_fidelity(ep, None, runs=runs, steps=30, seed0=seed0,
                            world_id=f"world_eden_{level}", narrate=False,
                            eden_level=level, eden_arm=arm)

    if seeds is None:
        res = _one(m_, s_)
        filled = None
    else:
        # The existing cell, extended. Never overwritten: the episodes already on
        # disk are good and re-drawing them is what this exists to stop.
        p = cell_path(model, arm, level, confirm=confirm)
        res = json.loads(p.read_text())
        keep = [r for r in res.get("runs", []) if r.get("commands")]
        want_steps = {len(r["commands"]) for r in keep}
        for sd in seeds:
            got = _one(1, sd)
            new = [r for r in got.get("runs", []) if r.get("commands")]
            for r in new:
                if want_steps and len(r["commands"]) not in want_steps:
                    raise SystemExit(
                        f"{level} {arm} seed {sd}: filled episode ran "
                        f"{len(r['commands'])} steps, the cell's others ran "
                        f"{sorted(want_steps)}. The schedule is not flat and a "
                        "gap-fill is NOT construction-identical. Refusing.")
            keep += new
        res["runs"] = sorted(keep, key=lambda r: r["seed"])
        filled = list(seeds)
    u = dict(ep.usage_total)
    prior = res.get("meta", {}) if seeds is not None else {}
    billed = round(u["prompt_tokens"] / 1e6 * pi
                   + u["completion_tokens"] / 1e6 * po, 5)
    res["meta"] = {
        "served_name": model, "endpoint": R.BASE_URL, "base_url": R.BASE_URL,
        "world_id": f"world_eden_{level}", "world_version": f"world_eden_{level}",
        "eden_level": level, "eden_arm": arm,
        "phrasing": "p1", "step_schedule": "v1", "narrate": False,
        "runs": m_, "steps": 30, "seed0": s_,
        "stage": "confirm" if confirm else "screen",
        "round7_pin": R.PINNED_ROUND7_HASH,
        "usage": u,
        "price_per_m": {"prompt": pi, "completion": po},
        # **The cell's cost is the SUM across attempts, not this invocation's.**
        # Round 3's top-up printed $15.02 for a sweep that billed $18.70, because
        # a re-run cell's discarded attempt was still charged. A gap-filled cell
        # carries both figures so neither reading is available by accident.
        "billed_usd": round(prior.get("billed_usd", 0.0) + billed, 5),
        "billed_this_attempt_usd": billed,
        "wall_s": round(prior.get("wall_s", 0) + (time.time() - t0)),
        # The assembly is recorded, not smoothed over.
        "filled_seeds": (prior.get("filled_seeds") or []) + (filled or []),
        "attempts": prior.get("attempts", 1) + (1 if seeds is not None else 0),
    }
    return res


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    R.assert_pinned()

    confirm_levels = ()
    if "--confirm" in sys.argv:
        confirm_levels = tuple(a for a in sys.argv[sys.argv.index("--confirm") + 1:]
                               if not a.startswith("-"))
        bad = [lv for lv in confirm_levels if lv not in R.LEVELS]
        if not confirm_levels or bad:
            raise SystemExit(
                f"--confirm needs one or more of {R.LEVELS}; got "
                f"{confirm_levels or '(none)'}")
    confirm = bool(confirm_levels)
    grid = R.cells(confirm=confirm_levels)
    OUT.mkdir(exist_ok=True)

    want = R.CONFIRM_EPISODES if confirm else R.EPISODES_PER_CELL
    todo, gaps = [], {}
    for c in grid:
        p_ = cell_path(*c, confirm=confirm)
        if not p_.exists():
            todo.append(c)
            continue
        try:
            pin = json.loads(p_.read_text())["meta"].get("round7_pin")
        except Exception:
            pin = None
        if pin != R.PINNED_ROUND7_HASH:
            raise SystemExit(
                f"{p_} exists but carries pin {pin!r}, not "
                f"{R.PINNED_ROUND7_HASH}. That is a cell from a different "
                "freeze; refusing to treat it as done. Move it aside "
                "deliberately.")
        # **A PARTIAL cell is not a completed cell.** An HTTP 402 mid-sweep once
        # left cells with 12 and 0 surviving episodes carrying the right pin;
        # keyed on pin alone they would have been skipped forever and the read
        # would have quoted a rate over a denominator of 12 beside rates over 24.
        n_ok = len([r for r in json.loads(p_.read_text()).get("runs", [])
                    if r.get("commands")])
        if n_ok < want:
            # **Fill the gap; do not re-draw the cell.** The 23 surviving
            # episodes are good, and re-running them is how one missing episode
            # cost $3.89 across three attempts in the screen.
            gaps[c] = missing_seeds(p_, R.CONFIRM_SEED0 if confirm else R.SEED0,
                                    want)
            print(f"  {p_.name}: {n_ok}/{want} — filling seeds {gaps[c]}")
            todo.append(c)

    stage = f"CONFIRM {' '.join(confirm_levels)}" if confirm else "SCREEN"
    print(f"round-7 {stage} — {len(grid)} cells, {len(todo)} to run, "
          f"{len(grid) - len(todo)} already on disk")
    print(f"pin {R.PINNED_ROUND7_HASH[:16]}…   m={want}  "
          f"seed0={R.CONFIRM_SEED0 if confirm else R.SEED0}\n")

    spent = 0.0
    for i, (model, arm, level) in enumerate(todo, 1):
        p = cell_path(model, arm, level, confirm=confirm)
        try:
            res = run_cell(model, arm, level, confirm=confirm,
                           seeds=gaps.get((model, arm, level)))
        except Exception as e:
            # A dead cell must not take the sweep with it. The gap is visible
            # because the read counts cells, not files.
            print(f"[{i}/{len(todo)}] {model} {arm} {level}  CELL FAILED: "
                  f"{str(e)[:160]}", flush=True)
            continue
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += res["meta"]["billed_usd"]
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"[{i}/{len(todo)}] {model.split('/')[-1]:<28} {arm} {level:<4} "
              f"n={n}/{want}  ${res['meta']['billed_usd']:.3f}  "
              f"{res['meta']['wall_s']}s   running ${spent:.2f}", flush=True)

    print(f"\ndone — ${spent:.2f} billed this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
