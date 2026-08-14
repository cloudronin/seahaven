"""ROUND 11 stage 4 — the eight-model cohort on W2 and W3, generation 3.

Block construction per cell, `terminal_at_zero=True` recorded in every meta.
Gap-fill by seed, never a redraw: the surviving episodes of a partial cell are
good and re-running them is what cost $3.89 in round 6.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round11 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402

OUT = Path("results")


def cell_path(model: str, arm: str, level: str) -> Path:
    return OUT / f"eden_e11_{model.replace('/', '__')}__{arm}__{level}.json"


def missing_seeds(p: Path, s0: int, m: int) -> list[int]:
    d = json.loads(p.read_text())
    have = {r["seed"] for r in d.get("runs", []) if r.get("commands") and "seed" in r}
    return sorted(set(range(s0, s0 + m)) - have)


def run_cell(model, arm, level, seeds=None):
    pi, po = R.COHORT[model]
    ep = Endpoint(base_url=R.BASE_URL, served_name=model,
                  api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
    t0 = time.time()
    m_ = R.episodes_for(arm)

    def _one(runs, seed0):
        return run_fidelity(ep, None, runs=runs, steps=30, seed0=seed0,
                            world_id=f"world_eden_{level}", narrate=False,
                            eden_level=level, eden_arm=arm,
                            terminal_at_zero=R.TERMINAL_AT_ZERO)

    if seeds is None:
        res, filled = _one(m_, R.SEED0), None
    else:
        p = cell_path(model, arm, level)
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
                        f"{len(r['commands'])} steps vs {sorted(want_steps)}. "
                        "The schedule is not flat; refusing.")
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
        "runs": m_, "steps": 30, "seed0": R.SEED0, "stage": "r11_cross_world",
        "terminal_at_zero": R.TERMINAL_AT_ZERO,
        "round11_pin": R.PINNED_ROUND11_HASH,
        "usage": u, "price_per_m": {"prompt": pi, "completion": po},
        "billed_usd": round(prior.get("billed_usd", 0.0) + billed, 5),
        "billed_this_attempt_usd": billed,
        "wall_s": round(prior.get("wall_s", 0) + (time.time() - t0)),
        "filled_seeds": (prior.get("filled_seeds") or []) + (filled or []),
        "attempts": prior.get("attempts", 1) + (1 if seeds is not None else 0),
    }
    return res


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    R.assert_pinned()
    OUT.mkdir(exist_ok=True)
    grid = R.cells()
    todo, gaps = [], {}
    for c in grid:
        want = R.episodes_for(c[1])
        p_ = cell_path(*c)
        if not p_.exists():
            todo.append(c)
            continue
        d = json.loads(p_.read_text())
        if d.get("meta", {}).get("round11_pin") != R.PINNED_ROUND11_HASH:
            raise SystemExit(f"{p_} carries a different pin; move it aside.")
        n_ok = len([r for r in d.get("runs", []) if r.get("commands")])
        if n_ok < want:
            gaps[c] = missing_seeds(p_, R.SEED0, want)
            print(f"  {p_.name}: {n_ok}/{want} — filling {len(gaps[c])} seeds")
            todo.append(c)
    print(f"round-11 W2/W3 — {len(grid)} cells, {len(todo)} to run")
    print(f"pin {R.PINNED_ROUND11_HASH[:16]}…  A1 m={R.EPISODES_A1} "
          f"A0 m={R.EPISODES_A0}  seed0={R.SEED0}  "
          f"terminal_at_zero={R.TERMINAL_AT_ZERO}\n")
    spent = 0.0
    for i, (model, arm, level) in enumerate(todo, 1):
        p = cell_path(model, arm, level)
        try:
            res = run_cell(model, arm, level, seeds=gaps.get((model, arm, level)))
        except Exception as e:
            print(f"[{i}/{len(todo)}] {model} {arm} {level}  FAILED: "
                  f"{str(e)[:150]}", flush=True)
            continue
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += res["meta"]["billed_this_attempt_usd"]
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"[{i}/{len(todo)}] {model.split('/')[-1][:26]:<28}{arm} {level}  "
              f"n={n}/{res['meta']['runs']}  ${res['meta']['billed_usd']:.3f}  "
              f"{res['meta']['wall_s']}s   running ${spent:.2f}", flush=True)
    print(f"\ndone — ${spent:.2f} billed this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
