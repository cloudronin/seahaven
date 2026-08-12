"""Run the round-2 grid. `--gate0` first, the rest only after it is read.

**Asserts the pin before spending anything.** A constant edited after the numbers
came in is invisible to every other check, because the numbers would still
recompute from the edited constant.

**Resumable, and that is a correctness property rather than a convenience.** A
cell that already exists on disk is skipped, so a run interrupted at cell 60 of
84 does not re-bill the first 60 -- and, more importantly, does not silently
re-run them under a different provider-side model version and mix two vintages
inside one figure.

Cells run one at a time; episodes inside a cell run `SEAHAVEN_CONCURRENCY` wide.
Serial at the cell level is deliberate: the provider throttles per account, and a
429 storm across six models would be indistinguishable from six slow models.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round2 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402

OUT = Path("results")


def cell_path(model: str, arm: str, level: str) -> Path:
    return OUT / f"eden_e2_{model.replace('/', '__')}__{arm}__{level}.json"


def run_cell(model: str, arm: str, level: str) -> dict:
    pi, po = R.COHORT[model]
    ep = Endpoint(base_url=R.BASE_URL, served_name=model,
                  api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
    t0 = time.time()
    res = run_fidelity(ep, None, runs=R.EPISODES_PER_CELL, steps=30,
                       seed0=R.SEED0, world_id=f"world_eden_{level}",
                       narrate=False, eden_level=level, eden_arm=arm)
    u = dict(ep.usage_total)
    res["meta"]["eden_level"] = level
    res["meta"]["eden_arm"] = arm
    res["meta"]["served_name"] = model
    res["meta"]["base_url"] = R.BASE_URL
    res["meta"]["round2_pin"] = R.PINNED_ROUND2_HASH
    res["meta"]["usage"] = u
    res["meta"]["price_per_m"] = {"prompt": pi, "completion": po}
    res["meta"]["billed_usd"] = round(
        u["prompt_tokens"] / 1e6 * pi + u["completion_tokens"] / 1e6 * po, 5)
    res["meta"]["wall_s"] = round(time.time() - t0)
    return res


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    R.assert_pinned()
    gate0 = "--gate0" in sys.argv
    grid = R.cells(gate0=gate0)
    OUT.mkdir(exist_ok=True)

    todo = [c for c in grid if not cell_path(*c).exists()]
    print(f"round-2 {'GATE 0' if gate0 else 'FULL'} — {len(grid)} cells, "
          f"{len(todo)} to run, {len(grid) - len(todo)} already on disk")
    print(f"pin {R.PINNED_ROUND2_HASH[:16]}…   m={R.EPISODES_PER_CELL}  "
          f"seed0={R.SEED0}\n")

    spent = 0.0
    for i, (model, arm, level) in enumerate(todo, 1):
        p = cell_path(model, arm, level)
        try:
            res = run_cell(model, arm, level)
        except Exception as e:
            # A dead cell must not take the sweep with it: the remaining models
            # are independent and their cells are still worth having. The gap is
            # visible because the read counts cells, not files.
            print(f"[{i}/{len(todo)}] {model} {arm} {level}  CELL FAILED: "
                  f"{str(e)[:120]}", flush=True)
            continue
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += res["meta"]["billed_usd"]
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"[{i}/{len(todo)}] {model.split('/')[-1]:<28} {arm} {level:<5} "
              f"n={n}/{R.EPISODES_PER_CELL}  ${res['meta']['billed_usd']:.3f}  "
              f"{res['meta']['wall_s']}s   running ${spent:.2f}", flush=True)

    print(f"\ndone — ${spent:.2f} billed this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
