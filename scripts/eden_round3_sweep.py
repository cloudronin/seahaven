"""Run the round-3 bracket grid. `--gate` first, the rest only after it is read.

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

from seahaven.eden import round3 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402

OUT = Path("results")


def cell_path(model: str, arm: str, level: str, rate: bool = False,
              topup: bool = False) -> Path:
    # The rate stage writes its OWN file: it is 72 additional episodes at offset
    # seeds, not a replacement for stage 1's 24, and overwriting would discard
    # data rather than extend it.
    tag = f"{level}topup" if topup else (f"{level}rate" if rate else level)
    return OUT / f"eden_e3_{model.replace('/', '__')}__{arm}__{tag}.json"


def run_cell(model: str, arm: str, level: str, rate: bool = False,
             topup: bool = False) -> dict:
    pi, po = R.COHORT[model]
    ep = Endpoint(base_url=R.BASE_URL, served_name=model,
                  api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
    t0 = time.time()
    m_ = R.TOPUP_EPISODES if topup else (R.RATE_EPISODES if rate
                                        else R.EPISODES_PER_CELL)
    s_ = R.TOPUP_SEED0 if topup else (R.RATE_SEED0 if rate else R.SEED0)
    if topup:
        # **A seed collision would silently double-count identical episodes.**
        # The committed cell holds indices 0-23 (seeds 5150-5173); these are
        # 24-95 (5174-5245). Checked against what is actually ON DISK rather
        # than against the constants, because the constants are the thing that
        # would be wrong.
        prior = cell_path(model, arm, level)
        if prior.exists():
            have = {r["seed"] for r in json.loads(prior.read_text()).get("runs", [])
                    if "seed" in r}
            clash = have & set(range(s_, s_ + m_))
            if clash:
                raise SystemExit(
                    f"{model} {level}: seed collision with the committed cell — "
                    f"{len(clash)} overlapping (e.g. {sorted(clash)[:5]}). "
                    "Pooling these would double-count identical episodes.")
    res = run_fidelity(ep, None, runs=m_, steps=30,
                       seed0=s_, world_id=f"world_eden_{level}",
                       narrate=False, eden_level=level, eden_arm=arm)
    u = dict(ep.usage_total)
    # `run_fidelity` returns no `meta`; the CLI builds it. Assembling it here
    # rather than shelling out to the CLI keeps the endpoint object -- and
    # therefore `usage_total`, which is the only measured cost -- reachable.
    res["meta"] = {
        "served_name": model, "endpoint": R.BASE_URL, "base_url": R.BASE_URL,
        "world_id": f"world_eden_{level}", "world_version": f"world_eden_{level}",
        "eden_level": level, "eden_arm": arm,
        "phrasing": "p1", "step_schedule": "v1", "narrate": False,
        "runs": m_, "steps": 30, "seed0": s_, "stage": "topup" if topup else ("rate" if rate else "main"),
        "round3_pin": R.PINNED_ROUND3_HASH,
        "usage": u,
        "price_per_m": {"prompt": pi, "completion": po},
        "billed_usd": round(u["prompt_tokens"] / 1e6 * pi
                            + u["completion_tokens"] / 1e6 * po, 5),
        "wall_s": round(time.time() - t0),
    }
    return res


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    R.assert_pinned()
    rate = "--rate" in sys.argv
    topup = "--topup" in sys.argv
    grid = R.cells(rate=rate and not topup, topup=topup)
    OUT.mkdir(exist_ok=True)

    # **A pre-existing file is only a completed cell if it SAYS it is.** Round 1
    # writes `eden_e3_<level>.json` -- the E-axis level 2 -- into the same
    # directory, so the `eden_e3_` prefix is already taken. No collision occurs
    # today because round-2 names carry the model and `__` separators, but the
    # resume check keys on path existence, so a colliding name would make the
    # sweep SKIP a cell believing it was done and leave a hole no count would
    # show. Verify the pin instead of trusting the path.
    todo = []
    for c in grid:
        p_ = cell_path(*c, rate=rate, topup=topup)
        if not p_.exists():
            todo.append(c)
            continue
        try:
            pin = json.loads(p_.read_text())["meta"].get("round3_pin")
        except Exception:
            pin = None
        if pin != R.PINNED_ROUND3_HASH:
            raise SystemExit(
                f"{p_} exists but carries pin {pin!r}, not "
                f"{R.PINNED_ROUND3_HASH}. That is either a foreign file at a "
                "round-2 cell path or a cell from a different freeze; refusing "
                "to treat it as done. Move it aside deliberately.")
        # **A PARTIAL cell is not a completed cell.** When the account hit HTTP
        # 402 mid-sweep, two cells landed with 12 and 0 surviving episodes and
        # the right pin. Keyed on pin alone they would have been skipped forever,
        # and the read would have quoted a rate over a denominator of 12 beside
        # rates over 24 -- exactly the "lost episodes are counted, never
        # absorbed" failure, moved from the read into the scheduler where the
        # read cannot see it.
        n_ok = len([r for r in json.loads(p_.read_text()).get("runs", [])
                    if r.get("commands")])
        want = (R.TOPUP_EPISODES if topup else
                (R.RATE_EPISODES if rate else R.EPISODES_PER_CELL))
        if n_ok < want:
            print(f"  re-running {p_.name}: {n_ok}/{want} "
                  f"episodes survived")
            todo.append(c)
    print(f"round-3 LATENCY — {len(grid)} cells, "
          f"{len(todo)} to run, {len(grid) - len(todo)} already on disk")
    print(f"pin {R.PINNED_ROUND3_HASH[:16]}…   m={R.EPISODES_PER_CELL}  "
          f"seed0={R.SEED0}\n")

    spent = 0.0
    for i, (model, arm, level) in enumerate(todo, 1):
        p = cell_path(model, arm, level, rate=rate, topup=topup)
        try:
            res = run_cell(model, arm, level, rate=rate, topup=topup)
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
              f"n={n}/{want if False else (R.TOPUP_EPISODES if topup else (R.RATE_EPISODES if rate else R.EPISODES_PER_CELL))}  ${res['meta']['billed_usd']:.3f}  "
              f"{res['meta']['wall_s']}s   running ${spent:.2f}", flush=True)

    print(f"\ndone — ${spent:.2f} billed this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
