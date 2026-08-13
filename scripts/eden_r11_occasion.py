"""ROUND 11 DIAGNOSTIC step 3 — is occasion variance specific to DS-V4-Flash?

Steps 1 and 2 came back clean:

  1. Request construction is BYTE-IDENTICAL between the round-10 sweep path and
     the top-up path, verified by building both offline with a recording policy
     rather than by reading the scripts. Same `EdenPolicy` (max_tokens=2048,
     temperature=0.9, seed*100003+step), same warmup, and the history carries the
     PARSED COMMAND on both paths -- the `eden is not None` branch in `_rollout`.
  2. No provider-side signal. `reasoning_tokens` is 0 in both halves for both
     models; the prompt-token ratio of 1.225 is predicted at 1.261 from episode
     length alone (the conversation is resent whole each step, so tokens scale
     ~n^2) and is therefore DOWNSTREAM of the rate change; and the six long
     replies in the original block would be expected 2.25 times in 752 later
     calls, so seeing zero has P=0.106 and is not significant. Excluding that
     tail the mean reply length is 10.68 vs 10.96 and the share of replies over
     30 characters is 3.73% vs 3.72%.

So the reply SHAPE is stable and only the decision changed. This step asks
whether any other model does the same thing.

**Run through the round-10 BLOCK construction** -- `run_fidelity(runs=24,
seed0=...)`, one call, exactly as the sweep does -- not the top-up's repeated
single-episode fills. Step 1 showed the two are equivalent, so this is belt and
braces.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round10 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402

OUT = Path("results")
LEVEL, ARM = "LAT", "A1"
SEED0 = 15200
EPISODES = 24

#: Prices taken from these models' own prior cells, so the billing figure is
#: comparable with every other round.
PROBE = {
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
    "google/gemma-4-31B-it": (0.39, 0.97),
}

#: **THE PRE-COMMITTED READING, fixed before a single episode is served.**
#: Chosen because these two bracket the cohort: cogito is the high pole and the
#: model DS-V4-Flash was tied with, gemma is a floor member whose baseline is 0/96
#: and therefore maximally sensitive to any upward move.
READING = {
    ("no", "no"): "DS-V4-Flash specific, or a one-off event.",
    ("yes", "no"): "Occasion variance concentrated at the HIGH end.",
    ("no", "yes"): "Occasion variance at the FLOOR — the most alarming, since "
                   "the floor is what round 10's surviving claim rests on.",
    ("yes", "yes"): "EVERY rate in this program is an occasion estimate.",
}

#: MDS against each model's existing generation-3 LAT cell, at 24 against 96.
#: Printed beside every verdict; a null means "no shift larger than this",
#: never "stable". gemma's downward MDS is undefined because it sits at zero.
MDS_NOTE = {
    "deepcogito/cogito-v2-1-671b": "baseline 36/96=0.375, detectable +/-0.250",
    "google/gemma-4-31B-it": "baseline 0/96=0.000, detectable +0.083 (no drop "
                             "possible from the floor)",
}


def cell_path(model: str) -> Path:
    return OUT / f"eden_e11occ_{model.replace('/', '__')}__{ARM}__{LEVEL}.json"


def assert_seeds_free() -> None:
    want = set(range(SEED0, SEED0 + EPISODES))
    for f in glob.glob("results/eden_e*.json"):
        d = json.loads(Path(f).read_text())
        if d.get("meta", {}).get("eden_level") != LEVEL:
            continue
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        if want & used:
            raise SystemExit(
                f"SEED COLLISION with {Path(f).name}: {sorted(want & used)[:6]}")


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    R.assert_pinned()
    assert_seeds_free()
    OUT.mkdir(exist_ok=True)

    print(f"ROUND-11 OCCASION PROBE — {EPISODES} episodes each, "
          f"seeds {SEED0}-{SEED0 + EPISODES - 1}")
    print(f"  pin {R.PINNED_ROUND10_HASH[:16]}…  terminal_at_zero={R.TERMINAL_AT_ZERO}")
    print("  PRE-COMMITTED READING:")
    for (a, b), v in READING.items():
        print(f"    cogito moves={a:<4} gemma moves={b:<4} -> {v}")
    print()

    spent = 0.0
    for model, (pi, po) in PROBE.items():
        p = cell_path(model)
        if p.exists():
            print(f"  {model}: cell exists, skipping")
            continue
        ep = Endpoint(base_url=R.BASE_URL, served_name=model,
                      api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
        t0 = time.time()
        try:
            # The BLOCK construction, exactly as the round-10 sweep issues it.
            res = run_fidelity(ep, None, runs=EPISODES, steps=30, seed0=SEED0,
                               world_id=f"world_eden_{LEVEL}", narrate=False,
                               eden_level=LEVEL, eden_arm=ARM,
                               terminal_at_zero=R.TERMINAL_AT_ZERO)
        except Exception as e:
            print(f"  {model}: CELL FAILED — {str(e)[:160]}", flush=True)
            continue
        u = dict(ep.usage_total)
        billed = round(u["prompt_tokens"] / 1e6 * pi
                       + u["completion_tokens"] / 1e6 * po, 5)
        res["meta"] = {
            "served_name": model, "endpoint": R.BASE_URL, "base_url": R.BASE_URL,
            "world_id": f"world_eden_{LEVEL}", "world_version": f"world_eden_{LEVEL}",
            "eden_level": LEVEL, "eden_arm": ARM,
            "phrasing": "p1", "step_schedule": "v1", "narrate": False,
            "runs": EPISODES, "steps": 30, "seed0": SEED0,
            "stage": "occasion_probe",
            "terminal_at_zero": R.TERMINAL_AT_ZERO,
            "round10_pin": R.PINNED_ROUND10_HASH,
            # **NOT a round-10 measurement cell.** The round-10 read globs
            # `eden_e10_*`, so this filename keeps it out of that table by
            # construction rather than by a filter someone must remember.
            "occasion_probe": True,
            "usage": u, "price_per_m": {"prompt": pi, "completion": po},
            "billed_usd": billed, "wall_s": round(time.time() - t0),
        }
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += billed
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"  {model.split('/')[-1]:<28} n={n}/{EPISODES}  ${billed:.3f}  "
              f"{res['meta']['wall_s']}s   running ${spent:.2f}", flush=True)

    print(f"\ndone — ${spent:.2f} billed. "
          "Read with scripts/eden_r11_occasion_read.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
