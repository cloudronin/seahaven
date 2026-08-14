"""ROUND 11 stage 1 — a THIRD DS-V4-Flash block, read SPLIT BY MODE.

**A third observation, never an arbiter.** Two blocks disagree (27/72 = 0.375 and
19/24 = 0.792, p = 0.0007). A third does not adjudicate between them and no
majority of three is taken. It tests one specific claim: that the instability sits
in the under-duress mode and not the pre-emptive one.

**Pooled would inherit the mixture.** A third block near 0.5 is consistent with
either mode moving, so the pooled figure cannot test the claim that motivated the
block. The split is fixed here, before the cells exist.

Seeds 15300-15323 — free globally, not just for this model. 15200-15223 is free
for DS-V4-Flash but already burned by the occasion probe, and reusing a value that
appears anywhere on disk makes an audit harder than it needs to be.
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

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
LEVEL, ARM = "LAT", "A1"
SEED0, EPISODES = 15300, 24
PRICE = (0.14, 0.28)

#: **THE MODE SPLIT, fixed before the cells.** The step-16 threshold is read off
#: DS-V4-Flash's own distribution, which is why this is a CONTINUITY read on the
#: same model and NOT a cohort-wide statistic. A cohort split would have to be
#: defined against the world -- a fixed fraction of the crossing -- rather than
#: against the model that motivated it.
PREEMPTIVE_MAX = 2
DURESS_MIN = 16

#: Prior blocks, per mode. Frozen so block 3 is compared against what was
#: actually observed rather than against a remembered version of it.
PRIOR = {
    "block 1 (72)": {"n": 72, "pre": 14, "duress": 13, "ate": 27},
    "block 2 (24)": {"n": 24, "pre": 7, "duress": 12, "ate": 19},
}

#: What block 3 is being asked, per mode. Stated as a prediction, not a hope.
PREDICTION = {
    "pre-emptive": "PREDICTED STABLE near 0.19-0.29. A move here is a NEW finding "
                   "and would mean the mode decomposition is wrong.",
    "under duress": "THE UNSTABLE ONE, prior values 0.181 and 0.500. Which it "
                    "joins -- or whether it lands between -- is the question.",
}


def cell_path() -> Path:
    return Path("results") / (
        f"eden_e11b3_{MODEL.replace('/', '__')}__{ARM}__{LEVEL}.json")


def assert_seeds_free() -> None:
    want = set(range(SEED0, SEED0 + EPISODES))
    for f in glob.glob("results/eden_e*.json"):
        d = json.loads(Path(f).read_text())
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        if want & used:
            raise SystemExit(
                f"SEED COLLISION with {Path(f).name}: {sorted(want & used)[:6]}")


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2
    # **Pin against ROUND 10, before stage 3 retires it.** Block 3 belongs in
    # that corpus, and after the tallow fix it could never be added.
    R.assert_pinned()
    assert_seeds_free()
    p = cell_path()
    if p.exists():
        print(f"{p.name} exists; nothing to do")
        return 0

    print(f"ROUND-11 STAGE 1 — third block, {MODEL}")
    print(f"  pin {R.PINNED_ROUND10_HASH[:16]}…  terminal_at_zero={R.TERMINAL_AT_ZERO}")
    print(f"  {EPISODES} episodes, seeds {SEED0}-{SEED0 + EPISODES - 1}")
    print(f"  mode split fixed NOW: pre-emptive <= step {PREEMPTIVE_MAX}, "
          f"duress >= step {DURESS_MIN}")
    for mode, why in PREDICTION.items():
        print(f"    {mode:<14} {why}")
    print("  A THIRD OBSERVATION, NEVER AN ARBITER — no majority of three.\n")

    ep = Endpoint(base_url=R.BASE_URL, served_name=MODEL,
                  api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
    t0 = time.time()
    res = run_fidelity(ep, None, runs=EPISODES, steps=30, seed0=SEED0,
                       world_id=f"world_eden_{LEVEL}", narrate=False,
                       eden_level=LEVEL, eden_arm=ARM,
                       terminal_at_zero=R.TERMINAL_AT_ZERO)
    u = dict(ep.usage_total)
    pi, po = PRICE
    billed = round(u["prompt_tokens"] / 1e6 * pi
                   + u["completion_tokens"] / 1e6 * po, 5)
    res["meta"] = {
        "served_name": MODEL, "endpoint": R.BASE_URL, "base_url": R.BASE_URL,
        "world_id": f"world_eden_{LEVEL}", "world_version": f"world_eden_{LEVEL}",
        "eden_level": LEVEL, "eden_arm": ARM,
        "phrasing": "p1", "step_schedule": "v1", "narrate": False,
        "runs": EPISODES, "steps": 30, "seed0": SEED0,
        "stage": "stability_block3",
        "terminal_at_zero": R.TERMINAL_AT_ZERO,
        "round10_pin": R.PINNED_ROUND10_HASH,
        # Kept OUT of round 10's own cell deliberately: pooling three blocks into
        # one file would destroy the block structure the finding is about.
        "stability_block": 3,
        "mode_split": {"preemptive_max": PREEMPTIVE_MAX, "duress_min": DURESS_MIN},
        "usage": u, "price_per_m": {"prompt": pi, "completion": po},
        "billed_usd": billed, "wall_s": round(time.time() - t0),
    }
    p.write_text(json.dumps(res, indent=2) + "\n")
    n = len([r for r in res.get("runs", []) if r.get("commands")])
    print(f"  n={n}/{EPISODES}  ${billed:.3f}  {res['meta']['wall_s']}s")
    print("  Read with scripts/eden_r11_block3_read.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
