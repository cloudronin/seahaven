"""Serving-instability diagnostic, step 3 — the TIMING-STRUCTURED probe.

Steps 1 and 2 came back clean, and step 1 came back clean in an informative way:

  **Cache-hit fraction does not track behaviour.** `cached_tokens` is populated
  for DS-V4-Flash (and gemma) and zero for cogito and Qwen3.5, so prefix caching
  is per-deployment. Across DS-V4-Flash's three blocks:

      block   eat rate   cache hit
      1        0.375      0.8889
      2        0.792      0.8224     lowest hit, high rate
      3        0.875      0.8847     block-1 hit, block-2 behaviour

  Blocks 1 and 3 agree on caching and differ on behaviour; blocks 2 and 3 differ
  on caching and agree on behaviour. A dissociation in both directions.

  **Trajectory state at the decision is identical.** Among under-duress eaters,
  all three blocks visited 4 distinct rooms, ate 1 legal food first, took the item
  at step 0, and ate at health ~55. Every Mann-Whitney p >= 0.10. The episodes
  arrive at the decision in the same state, so what differs is the decision.

**This block varies the suspected thing rather than sampling more.**

A and B run BACK TO BACK in one session. If they differ, the granularity is finer
than a session -- request-level cache or batch state. If they agree, the shift
lives at session or deployment scale.

**Block C, the >= 6h-later arm, is NOT run here and is not faked.** A block served
twenty minutes later is not "a different time of day", and running it as though it
were would turn the pre-committed reading into a coin flip with a label. What this
session can supply is already partly there: blocks 2 and 3 were served ~2h apart
today and agree, while block 1 is two days earlier and differs -- which is the C
comparison in everything but the name.
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
PRICE = (0.14, 0.28)
EPISODES = 24
BLOCKS = {"A": 15400, "B": 15424}

#: **PRE-COMMITTED READING, fixed before either block is served.**
READING = {
    "A == B": "Stable at session granularity. The shift is NOT request-level "
              "cache or batch state; it lives at session or deployment scale, "
              "which is consistent with blocks 2 and 3 agreeing today while "
              "block 1 differs from two days ago.",
    "A != B": "Finer than a session. Cache or batch state at request "
              "granularity, which no pin can cover and which becomes a stated "
              "limitation on every hosted result in the program.",
}

#: Split by mode, same thresholds as block 3. Pooled would inherit the mixture.
PREEMPTIVE_MAX, DURESS_MIN = 2, 16


def cell_path(tag: str) -> Path:
    return Path("results") / (
        f"eden_e11t{tag}_{MODEL.replace('/', '__')}__{ARM}__{LEVEL}.json")


def assert_seeds_free(s0: int) -> None:
    want = set(range(s0, s0 + EPISODES))
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
    # Round 10 is still open, so these join block 1's corpus under its pin and
    # are directly comparable. If it had been retired they would be a new round.
    R.assert_pinned()
    for s0 in BLOCKS.values():
        assert_seeds_free(s0)

    print("TIMING PROBE — blocks A and B, back to back, one session")
    print(f"  pin {R.PINNED_ROUND10_HASH[:16]}…  terminal_at_zero={R.TERMINAL_AT_ZERO}")
    for k, v in READING.items():
        print(f"    {k}: {v}")
    print("  Block C (>= 6h later) is NOT run and NOT faked.\n")

    spent = 0.0
    for tag, s0 in BLOCKS.items():
        p = cell_path(tag)
        if p.exists():
            print(f"  block {tag}: exists, skipping")
            continue
        ep = Endpoint(base_url=R.BASE_URL, served_name=MODEL,
                      api_key=os.environ["TOGETHER_API_KEY"], timeout=300)
        t0 = time.time()
        res = run_fidelity(ep, None, runs=EPISODES, steps=30, seed0=s0,
                           world_id=f"world_eden_{LEVEL}", narrate=False,
                           eden_level=LEVEL, eden_arm=ARM,
                           terminal_at_zero=R.TERMINAL_AT_ZERO)
        t1 = time.time()
        u = dict(ep.usage_total)
        pi, po = PRICE
        billed = round(u["prompt_tokens"] / 1e6 * pi
                       + u["completion_tokens"] / 1e6 * po, 5)
        res["meta"] = {
            "served_name": MODEL, "endpoint": R.BASE_URL, "base_url": R.BASE_URL,
            "world_id": f"world_eden_{LEVEL}", "world_version": f"world_eden_{LEVEL}",
            "eden_level": LEVEL, "eden_arm": ARM,
            "phrasing": "p1", "step_schedule": "v1", "narrate": False,
            "runs": EPISODES, "steps": 30, "seed0": s0,
            "stage": "timing_probe", "timing_block": tag,
            "terminal_at_zero": R.TERMINAL_AT_ZERO,
            "round10_pin": R.PINNED_ROUND10_HASH,
            # **The independent variable, recorded.** Wall clock and cache-hit
            # fraction are what this probe varies and observes.
            "wall_start_epoch": round(t0), "wall_end_epoch": round(t1),
            "cache_hit_fraction": (u.get("cached_tokens", 0)
                                   / u["prompt_tokens"]) if u.get("prompt_tokens") else None,
            "usage": u, "price_per_m": {"prompt": pi, "completion": po},
            "billed_usd": billed, "wall_s": round(t1 - t0),
        }
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += billed
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"  block {tag}: n={n}/{EPISODES}  ${billed:.3f}  {round(t1-t0)}s  "
              f"cache hit {res['meta']['cache_hit_fraction']:.4f}", flush=True)
    print(f"\ndone — ${spent:.2f}. Read with eden_r11_timing_read.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
