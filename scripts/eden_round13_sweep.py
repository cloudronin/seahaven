"""ROUND 13 — GPT-5.6 Terra on LAT/W2/W3. One session, running spend printed."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round13 as R  # noqa: E402
from seahaven.fidelity.endpoint import Endpoint  # noqa: E402
from seahaven.fidelity.runner import run_fidelity  # noqa: E402
from seahaven.eden.outcome import EDEN_MAX_TOKENS  # noqa: E402

OUT = Path("results")


class TerraPolicy:
    """`EdenPolicy` with the one value the model will not accept changed.

    **Everything else is verbatim**: same `max_tokens`, same seed derivation
    `seed * 100_003 + step`, and message assembly is upstream in `_rollout` so it
    cannot differ. Only `temperature` moves, from 0.9 to 1.0, because
    gpt-5.6-terra rejects 0.9 outright.

    A wrapper rather than an edit to `outcome.EdenPolicy`, because that module is
    hashed by the round-11, round-12 AND round-13 pins — changing it to serve one
    model would break three freezes to avoid one subclass.
    """

    def __init__(self, ep, temperature):
        self.ep = ep
        self.temperature = temperature
        self.name = getattr(ep, "served_name", "terra")

    @property
    def usage_total(self):
        return self.ep.usage_total

    def chat(self, messages, **kw):
        kw.setdefault("temperature", self.temperature)
        return self.ep.chat(messages, **kw)

    def reply(self, messages, *, step, seed):
        return self.ep.chat(messages, max_tokens=EDEN_MAX_TOKENS,
                            temperature=self.temperature,
                            seed=seed * 100_003 + step)


def cell_path(arm, level):
    return OUT / f"eden_e13_{R.SERVED_NAME.replace('/', '__')}__{arm}__{level}.json"


def resolve_model(key: str) -> str:
    """Resolve at run time and assert non-empty, per the spec. The string is the
    only pin available — OpenAI ships mid-cycle behaviour updates to a serving
    name, and no dated snapshot for this model is exposed."""
    req = urllib.request.Request(f"{R.BASE_URL}/models",
                                 headers={"Authorization": f"Bearer {key}"})
    ids = {m["id"] for m in json.loads(
        urllib.request.urlopen(req, timeout=60).read()).get("data", [])}
    if R.SERVED_NAME not in ids:
        raise SystemExit(f"{R.SERVED_NAME} is not on the models list")
    return R.SERVED_NAME


def assert_seeds_free():
    want = set(range(R.SEED0, R.SEED0 + R.EPISODES_A1))
    for f in glob.glob("results/eden_e*.json"):
        d = json.loads(Path(f).read_text())
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        if want & used:
            raise SystemExit(f"SEED COLLISION with {Path(f).name}")


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set")
        return 2
    R.assert_pinned()
    assert_seeds_free()
    resolved = resolve_model(key)
    OUT.mkdir(exist_ok=True)

    print(f"ROUND 13 — {resolved}, pin {R.PINNED_ROUND13_HASH[:16]}…")
    print(f"  PREDICTION (pinned): {R.PREDICTION}")
    print(f"  DEVIATION (pinned): temperature {R.TEMPERATURE} vs cohort "
          f"{R.COHORT_TEMPERATURE} — model constraint, not a chosen knob")
    print(f"  A1 m={R.EPISODES_A1} A0 m={R.EPISODES_A0}  seed0={R.SEED0}  "
          f"terminal_at_zero={R.TERMINAL_AT_ZERO}\n")

    spent = 0.0
    for _sn, arm, level in R.cells():
        p = cell_path(arm, level)
        if p.exists():
            print(f"  {arm} {level}: exists, skipping")
            continue
        ep = Endpoint(base_url=R.BASE_URL, served_name=resolved,
                      api_key=key, timeout=300)
        pol = TerraPolicy(ep, R.TEMPERATURE)
        t0 = time.time()
        res = run_fidelity(pol, None, runs=R.episodes_for(arm), steps=30,
                           seed0=R.SEED0, world_id=f"world_eden_{level}",
                           narrate=False, eden_level=level, eden_arm=arm,
                           terminal_at_zero=R.TERMINAL_AT_ZERO)
        t1 = time.time()
        u = dict(ep.usage_total)
        pi, po = R.PRICE
        billed = round(u["prompt_tokens"] / 1e6 * pi
                       + u["completion_tokens"] / 1e6 * po, 5)
        res["meta"] = {
            "served_name": resolved, "endpoint": R.BASE_URL,
            "base_url": R.BASE_URL, "resolved_model_string": resolved,
            "api_reported_model": resolved, "system_fingerprint": None,
            "world_id": f"world_eden_{level}",
            "world_version": f"world_eden_{level}",
            "eden_level": level, "eden_arm": arm,
            "phrasing": "p1", "step_schedule": "v1", "narrate": False,
            "runs": R.episodes_for(arm), "steps": 30, "seed0": R.SEED0,
            "stage": "r13_frontier",
            "terminal_at_zero": R.TERMINAL_AT_ZERO,
            "round13_pin": R.PINNED_ROUND13_HASH,
            "temperature": R.TEMPERATURE,
            "cohort_temperature": R.COHORT_TEMPERATURE,
            "temperature_deviation": R.TEMPERATURE_DEVIATION,
            "reasoning_effort": "default (not set, not tuned)",
            "wall_start_epoch": round(t0), "wall_end_epoch": round(t1),
            "usage": u, "price_per_m": {"prompt": pi, "completion": po},
            "billed_usd": billed, "wall_s": round(t1 - t0),
        }
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += billed
        n = len([r for r in res.get("runs", []) if r.get("commands")])
        print(f"  {arm} {level:<4} n={n}/{R.episodes_for(arm)}  ${billed:.3f}  "
              f"{round(t1-t0)}s   RUNNING ${spent:.2f}", flush=True)
    print(f"\ndone — ${spent:.2f} billed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
