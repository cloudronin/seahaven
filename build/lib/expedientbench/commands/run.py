"""`expdx run` — measure a NEW model with the same instrument.

`replicate` re-serves our cells and judges against bands. `run` measures a model
we never ran: any world, any arm, same pins, same preconditions, same reads. **No
bands exist for a model we never measured**, so the output is that model's own
table, not a PASS/FAIL — inventing a band would be inventing a comparison.

This is what makes the benchmark usable on fine-tuned and self-hosted models:
point it at any OpenAI-compatible endpoint. For self-hosters, note that your own
vLLM with fixed batching is the only serving mode in which the between-occasion
component can actually be pinned — the hosted-API occasion caveat is a property
of hosted serving, not of the instrument.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from seahaven.eden._shared import corpus as C
from ..backends import Backend, resolve
from ..backends.base import COHORT_TEMPERATURE


def _dry_run(be, level, arm, seed0) -> int:
    """Assemble and print the first request WITHOUT serving it.

    Reuses the payload-diff machinery from the serving diagnostic: a recording
    stand-in replaces the backend, so the assembled conversation is exactly what
    a real run would send.
    """
    from seahaven.fidelity.runner import run_fidelity

    seen = []

    class _Recorder:
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

        def chat(self, messages, **kw):
            return "ready"

        def reply(self, messages, *, step, seed):
            # **COPY, do not alias.** `_rollout` appends to this same list every
            # turn, so storing the reference would make every recorded request
            # show the FINAL conversation. The first version of this printed
            # "first request, 49 messages" for a request that had 2.
            seen.append((seed, step, [dict(m) for m in messages]))
            return "look"

    run_fidelity(_Recorder(), None, runs=1, steps=30, seed0=seed0,
                 world_id=f"world_eden_{level}", narrate=False,
                 eden_level=level, eden_arm=arm, terminal_at_zero=True)
    seed, step, msgs = seen[0]
    print(f"DRY RUN — nothing served, ${0:.2f} spent\n")
    print(f"  endpoint   {be.spec.base_url}")
    print(f"  model      {be.spec.model}")
    print(f"  temperature {be.spec.temperature}"
          + ("" if be.spec.temperature == COHORT_TEMPERATURE
             else f"   ** DEVIATION from the cohort's {COHORT_TEMPERATURE} **"))
    print(f"  seed       {seed} -> request seed {seed * 100_003 + step}")
    print(f"\n  first request, {len(msgs)} message(s):")
    for m in msgs:
        print(f"    [{m['role']}] {m['content'][:400]}")
    print(f"\n  ... and {len(seen)} turn(s) would follow in this episode.")
    return 0


def main(args) -> int:
    spec = resolve(args.endpoint, model=args.model, key_env=args.key_env)
    if args.dry_run:
        class _NoKey:
            pass
        be = _NoKey()
        be.spec = spec
        return _dry_run(be, args.level, args.arm, args.seed0)

    if args.budget is None:
        raise SystemExit(
            "run refuses to start without an explicit --budget. Frontier "
            "pricing is ~10x the open tier and estimates are soft.")
    be = Backend(spec)
    resolved = be.resolve_model()

    from seahaven.fidelity.runner import run_fidelity
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    spent = 0.0
    for arm, m in (("A1", args.m_a1), ("A0", args.m_a0)):
        p = out / (f"expdx_{resolved.replace('/', '__')}__{arm}__"
                   f"{args.level}.json")
        if p.exists():
            print(f"  {arm} {args.level}: exists, skipping")
            continue
        t0 = time.time()
        res = run_fidelity(be, None, runs=m, steps=30, seed0=args.seed0,
                           world_id=f"world_eden_{args.level}", narrate=False,
                           eden_level=args.level, eden_arm=arm,
                           terminal_at_zero=True)
        t1 = time.time()
        u = dict(be.usage_total)
        res["meta"] = {
            "served_name": resolved, "resolved_model_string": resolved,
            "base_url": spec.base_url, "endpoint": spec.base_url,
            "eden_level": args.level, "eden_arm": arm,
            "world_id": f"world_eden_{args.level}",
            "phrasing": "p1", "step_schedule": "v1", "narrate": False,
            "runs": m, "steps": 30, "seed0": args.seed0,
            "terminal_at_zero": True, "stage": "expdx_run",
            "temperature": spec.temperature,
            "cohort_temperature": COHORT_TEMPERATURE,
            "temperature_deviation": (
                None if spec.temperature == COHORT_TEMPERATURE
                else f"served at {spec.temperature}, cohort is "
                     f"{COHORT_TEMPERATURE}"),
            "wall_start_epoch": round(t0), "wall_end_epoch": round(t1),
            "wall_s": round(t1 - t0), "usage": u,
        }
        p.write_text(json.dumps(res, indent=2) + "\n")
        n = len(C.episodes(res))
        print(f"  {arm} {args.level:<5} n={n}/{m}  {round(t1-t0)}s   "
              f"tokens {u['prompt_tokens']}p/{u['completion_tokens']}c")
    print(f"\n  wrote to {out}/  — read with: expdx read --results {out}")
    print("  NO PASS/FAIL: no band exists for a model that was never measured.")
    return 0
