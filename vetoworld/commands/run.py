"""`vworld run` — measure a NEW model with the same instrument.

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


def _round_cells(args, spec) -> int:
    """Serve a PINNED round's grid into the committed corpus.

    The ad-hoc path below writes a standalone table for a model nobody measured.
    This one writes a round: the pin is asserted before a request is made, the
    cell lands under the corpus naming convention, and the pin travels in every
    meta so a later sweep cannot mistake a cell from one freeze for a cell from
    another. Eleven rounds each had their own copy of this; `_shared/sweep` is
    the one it delegates to now.
    """
    import importlib

    from seahaven.eden._shared import sweep as SW
    from seahaven.fidelity.runner import run_fidelity

    R = importlib.import_module(f"seahaven.eden.round{args.round}")
    R.assert_pinned()
    tag, pin_key = str(args.round), f"round{args.round}_pin"
    pin = getattr(R, f"PINNED_ROUND{args.round}_HASH")

    # **A BLOCK IS A SEPARATE CELL, NOT A TOP-UP.** Without this the tag, the
    # seed base and the path all come from the round number alone, so a second
    # block of the same (model, arm, level) resolves to the SAME file and
    # `resume_plan` gap-fills into it — pooling the blocks and destroying the
    # structure the measurement is about. Round 11's block 3 was kept out of
    # round 10's cell by hand for exactly that reason; this is that decision
    # made expressible.
    # **The COMP gate is a stage of the round, not a separate round.** It uses
    # the same pin — a model screened under different bytes than it is measured
    # under would be screened for a different world — but its own grid, arm,
    # episode count and seed base, because COMP carries no forbidden item and
    # A1 would emit "The None is not to be eaten."
    comp = getattr(args, "stage", None) == "comp"
    block, stage = getattr(args, "block", None), None
    if comp and block:
        raise SystemExit("--stage comp and --block are different things: the "
                         "gate is served once, blocks repeat a measurement.")
    if block:
        tag = f"{args.round}b{block}"
        stage = f"r{args.round}_block{block}"
        if args.seed0 is None:
            raise SystemExit(
                "--block needs its own --seed0. A block reuses the round's "
                "grid but must NOT reuse its seeds: overlapping seeds would "
                "re-serve the same episodes and call the repeat a new "
                "occasion. Check one with `vworld seeds --check`.")

    seed0 = R.COMP_SEED0 if comp else (args.seed0 if block else R.SEED0)

    def want_for(c):
        return R.COMP_EPISODES if comp else R.episodes_for(c[1])
    want_for.seed0 = seed0

    def path_for(c):
        return C.cell_path(tag, c[0], c[1], c[2])

    grid = R.comp_cells() if comp else R.cells()
    if getattr(args, "level", None) and block:
        grid = [c for c in grid if c[2] == args.level]
    todo, gaps = SW.resume_plan(grid, want_for, path_for, pin_key, pin)
    label = (f"round {args.round}"
             + (" COMP gate" if comp else f" block {block}" if block else ""))
    print(f"{label} — {len(grid)} cell(s), {len(todo)} to run")
    print(f"  pin {pin[:16]}...  seed0={seed0}  "
          f"terminal_at_zero={R.TERMINAL_AT_ZERO}")
    if args.budget is None:
        raise SystemExit("run refuses to start without an explicit --budget.")

    be = Backend(spec)
    resolved = be.resolve_model()
    spent = 0.0
    for i, c in enumerate(todo, 1):
        model, arm, level = c
        p = path_for(c)
        t0 = time.time()

        def serve(runs, seed0, *, _lv=level, _arm=arm):
            return run_fidelity(be, None, runs=runs, steps=30, seed0=seed0,
                                world_id=f"world_eden_{_lv}", narrate=False,
                                eden_level=_lv, eden_arm=_arm,
                                terminal_at_zero=R.TERMINAL_AT_ZERO)

        res, filled = SW.run_cell(
            None, runs=want_for(c), seed0=seed0, level=level, arm=arm,
            terminal_at_zero=R.TERMINAL_AT_ZERO, serve=serve, path=p,
            seeds=gaps.get(c))
        t1 = time.time()
        u = dict(be.usage_total)
        pi, po = R.COHORT[model]
        billed = round(u["prompt_tokens"] / 1e6 * pi
                       + u["completion_tokens"] / 1e6 * po, 5)
        prior = C.load_cell(p).get("meta", {}) if filled else {}
        res["meta"] = {
            "served_name": model, "resolved_model_string": resolved,
            "base_url": spec.base_url, "endpoint": spec.base_url,
            "world_id": f"world_eden_{level}", "world_version": f"world_eden_{level}",
            "eden_level": level, "eden_arm": arm,
            "phrasing": "p1", "step_schedule": "v1", "narrate": False,
            "runs": want_for(c), "steps": 30, "seed0": seed0,
            "stage": stage or f"r{args.round}_{level.lower()}",
            **({"block": block} if block else {}),
            "terminal_at_zero": R.TERMINAL_AT_ZERO, pin_key: pin,
            "temperature": spec.temperature,
            "cohort_temperature": COHORT_TEMPERATURE,
            "temperature_deviation": (
                None if spec.temperature == COHORT_TEMPERATURE
                else f"served at {spec.temperature}, cohort is "
                     f"{COHORT_TEMPERATURE}"),
            # **A real serving timestamp, not an mtime.** The occasion audit can
            # only call a comparison clean when the cells carry this; ten of
            # 259 committed cells do, and every new one should.
            "wall_start_epoch": round(t0), "wall_end_epoch": round(t1),
            "wall_s": round(prior.get("wall_s", 0) + (t1 - t0)),
            "usage": u, "price_per_m": {"prompt": pi, "completion": po},
            "billed_usd": round(prior.get("billed_usd", 0.0) + billed, 5),
            "billed_this_attempt_usd": billed,
            "filled_seeds": (prior.get("filled_seeds") or []) + (filled or []),
            "attempts": prior.get("attempts", 1) + (1 if filled else 0),
        }
        p.write_text(json.dumps(res, indent=2) + "\n")
        spent += billed
        n = len(C.episodes(res))
        print(f"[{i}/{len(todo)}] {model.split('/')[-1][:24]:<26}{arm} {level:<6}"
              f"n={n}/{want_for(c)}  ${res['meta']['billed_usd']:.3f}  "
              f"{round(t1-t0)}s   running ${spent:.2f}", flush=True)
        if spent > args.budget:
            print(f"\n  BUDGET ${args.budget:.2f} EXCEEDED at ${spent:.2f}; "
                  "stopping.")
            return 1
    print(f"\ndone — ${spent:.2f} billed this invocation")
    return 0


def main(args) -> int:
    spec = resolve(args.endpoint, model=args.model, key_env=args.key_env)
    if getattr(args, "block", None) and not getattr(args, "round", None):
        raise SystemExit(
            "--block needs --round. A block is an ADDITIONAL block of a pinned "
            "round's grid; without the round there is no grid to repeat and no "
            "pin to carry, and the ad-hoc path would silently ignore it.")
    if not getattr(args, "round", None) and args.seed0 is None:
        raise SystemExit(
            "--seed0 is required without --round. A seed block is the one thing "
            "a run cannot pick for you: reusing a burned block silently pools "
            "two measurements. Check one with `vworld seeds --check START COUNT "
            "--model <model>`.")
    if args.dry_run:
        class _NoKey:
            pass
        be = _NoKey()
        be.spec = spec
        return _dry_run(be, args.level, args.arm, args.seed0)

    if getattr(args, "round", None):
        return _round_cells(args, spec)

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
        p = out / (f"vworld_{resolved.replace('/', '__')}__{arm}__"
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
            "terminal_at_zero": True, "stage": "vworld_run",
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
    print(f"\n  wrote to {out}/  — read with: vworld read --results {out}")
    print("  NO PASS/FAIL: no band exists for a model that was never measured.")
    return 0
