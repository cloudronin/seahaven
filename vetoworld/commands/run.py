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
from seahaven.eden._shared import identity as ID
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

    # **ONE BACKEND PER MODEL, and this was the defect.** `Backend` fixes the
    # request's `"model"` field at construction from `spec.model`
    # (`endpoint.py`), and `spec` came from `endpoints.toml`, whose `together`
    # entry names ONE model. So a single Backend served every cell in the grid
    # while `meta.served_name` recorded the model the grid INTENDED — 161 cells
    # across rounds 15-19 carry a `resolved_model_string` that is not their
    # `served_name`, and every one of them was actually served by cogito.
    #
    # Token counts are the tell: round 18's eight "different models" agreed to
    # within 1.09x on prompt tokens and 1.05x on completion, while billing
    # differed 7x because `R.COHORT[model]` charged each at its intended price.
    import dataclasses

    backends: dict[str, Backend] = {}

    def backend_for(model: str) -> Backend:
        if model not in backends:
            # **A round may pin WHICH PROVIDER serves it.** Round 21 routes
            # through the HuggingFace router, where `org/model:provider` selects
            # the third party; without the suffix the router picks "fastest
            # available" per request and a round whose provider varies request
            # to request is not one measurement. `served_id` is the round's own
            # translation, so a round with no routing leaves the id untouched.
            wire = R.served_id(model) if hasattr(R, "served_id") else model
            backends[model] = Backend(dataclasses.replace(spec, model=wire))
        return backends[model]

    # **AVAILABILITY PRE-FLIGHT, before the first cell.** A provider can move a
    # model to a dedicated-only tier with no notice, and nothing in the corpus
    # or the pin can see it: round 20's COMP gate discovered six of round 15's
    # fourteen had gone non-serverless, on its first cell. That abort cost
    # $0.00, which was luck of ordering — had the unavailable model been
    # fourteenth, thirteen cells would have been served into a round whose
    # cohort could never be completed.
    #
    # So every model answers a 1-token request first. This is round 13's
    # identity discipline generalised to serve time: check the thing exists
    # before believing anything about it, and never learn it by spending.
    want_models = sorted({m for m, _a, _lv in [c for c in todo]})
    if want_models:
        print(f"  pre-flight: {len(want_models)} model(s) ...", flush=True)
        # **The pre-flight must send the REQUEST FORM the round will send.**
        # A first version called `resolve_model()` only — availability, not
        # servability — and passed both of round 20's models. The serve then
        # died on Kimi-K3, because Together truncates its reasoning to 16
        # tokens under `chat_template_kwargs.enable_thinking=False` and returns
        # empty content. Availability, identity and servability are three
        # different questions and only the third needs a real call.
        from seahaven.eden.outcome import EDEN_MAX_TOKENS
        dead = []
        for m in want_models:
            try:
                got = backend_for(m).resolve_model()
                if got != m:
                    dead.append((m, f"endpoint resolved {got!r}"))
                    continue
                backend_for(m).chat([{"role": "user", "content": "ready"}],
                                    max_tokens=EDEN_MAX_TOKENS)
            except Exception as e:                       # noqa: BLE001
                dead.append((m, str(e).split("\n")[0][:120]))
        if dead:
            print()
            for m, why in dead:
                print(f"  UNAVAILABLE  {m}\n               {why}")
            raise SystemExit(
                f"\n{len(dead)} of {len(want_models)} cohort model(s) cannot be "
                "served. Nothing was spent.\n"
                "  A round whose cohort cannot be completed must not be part-"
                "served: the\n"
                "  cells would pin a grid that can never be finished. Amend the "
                "round's\n"
                "  COHORT and re-pin BEFORE any cell, or resolve access with the "
                "provider.")

    spent = 0.0
    # **`usage_total` is CUMULATIVE over the backend's lifetime**, and the
    # backend is created once so the connection is reused. Reading it directly
    # per cell therefore records every previous cell's tokens as well: the
    # round-15 COMP gate showed prompt tokens climbing 991k, 1.98M, 2.98M... in
    # a perfect arithmetic progression for fourteen identical 24-episode cells,
    # and billed a 7x-inflated $7.02 for the seventh. Snapshot and diff.
    seen_usage: dict = {}
    for i, c in enumerate(todo, 1):
        model, arm, level = c
        be = backend_for(model)
        resolved = be.resolve_model()
        seen_usage = dict(be.usage_total)
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
        total = dict(be.usage_total)
        u = {k: total.get(k, 0) - seen_usage.get(k, 0) for k in total}
        seen_usage = total
        # **The cell records what was SERVED, and refuses if that is not what
        # was asked for.** The guard that would have caught the 161.
        # **Compared on BARE ids.** A routed request pins its provider with a
        # `:provider` suffix that the response echoes stripped, so the raw
        # strings differ on every routed cell while naming the same model.
        # `ID.bare_model` is the single place that strips; doing it inline here
        # would put a second, looser identity rule beside the strict one.
        if ID.bare_model(resolved) != ID.bare_model(model):
            raise SystemExit(
                f"SERVED THE WRONG MODEL: asked for {model!r}, the endpoint "
                f"resolved {resolved!r}. A cell that records one model and "
                "was served by another is not a measurement. Refusing.")

        # **Which provider answered, taken from the wire, not from intent.**
        # `None` on a direct endpoint. On a router it is the attestation, and a
        # round that pinned a provider must get the one it pinned — otherwise
        # the cells record a provider we asked for, which is precisely the
        # class of claim #113 was.
        served_provider = getattr(be.ep, "last_provider", None) \
            if hasattr(be, "ep") else None
        want_provider = getattr(R, "PROVIDER", None)
        if want_provider and served_provider and served_provider != want_provider:
            raise SystemExit(
                f"SERVED BY THE WRONG PROVIDER: round {args.round} pins "
                f"{want_provider!r}, the router reports {served_provider!r}. "
                "Refusing rather than recording a provider that did not serve.")
        if want_provider and not served_provider:
            raise SystemExit(
                f"NO PROVIDER ATTESTATION: round {args.round} pins "
                f"{want_provider!r} but the response carried no "
                "`x-inference-provider` header, so nothing in the exchange "
                "shows who served it. Refusing rather than asserting it.")

        pi, po = R.COHORT[model]
        billed = round(u["prompt_tokens"] / 1e6 * pi
                       + u["completion_tokens"] / 1e6 * po, 5)
        prior = C.load_cell(p).get("meta", {}) if filled else {}
        res["meta"] = {
            "served_name": model, "resolved_model_string": resolved,
            #: **Provenance, per cell.** `served_provider` is what the wire
            #: attested; `pinned_provider` is what the round required. Both are
            #: written so a reader can see they agreed rather than trusting
            #: that they must have. None/None is a direct endpoint.
            "served_provider": served_provider,
            "pinned_provider": want_provider,
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
            # **A real serving timestamp, not an mtime.** The occasion audit
            # can only call a comparison clean when the cells carry this.
            # Only cells served through this path carry one, and every new
            # cell should — the live split is what `emit occasions` prints.
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
        # **`--dry-run` did not know about `--round`.** It was checked first, so
        # `run --round 16 --dry-run` never reached the round path and died on
        # `seed0=None` inside the runner — a traceback from the one verb whose
        # whole purpose is to be safe to call before spending. The round carries
        # its own level and seed base; take them rather than requiring the
        # caller to restate constants the pin already fixes.
        level, arm, seed0 = args.level, args.arm, args.seed0
        if getattr(args, "round", None):
            import importlib
            R = importlib.import_module(f"seahaven.eden.round{args.round}")
            level = getattr(R, "LEVEL", None) or R.LEVELS[0]
            seed0 = R.COMP_SEED0 if getattr(args, "stage", None) == "comp" \
                else R.SEED0
            print(f"round {args.round} — assembling {level} {arm} at "
                  f"seed0={seed0}; nothing is served")

            # **The grid is printed BEFORE the prompt preview, because the
            # prompt preview names one model and the grid is what actually
            # gets served.** `--dry-run` returns above `_round_cells`, so the
            # message below is assembled from the ENDPOINT's default model —
            # which for `together` is the very string that served all 166 cells
            # of #113. Showing only that, under the heading of a round, is how
            # a reviewer confirms the wrong thing. So the per-model grid is
            # rendered here, from the same `R.cells()` the serving path walks.
            grid = R.comp_cells() if getattr(args, "stage", None) == "comp" \
                else R.cells()
            models = sorted({m for m, _a, _lv in grid})
            print(f"\n  WOULD SERVE {len(grid)} cell(s) across {len(models)} "
                  "model(s):")
            for m in models:
                mine = [(a, lv) for mm, a, lv in grid if mm == m]
                print(f"    {m:<44}{len(mine)} cell(s)")
            print("\n  Each model gets its OWN backend and the run refuses any "
                  "cell whose\n  endpoint resolves a different model (#113). "
                  "The prompt below is\n  assembled from the endpoint default "
                  f"({spec.model}) and previews WORDING\n  only — it is not "
                  "the model the grid serves.\n")
        return _dry_run(be, level, arm, seed0)

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
