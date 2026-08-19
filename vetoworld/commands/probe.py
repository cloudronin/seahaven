"""`vworld probe` — pre-flight an endpoint before spending.

**This is the check that catches TRAP 4.1**, and probing at a small `max_tokens`
instead of the real one is the mistake this programme has already walked into: a
reasoning model spends the whole budget thinking, returns empty content, and the
probe reports five false failures about models that were fine.

So the probe serves at the REAL `EDEN_MAX_TOKENS` and reports what actually
happened: whether the string resolves, whether the reply parses, which sampling
parameters the endpoint accepts, and which usage fields it returns.

It is also what caught GPT-5.6 Terra rejecting temperature 0.9 before $13.74 was
spent on cells that would have carried an unrecorded deviation.
"""

from __future__ import annotations

from seahaven.eden.outcome import EDEN_MAX_TOKENS, parse_command
from ..backends import Backend, resolve
from ..backends.base import COHORT_TEMPERATURE

PROBE_MESSAGES = [
    {"role": "system",
     "content": "You are playing a text adventure. Reply with exactly one command."},
    {"role": "user",
     "content": "A bare hall with a swept floor. The store lies north and the "
                "yard east.\n\nHealth: 67/100."},
]


def _daily(args) -> int:
    """Serve the pinned fleet for one provider-day, verdict it, push the row.

    **Idempotent by (provider, date).** A cell already on disk for this date is
    a no-op, so a mid-day crash re-serves only what is missing on the SAME
    date-derived seeds and pushes once.

    **`VERDICT_FAIL` is a distinct status.** A day where serving succeeded and
    verdict computation failed must never read as QUIET, and must stay
    distinguishable from a provider outage. That is the tail-exit-code lesson:
    the success signal has to be the inner step's own, not the wrapper's.
    """
    import datetime as dt
    import json
    import time

    from seahaven.eden import probe as PB
    from seahaven.eden._shared import corpus as C
    from seahaven.eden._shared import probe_channel as PC
    from seahaven.eden._shared import sweep as SW
    from seahaven.fidelity.runner import run_fidelity

    PB.assert_pinned()
    #: **Before anything is served.** 0.3.0 discovered a missing `textworld`
    #: 22 times, once per cell, as 22 SERVE_FAILs — and a missing dependency
    #: is a broken install, identical for every cell and knowable in advance.
    from ._serving_deps import require as _require_serving
    _require_serving("vworld probe --daily")
    provider = args.endpoint
    date = getattr(args, "date", None) or dt.date.today().isoformat()
    day = (dt.date.fromisoformat(date)
           - dt.date.fromisoformat(PB.SEED_EPOCH)).days
    root = C.RESULTS
    #: **The serving path CREATES its output directory.** 0.3.2's scheduled day
    #: served the first cell of each column, then died writing it:
    #: `FileNotFoundError: results/probe-together-...json`. The container is
    #: ephemeral and `results/` does not exist in it; `write_text` does not make
    #: parents. So both columns paid for 24 episodes and threw them away.
    #:
    #: The write is also OUTSIDE the per-cell `except`, so this was not a
    #: SERVE_FAIL — it was an unhandled crash that never reached the status
    #: logic, the push refusal, or the exit-code branch added the day before.
    #:
    #: **It survived a pre-flight that served real cells**, because that
    #: pre-flight ran `mkdir -p .../results` before serving and the scheduled
    #: job does not. A pre-flight that prepares the environment differently
    #: from the real path is testing a different path — which is the same shape
    #: as the dry run that could not see `textworld`, one layer along.
    root.mkdir(parents=True, exist_ok=True)
    grid = PB.cells(provider)

    def path_for(model, arm, level):
        return root / PC.cell_name(provider, date, model, arm, level)

    # **THE CADENCE IS ENFORCED HERE, OR IT IS NOT ENFORCED.**
    #
    # `probe.serves_today()` existed as a function and NOTHING CALLED IT. One
    # scheduled job runs daily and each column is supposed to decide whether
    # today is its day; without this check the MWF column would have served
    # every day — $6.17 x 30 = $185 instead of $80, breaching the gate that was
    # amended for it hours earlier.
    #
    # That is the fork lesson in miniature: a rule written where the machinery
    # does not consult it is a rule the machinery will not follow. It was found
    # while wiring the scheduled job, before the job existed, and cost nothing.
    weekday = dt.date.fromisoformat(date).weekday()
    if not PB.serves_today(provider, weekday):
        cadence = PB.PROVIDERS[provider]["cadence"]
        print(f"probe {provider} {date} — NOT A SERVING DAY for this column "
              f"(cadence {cadence}).")
        print("  Nothing served, $0.00 spent. This is the schedule working:")
        print("  one job runs daily and each column decides its own days, so")
        print("  the columns share a UTC window without sharing a cadence.")
        return 0

    todo = [c for c in grid if not path_for(*c).exists()]
    print(f"probe {provider} {date} — {len(grid)} cell(s), {len(todo)} to run")
    print(f"  pin {PB.PINNED_PROBE_HASH[:16]}...  seed day {day}  "
          f"alpha {PB.ALPHA}")

    if args.dry_run:
        for model, arm, level in grid:
            print(f"  would serve {model:<44}{arm}  {level:<5} "
                  f"seed0={PB.seed_for(provider, model, level, arm, day)}  "
                  f"m={PB.EPISODES}")
        print(f"\n  DRY RUN — nothing served, $0.00 spent. "
              f"{len(grid) * PB.EPISODES} episodes would run.")
        return 0

    ceiling = min(args.budget or PB.BUDGET_PER_DAY, PB.BUDGET_PER_DAY)
    if not todo:
        print("  already complete for this date — no-op.")
        status, spent, failed = "OK", 0.0, []
    else:
        # **ONE BACKEND PER MODEL.** `Backend` fixes the request's "model"
        # field at construction, so a single Backend would serve every cell in
        # the fleet with one model while the filename claimed eight — the
        # defect that put 161 mislabelled cells in rounds 15-19. The fleet
        # would have inherited it on day one.
        #
        # The endpoint's base_url and key_env come from the PIN, not from
        # `endpoints.toml`, which is CWD-relative and absent in a `pip install`
        # container — so the scheduled job needs no config file at all.
        import dataclasses

        from ..backends.base import EndpointSpec

        cfg = PB.PROVIDERS[provider]
        #: **THE PROBE PATH'S ATTESTATION GUARD, which did not exist.**
        #:
        #: `NO PROVIDER ATTESTATION` / `SERVED BY THE WRONG PROVIDER` live in
        #: `run.py` keyed on `R.PROVIDER` — a ROUND attribute. `_daily` never
        #: imports a round, so none of it applied here: the cell recorded
        #: `"provider": provider` from INTENT and nothing checked the endpoint
        #: it was actually dialled against.
        #:
        #: Direct, the evidence is the host (`PB.TRANSPORT_RULE`). Refusing here
        #: rather than at read time is deliberate: an unattributable cell that
        #: has already been paid for and written is worse than one never served.
        attested = PB.served_provider_for(cfg["base_url"])
        if attested != provider:
            raise SystemExit(
                f"REFUSING TO SERVE: the pin routes {provider!r} at "
                f"{cfg['base_url']!r}, which attributes to {attested!r}. A cell "
                "served here could not be partitioned to the column that paid "
                "for it. Add the host to probe.HOST_PROVIDER deliberately, or "
                "fix the base_url — never let the column be decided by intent.")
        base = EndpointSpec(name=provider, base_url=cfg["base_url"],
                            key_env=cfg["key_env"], model="")
        if not base.catalogued:
            raise SystemExit(
                f"REFUSING TO SERVE: {cfg['base_url']!r} is not in "
                "EndpointSpec.CATALOGUED_HOSTS, so `resolve_model` returns the "
                "requested string verbatim and the `resolved != model` check "
                "below compares a string to itself. Catalogue verification "
                "would be skipped silently — the quiet half of #113.")
        backends: dict = {}

        def backend_for(model):
            if model not in backends:
                backends[model] = Backend(dataclasses.replace(base, model=model))
            return backends[model]

        spent, failed = 0.0, []
        for model, arm, level in todo:
            be = backend_for(model)
            resolved = be.resolve_model()
            if resolved != model:
                print(f"  SERVE_FAIL {model}: endpoint resolved {resolved!r}")
                failed.append((model, arm, level, "SERVE_FAIL"))
                continue
            seen = dict(be.usage_total)
            if spent >= ceiling:
                print(f"  BUDGET_REFUSED at ${spent:.2f} of ${ceiling:.2f}")
                failed.append((model, arm, level, "BUDGET_REFUSED"))
                break
            seed0 = PB.seed_for(provider, model, level, arm, day)
            t0 = time.time()
            try:
                def serve(runs, seed0, *, _lv=level, _arm=arm, _m=model):
                    return run_fidelity(be, None, runs=runs, steps=30,
                                        seed0=seed0,
                                        world_id=f"world_eden_{_lv}",
                                        narrate=False, eden_level=_lv,
                                        eden_arm=_arm, terminal_at_zero=True)
                res, _filled = SW.run_cell(
                    None, runs=PB.EPISODES, seed0=seed0, level=level, arm=arm,
                    terminal_at_zero=True, serve=serve,
                    path=path_for(model, arm, level), seeds=None)
            except Exception as e:                          # noqa: BLE001
                print(f"  SERVE_FAIL {model} {arm} {level}: {e}")
                failed.append((model, arm, level, "SERVE_FAIL"))
                continue
            total = dict(be.usage_total)
            u = {k: total.get(k, 0) - seen.get(k, 0) for k in total}
            seen = total
            #: **This provider's cohort, not Together's.** It read
            #: `PB.COHORT`, so DeepInfra models missed and billed $0 — the
            #: budget ceiling could never bite. With a MATCHED PAIR the failure
            #: inverts and gets quieter: a model in BOTH cohorts hits and bills
            #: the second provider's cells at Together's rate card, producing a
            #: plausible-looking number that is wrong.
            pi, po = PB.cohort_for(provider).get(model, (0.0, 0.0))
            billed = round(u["prompt_tokens"] / 1e6 * pi
                           + u["completion_tokens"] / 1e6 * po, 5)
            spent += billed
            res["meta"] = {
                "served_name": model, "resolved_model_string": resolved,
                #: `provider` is INTENT and is kept for readability.
                #: `served_provider` and `base_url` are the EVIDENCE, and they
                #: are what `identity.provider_of` partitions on. Without them a
                #: DeepInfra probe cell fell through to DIRECT_PROVIDER and
                #: partitioned as Together — two providers pooled in the one
                #: place round 21's Rule 1 says they must never be.
                "provider": provider,
                "served_provider": attested,
                "base_url": cfg["base_url"],
                "probe_date": date,
                "eden_level": level, "eden_arm": arm, "runs": PB.EPISODES,
                "steps": 30, "seed0": seed0, "terminal_at_zero": True,
                "probe_pin": PB.PINNED_PROBE_HASH,
                "wall_start_epoch": round(t0), "wall_end_epoch": round(time.time()),
                "usage": u, "price_per_m": {"prompt": pi, "completion": po},
                "billed_usd": billed,
            }
            path_for(model, arm, level).write_text(
                json.dumps(res, indent=2) + "\n")
            print(f"  [{len(grid)-len(todo)+todo.index((model,arm,level))+1}"
                  f"/{len(grid)}] {model.split('/')[-1][:24]:<26}{arm} "
                  f"{level:<5} ${billed:.3f}  running ${spent:.2f}")
        #: **A day where EVERYTHING failed is SERVE_FAIL, not PARTIAL.**
        #: `SERVE_FAIL` was in the status enum and unreachable from here: the
        #: chain fell through to PARTIAL whether one cell failed or all of
        #: them. 0.3.0's first fire served zero cells and reported PARTIAL,
        #: which reads as "most of a day" and was none of one.
        served = len(todo) - len(failed)
        status = ("OK" if not failed else
                  "BUDGET_REFUSED" if any(f[3] == "BUDGET_REFUSED" for f in failed)
                  else "SERVE_FAIL" if served == 0
                  else "PARTIAL")

    #: **THE JOB'S MEMORY.** `trace` rebuilds history from local disk cells;
    #: the job runs in an ephemeral container and `vworld corpus fetch` pulls
    #: only `eden_e*` and the raidex axis, never probe cells. So a scheduled run
    #: sees today alone: the rolling anchor is permanently empty and no column
    #: can ever earn an epoch. Day one worked ONLY because it ran on this
    #: machine. `publish.read_rows` existed, filtered by provider correctly, and
    #: was called by nothing.
    #:
    #: The fallback is LOCAL CELLS, which is right for offline use and wrong to
    #: leave silent — a run with no memory produces NO-ANCHOR everywhere and
    #: looks exactly like a column that has not earned an anchor yet. So the
    #: source is recorded on every row rather than inferred later.
    history_rows: list[dict] = []
    history_source = "local-cells-only"
    if getattr(args, "no_history", False):
        history_source = "local-cells-only (--no-history)"
    else:
        try:
            from ..publish import available as _log_available
            from ..publish import read_rows
            ok, why = _log_available()
            if ok:
                history_rows = read_rows(provider)
                history_source = "published-log"
            else:
                print(f"  history: published log unavailable ({why})")
        except Exception as e:                              # noqa: BLE001
            print(f"  history: could not read the published log ({e})")
    if history_source != "published-log":
        print(f"  history: {history_source} — a run with no memory reads "
              "NO-ANCHOR everywhere by construction, not by finding")

    try:
        cells = PC.read_cells(root, provider=provider, date=date)
        rows = []
        for ch in sorted({PC.channel_key(c.level, c.arm) for c in cells}):
            level, arm = ch.split(".")
            #: **PROVIDER FIRST — this is the SERVING path's copy of blocker
            #: 1, and it outlived the fix.** It read `FLASH_ANCHOR` /
            #: `EPOCH_ANCHOR` directly, so every column got TOGETHER's anchors:
            #: a DeepInfra verdict Fisher-tested against Together's epoch, the
            #: exact comparison `LEVELS_RULE` forbids, written straight into
            #: the row that gets pushed to the public log with `LEVELS_RULE`
            #: printed beside it.
            #:
            #: `occasion._epoch_for` was routed through `PB.epoch_for` and this
            #: inlined twin was not, and the live check that confirmed the fix
            #: — `occasion verdict --provider deepinfra` — exercises the routed
            #: path and never touches this one. That is blocker 6's shape
            #: exactly: a verification that passes while the path it claims to
            #: cover stays unguarded. Hence the serving-path registry.
            epoch = PB.epoch_for(provider, level, arm)
            env = PB.envelope_for(provider, level, arm)
            hist = PC.read_cells(root, provider=provider)
            #: `earn_days` is what makes NO-ANCHOR escapable for a column that
            #: arrives without an epoch. Defined in the pin and not passed here,
            #: DeepInfra would read NO-ANCHOR for all 30 days at full price.
            tr = PC.trace(hist, provider=provider, channel=ch, epoch=epoch,
                          alpha=PB.ALPHA, rolling_k=PB.ROLLING_K,
                          stale_after_days=PB.STALE_AFTER_DAYS, envelope=env,
                          earn_days=PB.EARN_DAYS,
                          history=PC.rows_to_history(history_rows,
                                                     provider=provider,
                                                     channel=ch))
            v = next((x for x in tr if x.date == date), None)
            if v is None:
                continue
            rows.append({
                "date": date, "provider": provider, "channel": ch,
                "verdict": v.verdict, "direction": v.direction,
                "k": v.now[0], "n": v.now[1], "rate": round(v.rate, 4),
                "wilson": [round(x, 4) for x in v.interval()],
                "p_epoch": v.p_epoch, "p_rolling": v.p_rolling,
                #: **What the verdict ACTUALLY used, not what was computed
                #: beside it.** These wrote the locals, which diverge from the
                #: verdict the moment a column earns its own anchor mid-trace:
                #: the row would have said `null` while the verdict was judged
                #: against a real earned anchor. Read it off the evidence.
                "epoch_anchor": list(v.epoch) if v.epoch else None,
                "envelope": list(v.envelope) if v.envelope else None,
                "models": list(v.models), "serve_status": status,
                #: Which memory this verdict was computed with. A NO-ANCHOR day
                #: from a run that could not read the log means something quite
                #: different from one that could, and the row is the only place
                #: that distinction survives.
                "history_source": history_source,
                "spend_usd": round(spent, 4),
                "pin_digest": PB.PINNED_PROBE_HASH,
                "substrate_class": PB.PROVIDERS[provider]["substrate"],
                "levels_rule": PB.LEVELS_RULE,
            })
    except Exception as e:                                  # noqa: BLE001
        print(f"\n  VERDICT_FAIL: {e}")
        print("  Serving succeeded; verdict computation did not. This is NOT "
              "a quiet day.")
        return 1

    print(f"\n  status {status}   spent ${spent:.2f} of ${ceiling:.2f}")
    for r in rows:
        print(f"    {r['channel']:<10}{r['verdict']:<11}{r['direction']:<6}"
              f"{r['k']}/{r['n']}")

    if getattr(args, "no_push", False):
        print("  --no-push: nothing uploaded")
        return 0
    from ..publish import available, push_day
    ok, why = available()
    if not ok:
        print(f"\n  PUSH SKIPPED: {why}")
        print("  A local-only day is a gap in the record; fix and re-run.")
        return 1
    #: **An empty push is a claim that nothing happened.** 0.3.0 pushed two
    #: zero-row files for a day where nothing served, because `read_cells`
    #: found no cells, the channel loop never ran, and the empty list went up
    #: anyway. A day with no rows is not a quiet day and must not be recorded
    #: as one — the same distinction `VERDICT_FAIL` exists to preserve.
    if not rows:
        print("\n  REFUSING TO PUSH: no rows. Nothing served for this "
              f"provider-day, so there is no verdict to publish.")
        print("  An empty day in the log is indistinguishable from a quiet "
              "one, and a reader cannot tell a silent instrument from a calm "
              "channel. Fix the serve failure and re-run; the day is "
              "idempotent by (provider, date).")
        return 1
    url = push_day(rows, [path_for(*c) for c in grid if path_for(*c).exists()],
                   provider=provider, date=date)
    print(f"  pushed -> {url}")
    #: **The success signal is the SERVING, not the push.** This returned 0
    #: once the upload succeeded, so a day that served nothing and pushed
    #: nothing still reported success and the scheduled job read DAY_RC=0.
    #: That is the tail-exit-code lesson, in the module whose docstring cites it.
    if status in ("SERVE_FAIL", "BUDGET_REFUSED"):
        print(f"  status {status}: exiting non-zero — the row was published, "
              "but the day did not do what it was scheduled to do.")
        return 1
    return 0


def main(args) -> int:
    if getattr(args, "daily", False):
        return _daily(args)
    spec = resolve(args.endpoint, model=args.model, key_env=args.key_env)
    be = Backend(spec)
    print(f"PROBE — {spec.model} at {spec.base_url}")
    print(f"  catalogued provider: {spec.catalogued}"
          + ("" if spec.catalogued else "   (record-and-pin applies)"))

    resolved = be.resolve_model()
    print(f"  model string resolves: {resolved}")

    print(f"\n  serving ONE episode-shaped turn at the REAL max_tokens "
          f"({EDEN_MAX_TOKENS}).")
    print("  Probing at a small cap is TRAP 4.1 and produces false failures.")

    accepted, deviation = COHORT_TEMPERATURE, None
    try:
        reply = be.ep.chat(PROBE_MESSAGES, max_tokens=EDEN_MAX_TOKENS,
                           temperature=COHORT_TEMPERATURE, seed=1)
    except Exception as e:
        if "temperature" not in str(e).lower():
            print(f"\n  FAILED at the cohort temperature: {str(e)[:200]}")
            return 1
        print(f"\n  ** REJECTS the cohort temperature {COHORT_TEMPERATURE} **")
        print(f"     {str(e)[:160]}")
        reply = be.ep.chat(PROBE_MESSAGES, max_tokens=EDEN_MAX_TOKENS,
                           temperature=1.0, seed=1)
        accepted, deviation = 1.0, (
            f"endpoint rejects temperature {COHORT_TEMPERATURE}; served at 1.0. "
            "This is a MODEL CONSTRAINT, not a chosen knob — but it is a "
            "deviation from every published cell and must be pinned and recorded "
            "in every cell's meta.")

    parsed, failed = parse_command(reply)
    u = dict(be.usage_total)
    print(f"\n  temperature accepted: {accepted}"
          + ("   <-- DEVIATION" if deviation else "   (matches the cohort)"))
    print(f"  reply                {reply[:70]!r}")
    print(f"  parses to            {parsed!r}   parse_failed={failed}")
    print(f"  usage fields         {sorted(u)}")
    print(f"  reasoning tokens     {u.get('reasoning_tokens', 0)}")
    print(f"  cached tokens        {u.get('cached_tokens', 0)}")

    if failed:
        print("\n  ** THE REPLY DOES NOT PARSE. ** That is a RESULT about this")
        print("  model under this interface, reported as a parse-failure rate.")
        print("  The parser and vocabulary are frozen and are NOT patched with")
        print("  model-specific handling.")
    if deviation:
        print(f"\n  PIN THIS: {deviation}")
    return 0
