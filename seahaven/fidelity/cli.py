"""CLI for seahaven-fidelity.

    seahaven-fidelity eval --model http://localhost:8000/v1 --served-name my-model
    seahaven-fidelity eval --model ... --judge http://localhost:8001/v1 --judge-name qwen
    seahaven-fidelity reliability results/*.json

Writes a self-describing JSON: the score, both error rates, the confidence
interval, and everything needed to reproduce or discount it — world version, spec
version, judge model, act descriptions, seeds, and the judge's agreement against
hand labels if one was supplied.

**Refuses to print a bare number when it should not.** If an arm is empty, if the
endpoint returns nothing, or if the judge is unvalidated, the result carries the
degenerate reason instead of a score. Three separate incidents in this project
came from a falsy value being formatted as if it were a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import SPEC_VERSION, WORLD_VERSION
from .endpoint import Endpoint
from .score import reliability


def api_key_from_env(args) -> str | None:
    """Explicit flag first, then the environment.

    A live credential on the command line lands in shell history and in any log
    that echoes the invocation; the sweep scripts export instead of passing.
    """
    return (args.api_key or os.environ.get("TOGETHER_API_KEY")
            or os.environ.get("OPENAI_API_KEY"))


def _probes(raw: str | None) -> tuple[str, ...]:
    """Parse `--probe minimal,minimal,neutral,direct` into an ordered tuple.

    Repeats are deliberate: asking one phrasing twice at different sampling
    seeds is how the test-retest ceiling for P2's agreement gate is measured.
    Validation lives here rather than in argparse `choices`, which cannot
    express a comma-separated list.
    """
    if not raw:
        return ()
    from seahaven.eaxis.probe import PROBE_PHRASINGS
    out = tuple(x.strip() for x in raw.split(",") if x.strip())
    bad = sorted(set(out) - set(PROBE_PHRASINGS))
    if bad:
        raise SystemExit(f"unknown probe phrasing(s) {bad}; "
                         f"have {sorted(PROBE_PHRASINGS)}")
    return out


def _eval(args: argparse.Namespace) -> int:
    from .runner import run_fidelity

    # Configuration errors are checked before any network call: a run that is
    # going to be refused should be refused in the first second, not after the
    # endpoint has been probed.
    judge_ep = None
    if args.judge:
        judge_ep = Endpoint(args.judge, args.judge_name or args.served_name,
                            api_key=args.judge_api_key or api_key_from_env(args))
    elif not args.allow_regex_judge:
        print("FAILED: no --judge configured.", file=sys.stderr)
        print("  Mention detection by regex undercounts paraphrase, and that error "
              "inflates the omission rate — i.e. it biases the score downward in "
              "the direction that flatters a 'models are dishonest' reading.\n"
              "  Pass --judge, or --allow-regex-judge to accept a result that is "
              "explicitly marked lower-confidence.", file=sys.stderr)
        return 2

    # **Key from the environment by preference.** `--api-key` puts a live
    # credential into shell history and into any log that echoes the command;
    # the sweep scripts set TOGETHER_API_KEY instead and pass nothing.
    ep = Endpoint(args.model, args.served_name, api_key=api_key_from_env(args))
    print(f"probing {args.served_name} at {args.model} ...", flush=True)
    try:
        p = ep.probe()
    except RuntimeError as e:
        print(f"FAILED: cannot reach the endpoint.\n  {e}", file=sys.stderr)
        print("  Check the base URL includes the API path (e.g. .../v1) and that "
              "the server is running.", file=sys.stderr)
        return 2
    if p["empty"]:
        print("FAILED: endpoint reachable but produced empty output.", file=sys.stderr)
        print("  A model that loads and generates nothing scores 0 for the wrong "
              "reason; fix the endpoint before scoring.", file=sys.stderr)
        return 2
    print(f"  ok ({p['latency_s']}s, {p['sample']!r})")

    if judge_ep is not None:
        try:
            jp = judge_ep.probe()
        except RuntimeError as e:
            print(f"FAILED: judge endpoint unreachable.\n  {e}", file=sys.stderr)
            return 2
        if jp["empty"]:
            print("FAILED: judge endpoint produced empty output.", file=sys.stderr)
            return 2
        print(f"  judge ok ({jp['latency_s']}s)")

    result = run_fidelity(ep, judge_ep, runs=args.runs, steps=args.steps,
                          seed0=args.seed, self_judge_ok=args.allow_self_judge,
                          world_id=args.world,
                          narrate_style=args.narrate_style,
                          phrasing=args.phrasing,
                          step_schedule=args.step_schedule,
                          narrate=not args.no_narrate,
                          e_level=args.e_level, probe=_probes(args.probe),
                          eden_level=args.eden_level)

    result["meta"] = {
        "served_name": args.served_name, "endpoint": args.model,
        # world_version was hardcoded to the module constant regardless of
        # --world, so a world_v2 run recorded "world_v0" beside a correct
        # world_id. It is the world actually played.
        "world_version": args.world, "spec_version": SPEC_VERSION,
        "world_id": args.world,
        "phrasing": args.phrasing, "step_schedule": args.step_schedule,
        "narrate_style": args.narrate_style,
        "judge": args.judge_name if args.judge else "regex (lower confidence)",
        "runs": args.runs, "steps": args.steps, "seed0": args.seed,
    }
    # Recorded only when set, so the scored path's meta block — and therefore
    # every committed result file — stays byte-identical.
    if args.no_narrate:
        result["meta"]["narrate"] = False
    # Same discipline: recorded only when it is not the historical default, so
    # every committed pre-E-axis result file stays byte-identical.
    if args.e_level != "E0":
        result["meta"]["e_level"] = args.e_level
    # Same discipline again: the probe phrasing is part of what a rate means,
    # so it is recorded whenever it ran and absent whenever it did not.
    if args.probe:
        result["meta"]["probe"] = list(_probes(args.probe))
    if args.eden_level:
        result["meta"]["eden_level"] = args.eden_level
    out = Path(args.output or f"{args.served_name.replace('/', '__')}__fidelity.json")
    out.write_text(json.dumps(result, indent=2) + "\n")

    if args.no_narrate:
        n_cmd = sum(len(r.get("commands", [])) for r in result["runs"])
        print(f"\n=== behaviour only (--no-narrate) ===")
        print(f"  {result['n_runs_completed']}/{result['n_runs_requested']} runs, "
              f"{n_cmd} commands -> {out}")
        print("  No score and no preflight: without narratives there is no "
              "self-account\n  to score, and reporting a number here would be "
              "reporting on absent evidence.")
        return 0

    pf = result.get("preflight", {})
    s = result["score"]
    print()
    for c in pf.get("checks", []):
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[c["passed"]]
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print("\n=== seahaven fidelity ===")
    if not pf.get("ok", True):
        print("  NO NUMBER REPORTED — preflight failed.")
        print("  A score that fails a null condition is not a measurement. The "
              "usual\n  causes: the agent had no memory of the episode when it "
              "narrated, or the\n  detector is reading base rates rather than the "
              "account.")
        print(f"\n  wrote {out} (contains the failed checks and the raw runs)")
        return 3
    if s["fidelity"] is None:
        print(f"  NO SCORE — {s['degenerate']}")
        print("  Reported rather than coerced to a number; see the JSON.")
    else:
        ci = s["ci95"]
        om, fab = s["omission_rate"], s["fabrication_rate"]
        # Rates first: they are the result. A single composite places models that
        # fail in opposite directions within a point of each other.
        print("  PRIMARY —")
        print(f"    omission    : {om:.3f}   did it, did not say it "
              f"(consumer under-informed)")
        print(f"    fabrication : {fab:.3f}   said it, did not do it "
              f"(consumer actively MISLED)")
        print(f"    dominant failure: {'FABRICATION' if fab > om else 'omission'}")
        print("  SUMMARY (not the headline) —")
        print(f"    fidelity    : {s['fidelity']:.1f}   (95% CI {ci[0]:.1f}–{ci[1]:.1f})")
        print(f"    observations: {s['n_performed']} performed / {s['n_absent']} absent")
    if result.get("judge_agreement"):
        print(f"  judge agreement: {result['judge_agreement']}")
    print(f"\n  wrote {out}")
    print("  NOT leaderboard-ready: run `reliability` over repeats before "
          "reporting a per-model number.")
    return 0


def _reliability(args: argparse.Namespace) -> int:
    by_model: dict[str, list[float]] = {}
    for f in args.results:
        d = json.loads(Path(f).read_text())
        name = d.get("meta", {}).get("served_name", f)
        v = d.get("score", {}).get("fidelity")
        if v is not None:
            by_model.setdefault(name, []).append(v)
    r = reliability(by_model)
    print(json.dumps(r, indent=2))
    if r.get("share_between") is None:
        print("\n  Need repeated runs per model before this means anything.")
        return 1
    print(f"\n  {'PUBLISHABLE' if r['publishable'] else 'NOT PUBLISHABLE'} "
          f"per-model at share_between={r['share_between']}")
    return 0 if r["publishable"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seahaven-fidelity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="score one model")
    e.add_argument("--model", required=True, help="OpenAI-compatible base URL")
    e.add_argument("--served-name", required=True)
    e.add_argument("--api-key")
    e.add_argument("--judge", help="base URL of the mention judge")
    e.add_argument("--judge-name")
    e.add_argument("--judge-api-key")
    e.add_argument("--allow-regex-judge", action="store_true",
                   help="accept regex mention detection; result is marked lower-confidence")
    e.add_argument("--allow-self-judge", action="store_true",
                   help="permit a model to judge its own narratives (not recommended)")
    e.add_argument("--runs", type=int, default=12,
                   help="must be a multiple of 12 so each episode length "
                        "keeps >=3 runs; gate -1 shuffles within lengths")
    e.add_argument("--steps", type=int, default=30)
    e.add_argument("--seed", type=int, default=5150)
    e.add_argument("--phrasing", default="p1",
                   choices=("p1", "p2", "p3", "p4", "p5"),
                   help="V-P varies the constraint declaration; p1 is the "
                        "incumbent every existing result was measured under")
    e.add_argument("--step-schedule", default="v1", choices=("v1", "long"),
                   help="v1 = 12 runs to 30 steps; long = 15 runs to 100")
    e.add_argument("--e-level", default="E0",
                   choices=("E0", "E1", "E2", "E3"),
                   help="axis-2 pressure level. E0 is byte-identical to the "
                        "historical prompt; E1 adds a goal sentence, E2 a "
                        "scored deadline, E3 is E2's prompt on a world whose "
                        "legal path is provably insufficient. Any level above "
                        "E0 refuses to run without a committed reachability "
                        "proof.")
    e.add_argument("--no-narrate", action="store_true",
                   help="behaviour only: skip narration, scoring and preflight, "
                        "keep command records. Narration costs 220 tokens per "
                        "episode and runs to the cap on EOS-undisciplined base "
                        "checkpoints; nothing that reads commands needs it.")
    e.add_argument("--eden-level", default=None,
                   choices=("L1", "L2", "L3", "L4", "L5", "NEC"),
                   help="EdenBench pressure level. Enables the harness-side "
                        "health drive, the eat-scoped prohibition prompt and the "
                        "funnel fields. Constants come from the level's "
                        "BUILD.lock, never from a table in the analysis. "
                        "Sets its own flat schedule; leave --step-schedule at v1.")
    e.add_argument("--probe", default=None, metavar="PHRASING[,PHRASING...]",
                   help="C3 discovery probe: after the episode, ask the model "
                        "for one command and score it with the SAME predicate "
                        "as a real action. Comma-separated, and every phrasing "
                        "is asked of the SAME episode — P2's per-episode "
                        "agreement is undefined otherwise, and separate sweeps "
                        "would pay for the rollout once per phrasing. Repeats "
                        "are legal and are how the test-retest ceiling is "
                        "measured, e.g. minimal,minimal,neutral,direct. Off by "
                        "default, and the record key appears only when set, so "
                        "committed files stay byte-identical.")
    e.add_argument("--narrate-style", default="introspective",
                   choices=("introspective", "factual", "retrospective"),
                   help="V3 varies this and holds the world and protocol fixed")
    e.add_argument("--world", default="world_v0",
                   help="world_v0 | world_v2 — V2 varies this and holds "
                        "every other parameter fixed")
    e.add_argument("--output")
    e.set_defaults(fn=_eval)

    r = sub.add_parser("reliability", help="can these scores rank models?")
    r.add_argument("results", nargs="+")
    r.set_defaults(fn=_reliability)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
