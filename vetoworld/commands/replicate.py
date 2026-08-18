"""`vworld replicate` — re-serve our cells, judged against bands.

**Two claims, two success criteria.** `verify` recomputes the paper from
committed cells and is byte-exact, achievable by anyone, $0. This is the other
claim — that a NEW run lands where ours did — and it is achievable only within
bands, because the programme measured a 0.319 between-occasion shift whose
mechanism is unresolved.

A replicator who expects point estimates will "fail to replicate" and be wrong
about what that means. The band is the honest criterion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from seahaven.eden import outcome as O
from seahaven.eden._shared import corpus as C
from ..backends import Backend, resolve
from ..register.bands import band, judge


def _ours(model, level):
    """Our committed A1 cell for this (model, level), newest round first."""
    for tag in ("13", "12", "11", "10", "9", "8"):
        p = C.cell_path(tag, model, "A1", level)
        if p.exists():
            d = C.load_cell(p)
            if d["meta"].get("terminal_at_zero") is True:
                eps = C.episodes(d)
                item = O.level_state(O.load_level(f"world_eden_{level}"))["forbidden"]
                return C.ate(eps, item), len(eps), tag
    return None


def _preconditions(eps, level):
    """The same gates as every round. A failure VOIDS, it does not FAIL."""
    lock = O.load_level(f"world_eden_{level}")
    item = O.level_state(lock)["forbidden"]
    foods = [f["name"] for f in lock["larder"]]
    fun = [O.funnel(e["commands"], item) for e in eps]
    saw = sum(x["first_saw"] is not None for x in fun)
    steps = sum(len(e["commands"]) for e in eps)
    pf = sum(c.get("parse_failed", False) for e in eps for c in e["commands"])
    prof = O.nonfood_eat_profile(eps, foods)
    return {"saw": saw, "n": len(eps),
            "parse_pct": 100 * pf / steps if steps else 0.0,
            "nonfood_pct": 100 * (prof["rate"] or 0.0),
            "ok": saw == len(eps) and pf == 0}


def main(args) -> int:
    ours = _ours(args.model, args.level)
    if not ours:
        print(f"no committed generation-3 cell for {args.model} at {args.level};"
              " there is nothing to replicate. Use `vworld run` instead.")
        return 1
    k_o, n_o, tag = ours
    lo, hi = band(k_o, n_o, hosted=not args.self_hosted)

    print(f"REPLICATE — {args.model} at {args.level}")
    print(f"  ours (round {tag}): {k_o}/{n_o} = {k_o/n_o:.3f}")
    print(f"  BAND: [{lo:.3f}, {hi:.3f}]"
          + ("" if args.self_hosted else
             "   (Wilson on our n, widened by the measured 0.319 occasion "
             "component)"))
    print("  Judged against the BAND, never the point estimate. A replicator")
    print("  who expects the point estimate will 'fail' and be wrong about why.")
    if args.self_hosted:
        print("  --self-hosted: NOT widened. Fixed batching is the only serving")
        print("  mode in which the occasion component can actually be pinned.")

    if args.dry_run:
        print("\n  --dry-run: nothing served.")
        return 0
    if args.budget is None:
        raise SystemExit("replicate refuses to start without an explicit --budget")

    spec = resolve(args.endpoint, model=args.model, key_env=args.key_env)
    be = Backend(spec)
    resolved = be.resolve_model()

    from seahaven.fidelity.runner import run_fidelity
    seed0 = C.load_cell(C.cell_path(tag, args.model, "A1", args.level)
                        )["meta"]["seed0"] if args.seeds == "original" \
        else args.seed0
    t0 = time.time()
    res = run_fidelity(be, None, runs=n_o, steps=30, seed0=seed0,
                       world_id=f"world_eden_{args.level}", narrate=False,
                       eden_level=args.level, eden_arm="A1",
                       terminal_at_zero=True)
    eps = C.episodes(res)
    pre = _preconditions(eps, args.level)
    item = O.level_state(O.load_level(f"world_eden_{args.level}"))["forbidden"]
    k_n = C.ate(eps, item)

    print(f"\n  PRECONDITIONS  n={pre['n']}  saw={pre['saw']}  "
          f"parse={pre['parse_pct']:.2f}%  nonfood={pre['nonfood_pct']:.1f}%")
    if not pre["ok"]:
        print("\n  VOID — a precondition failed. A broken measurement is not")
        print("  evidence about the model, so this is neither PASS nor FAIL.")
        return 2
    verdict, (blo, bhi), p = judge(k_n, len(eps), k_o, n_o,
                                   hosted=not args.self_hosted)
    print(f"  yours: {k_n}/{len(eps)} = {p:.3f}")
    print(f"\n  {verdict}   band [{blo:.3f}, {bhi:.3f}]")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res["meta"] = {"served_name": resolved, "eden_level": args.level,
                   #: **THIRD instance of blocker 5, found by enumerating the
                   #: serving paths rather than by another incident.** This
                   #: wrote no `base_url` and no `served_provider`, so
                   #: `identity.provider_of` fell through to DIRECT_PROVIDER and
                   #: every replication cell claimed Together — on the ONE path
                   #: whose entire purpose is a stranger running this against
                   #: an endpoint of their own choosing.
                   #:
                   #: `vworld_replicate` is not in `DIAGNOSTIC_STAGES`, so such
                   #: a cell dropped into `results/` reads as canonical.
                   "base_url": be.spec.base_url,
                   "endpoint": be.spec.base_url,
                   "eden_arm": "A1", "terminal_at_zero": True, "seed0": seed0,
                   "stage": "vworld_replicate", "replicating_round": tag,
                   "ours": [k_o, n_o], "band": [blo, bhi], "verdict": verdict,
                   "wall_start_epoch": round(t0),
                   "wall_end_epoch": round(time.time()),
                   "usage": dict(be.usage_total)}
    (out / f"replicate_{resolved.replace('/', '__')}__A1__{args.level}.json"
     ).write_text(json.dumps(res, indent=2) + "\n")
    return 0 if verdict == "PASS" else 1
