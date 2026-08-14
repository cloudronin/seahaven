"""`vworld` — one front door to VetoWorld.

**The command and the package point at each other**, which the previous name did
not: `pip install vetoworld` used to hand you `expdx`. `vw` lost to Vowpal
Wabbit's binary and `veto` to a package already on PyPI — see `docs/naming.md`.

Modelled on `seahaven/fidelity/cli.py`, the one place in the repo that already
did argparse subcommands properly. Fourteen scripts used raw `sys.argv` checks
like `"--comp" in sys.argv`, so none had `--help` and a mistyped flag silently ran
the other path and spent real money.

**The $0 verbs must run with no key at all.** A command that costs nothing cannot
require credentials, and that is asserted rather than intended.
"""

from __future__ import annotations

import argparse
import sys

#: Verbs that must never require a credential.
FREE_VERBS = ("verify", "worlds", "read", "emit", "seeds", "doctor",
              "pin check", "corpus")

#: Verbs that serve episodes and therefore cost money.
SPENDING_VERBS = ("run", "replicate", "probe")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vworld",
        description="VetoWorld — a benchmark of expedience under terminal "
                    "stakes. Recompute the paper, or measure a model.")
    p.add_argument("--version", action="store_true",
                   help="print the version and exit")
    sub = p.add_subparsers(dest="verb", metavar="VERB")

    r = sub.add_parser("read", help="the standing reads on any corpus ($0)")
    r.add_argument("--results", default="results",
                   help="corpus directory (default: results)")
    r.add_argument("--level", action="append",
                   help="restrict to a world; repeatable")
    r.add_argument("--generation", choices=("gen1", "gen2", "gen3"),
                   help="restrict to one generation; never pools across them")
    r.add_argument("--model", action="append", help="restrict to a model")
    r.add_argument("--diagnostics", action="store_true",
                   help="include diagnostic blocks (occasion probe, stability "
                        "blocks, timing probes). Excluded by default: pooling "
                        "them into a measurement averages away the effect they "
                        "were served to detect")

    w = sub.add_parser("worlds", help="the world-validation gates ($0)")
    w.add_argument("--level", action="append",
                   help="world to validate; default all authored")

    s = sub.add_parser("seeds", help="the burned-block registry ($0)")
    s.add_argument("--check", nargs=2, type=int, metavar=("START", "COUNT"),
                   help="check a proposed block against every cell on disk")
    s.add_argument("--model", help="narrow to one model (seed space is per model)")
    s.add_argument("--level", help="narrow to one world")

    e = sub.add_parser("emit", help="render a register artifact ($0)")
    e.add_argument("artifact", help="matrix | occasions | seeds | spend | ...")

    sub.add_parser("doctor", help="environment, corpus and pin health ($0)")
    sub.add_parser("verify", help="recompute every manuscript figure ($0)")

    pr = sub.add_parser("probe", help="pre-flight an endpoint (serves 1 turn)")
    pr.add_argument("endpoint", help="named entry in endpoints.toml, or a base URL")
    pr.add_argument("--model", help="model string (required with a base URL)")
    pr.add_argument("--key-env", help="env var holding the key")

    rn = sub.add_parser("run", help="measure a NEW model (serves cells)")
    rn.add_argument("endpoint", help="named entry in endpoints.toml, or a base URL")
    rn.add_argument("--model", help="model string (required with a base URL)")
    rn.add_argument("--key-env", help="env var holding the key")
    rn.add_argument("--level", default="LAT", help="world (default LAT)")
    rn.add_argument("--arm", default="A1", choices=("A1", "A0"),
                    help="arm for --dry-run")
    rn.add_argument("--m-a1", type=int, default=48, help="A1 episodes")
    rn.add_argument("--m-a0", type=int, default=24, help="A0 episodes")
    rn.add_argument("--round", type=int,
                    help="serve a PINNED round's grid into the corpus; takes "
                         "its cohort, worlds, m and seed0 from the round module")
    rn.add_argument("--seed0", type=int,
                    help="seed block start; check it with `vworld seeds --check`. "
                         "Not used with --round, which carries its own.")
    rn.add_argument("--out", default="results_run", help="output directory")
    rn.add_argument("--budget", type=float,
                    help="USD ceiling; run refuses to start without one")
    rn.add_argument("--dry-run", action="store_true",
                    help="assemble and print the first request; serve nothing")

    rp = sub.add_parser("replicate", help="re-serve our cells, judged vs bands")
    rp.add_argument("endpoint")
    rp.add_argument("--model", required=True)
    rp.add_argument("--level", default="LAT")
    rp.add_argument("--key-env")
    rp.add_argument("--seeds", choices=("fresh", "original"), default="fresh")
    rp.add_argument("--seed0", type=int, default=30000)
    rp.add_argument("--self-hosted", action="store_true",
                    help="do not widen by the occasion component; fixed batching "
                         "is the only mode where it can be pinned")
    rp.add_argument("--budget", type=float)
    rp.add_argument("--out", default="results_replicate")
    rp.add_argument("--dry-run", action="store_true")

    cp = sub.add_parser("corpus", help="fetch/checksum the committed cells ($0)")
    cp.add_argument("action", choices=("fetch", "status", "manifest"))
    cp.add_argument("--results", default="results")
    cp.add_argument("--repo", default=None,
                    help="dataset to fetch from (default: the published one)")
    cp.add_argument("--force", action="store_true",
                    help="replace a corpus that is already on disk")

    pn = sub.add_parser("pin", help="the pin lifecycle ($0 for check)")
    pn.add_argument("action", choices=("check", "new", "retire"))
    pn.add_argument("--round", type=int, help="round number for new/retire")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__
        print(__version__)
        return 0
    if not args.verb:
        parser.print_help()
        return 1
    if getattr(args, "repo", None) is None and args.verb == "corpus":
        from .commands.corpus import DATASET
        args.repo = DATASET

    from .commands import (corpus, doctor, emit, pin, probe, read, replicate,
                           run, seeds, verify, worlds)
    return {"read": read.main, "worlds": worlds.main, "seeds": seeds.main,
            "emit": emit.main, "doctor": doctor.main, "verify": verify.main,
            "probe": probe.main, "run": run.main,
            "pin": pin.main, "replicate": replicate.main,
            "corpus": corpus.main}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
