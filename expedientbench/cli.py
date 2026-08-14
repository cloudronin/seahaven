"""`expdx` — one front door.

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
FREE_VERBS = ("verify", "worlds", "read", "emit", "seeds", "doctor")

#: Verbs that serve episodes and therefore cost money.
SPENDING_VERBS = ("run", "replicate", "probe")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="expdx",
        description="expedientbench — recompute the paper, or measure a model.")
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

    from .commands import doctor, emit, read, seeds, verify, worlds
    return {"read": read.main, "worlds": worlds.main, "seeds": seeds.main,
            "emit": emit.main, "doctor": doctor.main,
            "verify": verify.main}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
