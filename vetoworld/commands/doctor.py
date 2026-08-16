"""`vworld doctor` — environment, corpus and pin health. $0, no key."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROVIDERS = {"TOGETHER_API_KEY": "Together", "OPENAI_API_KEY": "OpenAI"}


def main(_args=None) -> int:
    bad = 0
    print(f"python           {sys.version.split()[0]}")
    # Echoing the user's own path is useful for diagnosing "wrong
    # directory". It is their string, not ours, and is labelled so.
    print(f"working dir      {Path.cwd()}   (your path)")

    n_cells = len(list(Path("results").glob("eden_e*.json"))) \
        if Path("results").exists() else 0
    print(f"corpus           {n_cells} cells"
          + ("" if n_cells else "   ** results/ missing or empty **"))
    bad += not n_cells

    print("\npins")
    # **One canonical list, imported.** This tuple was a second copy and it
    # drifted: it stopped at 13 while `pin.ROUNDS` gained 14, so `doctor` quietly
    # stopped checking the only round that could still serve cells. Third
    # instance of a duplicated vocabulary list going stale, after
    # registry-disjointness and `emit.ARTIFACTS`/`fn` — hence the assertion in
    # `tests/test_vworld_cli.py` rather than only the fix.
    from .pin import ROUNDS, STANDING
    for n in ROUNDS:
        try:
            m = importlib.import_module(f"seahaven.eden.round{n}")
        except Exception as e:
            print(f"  round{n:<4}IMPORT FAILED: {e}")
            bad += 1
            continue
        try:
            m.assert_pinned()
            state = "OPEN, verifies"
        except SystemExit as e:
            state = ("CLOSED (refuses, as designed)"
                     if "CLOSED" in str(e) else "** BROKEN **")
            bad += "BROKEN" in state
        retired = [a for a in dir(m) if a.startswith("retired_")
                   and callable(getattr(m, a))]
        extra = ""
        for fn in retired:
            want = {a: getattr(m, a) for a in dir(m) if a.startswith("RETIRED_")
                    and isinstance(getattr(m, a), str) and len(getattr(m, a)) == 64}
            got = getattr(m, fn)()
            ok = got in want.values()
            extra += f"  {fn}:{'ok' if ok else '** DOES NOT RECOMPUTE **'}"
            bad += not ok
        print(f"  round{n:<4}{state}{extra}")

    # **Standing instruments are pins too.** The daily probe has no round
    # number, so a loop over `ROUNDS` alone would leave the one pin that a
    # scheduled job depends on unwatched by the health check people run first.
    for name in STANDING:
        try:
            m = importlib.import_module(f"seahaven.eden.{name}")
            m.assert_pinned()
            state = "OPEN, verifies"
        except SystemExit as e:
            state = ("CLOSED (refuses, as designed)"
                     if "CLOSED" in str(e) else "** BROKEN **")
            bad += "BROKEN" in state
        except Exception as e:
            state, bad = f"IMPORT FAILED: {e}", bad + 1
        print(f"  {name:<9}{state}")

    print("\nprovider keys visible (never printed, only presence)")
    for env, name in PROVIDERS.items():
        print(f"  {name:<10}{'yes' if os.environ.get(env) else 'no'}")
    print("\n  The $0 verbs — verify, worlds, read, emit, seeds, doctor — do not")
    print("  use these. A command that costs nothing must not need credentials.")
    print(f"\n{bad} problem(s)")
    return 1 if bad else 0
