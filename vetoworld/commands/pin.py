"""`vworld pin` — the pin lifecycle, done by hand eleven times.

**This is where "never re-pin to make it green" stops being a habit.**

    pin check    recompute every pin, live and retired. $0, no key.
    pin new      compute a round's hash and refuse if the tree is dirty
    pin retire   close a round on the round-6 pattern

`pin new` refusing on a dirty tree is the point: a pin is a claim about what the
code was when the cells were served, and computing it over uncommitted edits
makes that claim false at the moment it is written.
"""

from __future__ import annotations

import importlib
import subprocess

ROUNDS = (2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def _mod(n):
    return importlib.import_module(f"seahaven.eden.round{n}")


def _dirty() -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain"],
                       capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _check() -> int:
    bad = 0
    print(f"  {'round':<9}{'state':<32}retired digests")
    for n in ROUNDS:
        m = _mod(n)
        try:
            m.assert_pinned()
            state = "OPEN, verifies"
        except SystemExit as e:
            state = ("CLOSED (refuses, as designed)"
                     if "CLOSED" in str(e) else "** BROKEN **")
            bad += "BROKEN" in state
        notes = []
        for a in dir(m):
            if a.startswith("retired_") and callable(getattr(m, a)):
                want = {getattr(m, c) for c in dir(m)
                        if c.startswith("RETIRED_")
                        and isinstance(getattr(m, c), str)
                        and len(getattr(m, c)) == 64}
                ok = getattr(m, a)() in want
                bad += not ok
                notes.append(f"{a}:{'ok' if ok else '** DOES NOT RECOMPUTE **'}")
        print(f"  round{n:<4}{state:<32}{'  '.join(notes)}")
    print(f"\n  {bad} problem(s)")
    if bad:
        print("  A broken pin means the code moved after the cells were served.")
        print("  REVERT the artifact. Do not re-pin to make this green — the")
        print("  literal is the record of what a round was actually run under.")
    return 1 if bad else 0


def _new(round_no: int) -> int:
    m = _mod(round_no)
    dirty = _dirty()
    if dirty:
        print(f"REFUSING: the working tree has {len(dirty)} uncommitted change(s).")
        for ln in dirty[:8]:
            print(f"    {ln}")
        print("\n  A pin is a claim about what the code WAS when cells were")
        print("  served. Computing it over uncommitted edits makes that claim")
        print("  false the moment it is written. Commit first, then pin.")
        return 1
    got = m.current_hash()
    pinned = [a for a in dir(m) if a.startswith("PINNED_")
              and isinstance(getattr(m, a), str)]
    print(f"round {round_no} payload hash\n\n    {got}\n")
    if pinned:
        cur = getattr(m, pinned[0])
        if cur == got:
            print(f"  {pinned[0]} already matches. Nothing to do.")
            return 0
        print(f"  {pinned[0]} currently holds\n    {cur or '(empty)'}")
        print("\n  If this round has ALREADY SERVED CELLS, do not overwrite it —")
        print("  that rewrites the record. If it has not, paste the hash above,")
        print("  commit it BEFORE running any cell, and say so in the message.")
    return 0


def _retire(round_no: int) -> int:
    m = _mod(round_no)
    try:
        m.assert_pinned()
        print(f"round {round_no} is OPEN and verifies.")
    except SystemExit as e:
        if "CLOSED" in str(e):
            print(f"round {round_no} is already CLOSED.")
            return 0
        print(f"round {round_no}'s pin is BROKEN, which is not the same as "
              "retired.")
    print("\n  RETIRE, on the round-6 pattern:")
    print("   1. `assert_pinned` raises unconditionally with a reason.")
    print("   2. The digest survives as a frozen literal recomputable from")
    print("      frozen constants — `retired_*_hash()` against `RETIRED_*_PIN`.")
    print("   3. tests/test_pin_invariance.py gains the new retired digest.")
    print("\n  A retired round's cells stay valid; only the ability to ADD cells")
    print("  is lost. That is the whole point of retiring rather than deleting.")
    return 0


def main(args) -> int:
    if args.action == "check":
        return _check()
    if args.round is None:
        print(f"`pin {args.action}` needs --round N")
        return 1
    return (_new if args.action == "new" else _retire)(args.round)
