"""`vworld bundle` — emit the manuscript data bundle. $0, offline, no key.

The verb is thin on purpose: `register/bundle.py` holds the logic so it can be
tested without going through argparse, and so the bundle stays a REGISTER
function like everything else it collects.
"""

from __future__ import annotations


def main(args) -> int:
    from pathlib import Path

    from ..register.bundle import build

    out = Path(getattr(args, "out", None) or "manuscript-data")
    print(f"BUNDLE -> {out}\n")
    failed = build(out)
    print(f"\n  $0. Nothing served, no key read, no network touched.")
    return 1 if failed else 0
