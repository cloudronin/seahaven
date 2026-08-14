"""`expdx corpus` — fetch and checksum the committed cells. $0 after the fetch.

The cells are distributed separately from the code: 247 MB of episodes against a
4 MB wheel. `verify` needs them and refuses without them rather than reporting
every figure as drifted.

**The manifest is the point, not the download.** A corpus that cannot be
checksummed is a corpus a replicator cannot trust, so `corpus status` reports the
digest of what is on disk and `corpus manifest` writes the one the paper cites.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from seahaven.eden._shared import corpus as C

MANIFEST = Path("corpus.manifest.json")


def _digest_corpus(root: Path) -> tuple[str, int, int]:
    """`(digest, cells, bytes)` — order-independent and content-addressed."""
    h = hashlib.sha256()
    n = total = 0
    for p in sorted(root.glob("eden_e*.json")):
        b = p.read_bytes()
        h.update(p.name.encode())
        h.update(hashlib.sha256(b).digest())
        n += 1
        total += len(b)
    return h.hexdigest(), n, total


def main(args) -> int:
    root = Path(args.results)
    if args.action == "status":
        if not root.exists():
            print(f"NO CORPUS at {root.resolve()}")
            print("  The cells ship separately from the code. Fetch them, then")
            print("  run `expdx verify` from a directory that contains them.")
            return 2
        digest, n, total = _digest_corpus(root)
        print(f"corpus at {root.resolve()}")
        print(f"  cells   {n}")
        print(f"  bytes   {total:,}")
        print(f"  digest  {digest}")
        if MANIFEST.exists():
            want = json.loads(MANIFEST.read_text())
            same = want.get("digest") == digest
            print(f"\n  manifest {'MATCHES' if same else '** DIFFERS **'}"
                  f"  ({want.get('cells')} cells, {want.get('digest','')[:16]}…)")
            if not same:
                print("  Your corpus is not the one the manuscript was computed")
                print("  from. Re-fetch, or expect `verify` to disagree.")
                return 1
        else:
            print("\n  no manifest on disk; write one with `corpus manifest`")
        return 0

    digest, n, total = _digest_corpus(root)
    MANIFEST.write_text(json.dumps(
        {"digest": digest, "cells": n, "bytes": total,
         "note": "sha256 over each cell's name and content digest, sorted. "
                 "Order-independent, so a re-fetch in any order verifies."},
        indent=2) + "\n")
    print(f"wrote {MANIFEST}: {n} cells, digest {digest[:16]}…")
    print("  The paper cites this digest beside the repo tag.")
    return 0
