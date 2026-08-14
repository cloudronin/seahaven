"""`vworld corpus` — fetch and checksum the committed cells. $0 after the fetch.

The cells are distributed separately from the code: 107 MB of episodes against a
4 MB wheel. `verify` needs them and refuses without them rather than reporting
every figure as drifted.

**The manifest is the point, not the download.** A corpus that cannot be
checksummed is a corpus a replicator cannot trust, so `corpus status` reports the
digest of what is on disk, `corpus manifest` writes the one the paper cites, and
`corpus fetch` installs nothing that does not match it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from seahaven.eden._shared import corpus as C

MANIFEST = Path("corpus.manifest.json")

#: The published dataset. **Not pushed as of this release** — `fetch` says so
#: plainly rather than surfacing a 404 the user has to interpret.
DATASET = "vetoworld/vetoworld-corpus"
_API = "https://huggingface.co/api/datasets/{repo}/tree/main/results"
_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


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


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vworld"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _listing(repo: str) -> list[str]:
    """Cell filenames under the dataset's `results/`, from the public tree API.

    **No `huggingface_hub` dependency.** This package declares no runtime
    dependencies at all, deliberately, so it installs anywhere a model is being
    served. A fetch that dragged in an SDK would undo that for the one verb a
    replicator runs first — and `urllib` is already how the endpoint client
    talks to the wire, for the same reason.
    """
    rows = json.loads(_get(_API.format(repo=repo)))
    return sorted(r["path"].split("/")[-1] for r in rows
                  if r.get("type") == "file"
                  and r["path"].split("/")[-1].startswith("eden_e")
                  and r["path"].endswith(".json"))


def _fetch(args) -> int:
    repo, root = args.repo, Path(args.results)
    have = C.corpus_present(root) if root.exists() else 0
    if have and not args.force:
        print(f"{have} cell(s) already in {root.resolve()}.")
        print("  Refusing to overwrite. Check them with `vworld corpus status`,")
        print("  or pass --force to replace them.")
        return 1

    print(f"fetching {repo} -> {root.resolve()}")
    try:
        names = _listing(repo)
    except urllib.error.HTTPError as e:
        # **HuggingFace answers 401 for a dataset it will not show you, and it
        # does so whether the dataset is private or does not exist at all** — it
        # deliberately does not leak existence. Verified against a real
        # nonsense repo, which also returns 401. So 401/403/404 are one case
        # here; coding only to 404 produced a bare "Unauthorized" for the state
        # this command is in today, which tells a replicator nothing.
        if e.code in (401, 403, 404):
            print(f"\n  CANNOT READ {repo} (HTTP {e.code}).")
            print("  Either it does not exist or it is private — HuggingFace")
            print("  answers the same way for both. As of this release the")
            print("  corpus is not published: use the repository's own")
            print("  `results/` directory, or --repo <owner>/<name>.")
            return 2
        print(f"\n  HTTP {e.code} listing {repo}: {e.reason}")
        return 2
    except urllib.error.URLError as e:
        print(f"\n  cannot reach huggingface.co: {e.reason}")
        return 2

    if not names:
        print(f"\n  {repo} has no cells under results/. Wrong dataset?")
        return 2

    # **Stage, then swap.** An interrupted fetch that left half a corpus in
    # place would make `verify` report drifted figures — a statement about the
    # network dressed up as a statement about the manuscript, which is the exact
    # confusion the empty-corpus guard already exists to prevent.
    stage = root.with_name(root.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for i, name in enumerate(names, 1):
        (stage / name).write_bytes(
            _get(_FILE.format(repo=repo, path=f"results/{name}")))
        if i % 25 == 0 or i == len(names):
            print(f"  {i}/{len(names)} cells", flush=True)

    digest, n, total = _digest_corpus(stage)
    want = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if want and want.get("digest") != digest:
        print(f"\n  ** DIGEST MISMATCH **\n    expected {want['digest']}"
              f"\n    fetched  {digest}")
        print("  What was fetched is not the corpus this code's manifest names.")
        print(f"  Left in {stage.name}/ rather than installed; nothing replaced.")
        return 1

    if root.exists():
        shutil.rmtree(root)
    stage.rename(root)
    print(f"\n  {n} cells, {total:,} bytes, digest {digest[:16]}…")
    print(f"  manifest {'MATCHES' if want else 'not on disk — UNCHECKED'}")
    print("\n  Now: `vworld verify`")
    return 0


def main(args) -> int:
    if args.action == "fetch":
        return _fetch(args)

    root = Path(args.results)
    if args.action == "status":
        if not root.exists():
            print(f"NO CORPUS at {root.resolve()}")
            print("  The cells ship separately from the code. Get them with")
            print("  `vworld corpus fetch`, then run `vworld verify` here.")
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
