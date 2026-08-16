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
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import identity as ID

#: **The manifest ships WITH THE CODE, not with the corpus.**
#:
#: It is the claim the code makes about which cells it expects, so it has to
#: arrive from the other side of the check. A manifest downloaded alongside the
#: corpus would have the corpus vouching for itself, which is worth nothing.
#:
#: It lived at the repo root and shipped in neither the wheel nor the dataset, so
#: `pip install vetoworld && vworld corpus fetch` printed `manifest not on disk —
#: UNCHECKED`: the digest check that is the entire point of `fetch` silently did
#: not happen, for exactly the stranger who needed it most.
MANIFEST = Path(__file__).resolve().parent.parent / "corpus.manifest.json"

#: The published dataset. Its digest is `corpus.manifest.json`, and `fetch`
#: installs nothing that does not match it.
DATASET = "cloudronin/vetoworld-corpus"
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


class TransientFetchError(RuntimeError):
    """A network failure that survived every retry. Distinct from an HTTP status
    because the fix is different: wait and re-run, rather than check the name."""


#: Transient conditions worth another attempt. 429 is rate limiting, 5xx is the
#: far end failing; both are about the moment, not the request.
_RETRY_CODES = (408, 429, 500, 502, 503, 504)
_ATTEMPTS = 5


def _get(url: str, timeout: int = 60) -> bytes:
    """One file, with backoff.

    **A fetch is 259 sequential requests, so transient failure is the normal
    case rather than the exceptional one.** The first version had no retry and
    died on `RemoteDisconnected` partway through a real download — a stack trace
    where a replicator needed a progress bar. `Retry-After` is honoured because
    the far end saying "wait this long" is better information than any backoff
    curve computed here, which is the same rule the serving client follows.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "vworld"})
    delay = 1.0
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_CODES or attempt == _ATTEMPTS:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as e:
            if attempt == _ATTEMPTS:
                raise TransientFetchError(
                    f"{type(e).__name__}: {e} — after {_ATTEMPTS} attempts") from e
            wait = delay
        time.sleep(wait)
        delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


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
            print("  answers the same way for both, so this is not evidence")
            print("  about which. Check the name, or pass --repo <owner>/<name>.")
            return 2
        print(f"\n  HTTP {e.code} listing {repo}: {e.reason}")
        return 2
    except urllib.error.URLError as e:
        print(f"\n  cannot reach huggingface.co: {e.reason}")
        return 2
    except TransientFetchError as e:
        print(f"\n  could not list {repo}: {e}")
        return 2

    if not names:
        print(f"\n  {repo} has no cells under results/. Wrong dataset?")
        return 2

    # **Stage, then swap.** An interrupted fetch that left half a corpus in
    # place would make `verify` report drifted figures — a statement about the
    # network dressed up as a statement about the manuscript, which is the exact
    # confusion the empty-corpus guard already exists to prevent.
    stage = root.with_name(root.name + ".partial")
    stage.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in stage.glob("eden_e*.json")}
    if done:
        print(f"  resuming: {len(done)} cell(s) already staged")
        names = [n for n in names if n not in done]
    print(f"  {len(names)} cells to pull")
    for i, name in enumerate(names, 1):
        try:
            (stage / name).write_bytes(
                _get(_FILE.format(repo=repo, path=f"results/{name}")))
        except (TransientFetchError, urllib.error.HTTPError) as e:
            # **Report the file and keep the partial.** A traceback from inside
            # urllib names a socket, not a cell, and deleting what was already
            # pulled would make a re-run start from zero on a connection that
            # has already shown it is unreliable.
            print(f"\n  FETCH FAILED at cell {i}/{len(names)} ({name})")
            print(f"    {e}")
            print(f"  {i - 1} cell(s) kept in {stage.name}/. Re-run to resume;")
            print("  nothing was installed, so `verify` still sees your old copy.")
            return 2
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


def _identity(args) -> int:
    """Report the served-model status of every cell. $0, read-only.

    Exists so the precondition has a standing surface rather than living only
    in a test: a rule nobody can look up is a rule that gets forgotten, which
    is the whole story of `resolved_model_string`.
    """
    root = Path(args.results)
    rows = [(p, C.load_cell(p).get("meta", {})) for p in
            sorted(root.glob("eden_e*.json"))]
    counts = ID.tally(m for _p, m in rows)

    print(f"MODEL IDENTITY — {len(rows)} cells at {root.resolve()}\n")
    print("  Was each cell served by the model it claims? The endpoint's own")
    print("  report is the only evidence; an argument about how the runner")
    print("  ought to behave is not evidence.\n")
    for st in (ID.VERIFIED, ID.CORRECTED, ID.UNVERIFIED, ID.MISLABELLED):
        print(f"  {counts[st]:>4}  {st}")

    bad = [p for p, m in rows if ID.model_identity(m).status == ID.MISLABELLED]
    if bad:
        print(f"\n  ** {len(bad)} cell(s) claim a model that did not serve "
              "them and are NOT corrected. **")
        for p in bad[:10]:
            print(f"     {p.name}")
        if len(bad) > 10:
            print(f"     ... and {len(bad) - 10} more")
        print("\n  Run `vworld corpus relabel` to record what actually served.")
        return 1

    if counts[ID.UNVERIFIED]:
        print(f"\n  {counts[ID.UNVERIFIED]} cell(s) predate the served-model "
              "record. Not a failure —")
        print("  but NOT a clean bill either. The check cannot be run on them "
              "at all,")
        print("  and no read may quietly promote that to a pass.")
    return 0


def _relabel(args) -> int:
    """Record what actually served, keeping what was asked for beside it.

    **Relabel rather than delete.** The affected cells are valid data about the
    model that really served them — including the only identical-input
    repeatability study the programme has ever had, which arrived by accident
    when eight "models" turned out to be one. Deleting them would discard a
    measurement to hide a mistake.

    The filenames are deliberately NOT corrected: eight of round 18's cells
    would collide on one name, and the manifest and pins address cells by name.
    A filename is now a record of a REQUEST.

    Idempotent: re-running finds nothing to do.
    """
    root = Path(args.results)
    todo = []
    for p in sorted(root.glob("eden_e*.json")):
        payload = C.load_cell(p)
        ident = ID.model_identity(payload.get("meta", {}))
        if ident.status == ID.MISLABELLED:
            todo.append((p, payload, ident))

    if not todo:
        print("Nothing to relabel: no cell claims a model that did not serve "
              "it.")
        return 0

    print(f"RELABEL — {len(todo)} cell(s) whose served model was not what the "
          "name claims\n")
    by_served: dict[str, int] = {}
    for _p, _pl, ident in todo:
        by_served[ident.served] = by_served.get(ident.served, 0) + 1
    for served, n in sorted(by_served.items()):
        print(f"  {n:>4} cells actually served by {served}")

    if getattr(args, "dry_run", False):
        print("\n  --dry-run: nothing written.")
        return 0

    for p, payload, ident in todo:
        meta = payload["meta"]
        meta["requested_model"] = ident.requested
        meta["served_model"] = ident.served
        meta[ID.CORRECTION_KEY] = {
            "issue": ID.ISSUE,
            "requested_model": ident.requested,
            "served_model": ident.served,
            "why": "The runner built one Backend for a whole grid; Backend "
                   "fixes the request's model field at construction, so the "
                   "grid's model tuple selected a filename and a price, never "
                   "a served model. The filename still spells the request.",
        }
        #: `indent=2`, no `sort_keys` — exactly how `run.py` writes a cell. The
        #: new keys append to `meta`, so the diff is the correction and
        #: nothing else. Reformatting 166 large cells would bury it.
        p.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  relabelled {len(todo)} cells.")
    print("  `served_model` is now the measurement identity; "
          "`requested_model` is kept")
    print("  as the record of the bug. Regenerate the manifest with "
          "`vworld corpus manifest`.")
    return 0


def main(args) -> int:
    if args.action == "fetch":
        return _fetch(args)
    if args.action == "identity":
        return _identity(args)
    if args.action == "relabel":
        return _relabel(args)

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
