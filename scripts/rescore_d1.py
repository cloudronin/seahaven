"""Phase A — re-score every existing run with the frozen D1 detector.

No new episodes. The 500 rollouts and their narratives are already on disk; only
the *mention* side changes, from a regex to `gpt-4.1-mini` under the exact
prompt, temperature and dual-ask discipline frozen in `prereg-v3` (D1).

**What this answers.** Every number the project currently holds — the Phase 1
table, V2's cross-world correlation, V4's confounds — was computed with a string
matcher that scores **exactly 0.000** on the decisive stratum. If the score is
an artefact of that detector, it will move here. G1, G2a/G2b and G3 are all
decided by this run.

**Dual-ask, and disagreement counts as not-a-claim.** Frozen in v3, so an
unstable judgement cannot be quietly resolved toward whichever answer flatters
the result.

Caching is by (narrative, entity), so re-runs cost nothing and the two worlds'
shared phrasings are not paid for twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from adjudicate_v1 import ask  # noqa: E402

from seahaven.fidelity.endpoint import Endpoint  # noqa: E402

CACHE = Path("results/d1_cache.jsonl")
MODEL = "gpt-4.1-mini"


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["h"]] = d["v"]
    return out


def h_of(narrative: str, entity: str) -> str:
    return hashlib.sha1(f"{entity}||{narrative}".encode()).hexdigest()


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    ep = Endpoint("https://api.openai.com/v1", MODEL, api_key=key)

    files = sorted(Path("results").glob("x_*.json"))
    jobs, seen = [], set()
    for f in files:
        d = json.loads(f.read_text())
        for run in d["runs"]:
            for k in run["acts"]:
                hh = h_of(run["narrative"], k)
                if hh not in seen:
                    seen.add(hh)
                    jobs.append((hh, run["narrative"], k))
    cache = load_cache()
    todo = [j for j in jobs if j[0] not in cache]
    print(f"{len(files)} result files, {len(jobs)} unique (narrative, entity) pairs, "
          f"{len(todo)} uncached")

    if todo:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        lock_out = open(CACHE, "a")

        def one(t):
            hh, nar, k = t
            rel, ent = k.split(":", 1)
            try:
                a = ask(ep, nar, rel, ent, seed=61000 + (hash(hh) % 10000))
                b = ask(ep, nar, rel, ent, seed=88000 - (hash(hh) % 10000))
            except Exception:
                return hh, None
            # Frozen: disagreement counts as NOT a claim.
            return hh, (bool(a) and bool(b) if a is not None and b is not None else None)

        done = 0
        with ThreadPoolExecutor(max_workers=14) as pool:
            for hh, v in pool.map(one, todo):
                cache[hh] = v
                lock_out.write(json.dumps({"h": hh, "v": v}) + "\n")
                done += 1
                if done % 500 == 0:
                    lock_out.flush()
                    print(f"  {done}/{len(todo)}", flush=True)
        lock_out.close()

    failed = sum(1 for v in cache.values() if v is None)
    print(f"cache holds {len(cache)} judgements, {failed} unresolved")

    # Rewrite each result file with D1 mentions. Ground truth is untouched.
    out_dir = Path("results/d1")
    out_dir.mkdir(exist_ok=True)
    n_flip = n_tot = 0
    for f in files:
        d = json.loads(f.read_text())
        for run in d["runs"]:
            for k, v in run["acts"].items():
                m = cache.get(h_of(run["narrative"], k))
                if m is None:
                    continue
                n_tot += 1
                n_flip += int(bool(m) != bool(v["mentioned"]))
                v["mentioned"] = bool(m)
        d["meta"]["detector"] = f"D1:{MODEL}"
        (out_dir / f.name).write_text(json.dumps(d) + "\n")
    print(f"wrote {len(files)} files to {out_dir}/")
    print(f"D1 disagrees with the regex on {n_flip}/{n_tot} "
          f"({100 * n_flip / max(1, n_tot):.1f}%) of entity judgements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
