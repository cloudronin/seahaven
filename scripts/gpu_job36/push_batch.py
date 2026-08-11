"""Push a model's whole cell set in ONE commit.

**This is the fix for the defect that cost Phase 1a twenty cells.** `push.py`
uploads one file per commit, so an 18-model sweep makes 540 commits against a
single dataset branch, and the commit endpoint starts returning 500s under that
load — the same failure that took four of Google's six uploads in `gpu_job18`,
which was blamed on concurrency there but happens with serial pushes too.

`upload_folder` with `allow_patterns` batches every matching file into a single
commit. Eighteen commits instead of five hundred and forty.

Retries still exist, because a batched commit can still fail transiently — but
now a failure loses a retryable batch rather than a scattering of individual
cells that have to be diffed against the Hub to discover.

    python push_batch.py <folder> <glob>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("A1B_REPO", "cloudronin/seahaven-a1b-results")


def main() -> int:
    folder, pattern = Path(sys.argv[1]), sys.argv[2]
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"[push] nothing matching {pattern} in {folder}", flush=True)
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for attempt in range(4):
        try:
            api.upload_folder(
                folder_path=str(folder), repo_id=REPO, repo_type="dataset",
                allow_patterns=[pattern],
                commit_message=f"phase1a backfill: {pattern} ({len(files)} cells)")
            print(f"[push] {len(files)} files matching {pattern} -> {REPO} "
                  f"in ONE commit", flush=True)
            return 0
        except Exception as e:                                 # noqa: BLE001
            print(f"[push] attempt {attempt+1} failed for {pattern}: "
                  f"{type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    print(f"[push] GAVE UP on {pattern} — {len(files)} cells remain unpushed",
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
