"""Push a result file to the Hub immediately.

The job log is not a reliable channel: a previous run stalled with no visible
output for 37 minutes while the process was still alive, so "no log" and "no
progress" became indistinguishable. Pushing after each phase makes progress
observable from outside the job and means a stall costs only the phase in
flight, not the whole run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = os.environ.get("A1B_REPO", "cloudronin/seahaven-a1b-results")


def main() -> int:
    from huggingface_hub import HfApi

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[push] nothing at {path}", flush=True)
        return 0
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    import time
    for attempt in range(4):
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=REPO, repo_type="dataset")
            print(f"[push] {path.name} -> {REPO}", flush=True)
            return 0
        except Exception as e:
            # The commit endpoint returns 500 when concurrent commits collide.
            # Retrying with backoff is the difference between a lost result and
            # a slow one.
            print(f"[push] attempt {attempt+1} failed for {path.name}: "
                  f"{type(e).__name__}", flush=True)
            time.sleep(3 * (attempt + 1))
    print(f"[push] GAVE UP on {path.name}", flush=True)
    return 0

def _unused(api, path):
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=REPO,
            repo_type="dataset",
        )
        print(f"[push] {path.name} -> {REPO}", flush=True)
    except Exception as exc:  # never let a push failure kill the run
        print(f"[push] FAILED for {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
