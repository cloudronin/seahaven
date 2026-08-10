"""Pull result files from the Hub into `results/`.

Every GPU job pushes each result file the moment it is produced, because the
job log is not a reliable progress channel. Getting them back down has been an
ad-hoc `hf download` invocation in each session, which is exactly the kind of
step that quietly differs between runs — so it lives here now.

**Only adds.** Local files are never overwritten and never deleted: `results/`
holds analysis output alongside job output, and more than one glob in this
project has already matched its own results (`v3_narration.json`,
`vp_phrasing.json`). Refusing to overwrite makes a stale local copy visible as
a skip rather than silently replacing a file some published number was computed
from.

    python scripts/pull_results.py                # everything missing
    python scripts/pull_results.py 'vp_*Base*'    # one glob
"""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path

REPO = os.environ.get("A1B_REPO", "cloudronin/seahaven-a1b-results")
DEST = Path("results")


def main(patterns: list[str]) -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; export it before running", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    remote = [f for f in api.list_repo_files(REPO, repo_type="dataset")
              if f.endswith(".json")]
    if patterns:
        remote = [f for f in remote
                  if any(fnmatch.fnmatch(f, p) for p in patterns)]

    DEST.mkdir(exist_ok=True)
    got, skipped = 0, 0
    for name in sorted(remote):
        target = DEST / Path(name).name
        if target.exists():
            skipped += 1
            continue
        api.hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset",
                            local_dir=str(DEST))
        got += 1
        print(f"  + {target.name}")

    print(f"\n{got} downloaded, {skipped} already present "
          f"({len(remote)} matched on the Hub)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
