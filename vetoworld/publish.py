"""The ONLY module permitted to import the HuggingFace SDK.

`pyproject.toml` declares `dependencies = []` so `vworld corpus fetch` — the
verb a replicator runs first — installs anywhere with nothing behind it. The
daily probe is the one verb that WRITES, and an authenticated upload cannot
reuse `corpus.py`'s hand-rolled unauthenticated urllib path, so the SDK lives
behind the `probe` extra and behind this module.

`test_ONLY_the_publish_module_may_import_the_HF_SDK` names this file. Import
`huggingface_hub` anywhere else and every verb that touches that module breaks
on a base install — including the $0 ones someone runs before they have any
credentials at all.

**The import is deferred into the functions**, not module-level, so importing
`vetoworld.publish` itself is safe without the extra. A caller learns it is
missing from a sentence naming the fix, not from a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["available", "push_day", "read_rows", "DATASET"]

#: Separate from the corpus. The corpus is a frozen artifact verified by digest;
#: this is an append-only log that grows daily and is never digest-checked,
#: because a growing file has no single digest to check against.
DATASET = "cloudronin/vetoworld-occasion-log"


def available() -> tuple[bool, str]:
    """`(ok, reason)` — never raises, so a caller can degrade rather than die."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False, ("huggingface_hub is not installed. It is behind the "
                       "`probe` extra: pip install 'vetoworld[probe]'")
    import os
    if not os.environ.get("HF_TOKEN"):
        return False, "HF_TOKEN is not set; the daily push needs a write token"
    return True, ""


def _api():
    import os

    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def push_day(rows: list[dict], cells: list[Path], *, provider: str, date: str,
             repo: str | None = None, attempts: int = 4) -> str:
    """Upload one provider-day: the row, and the cells behind it.

    **Push is first class.** A failure here fails the job loudly rather than
    leaving a silent local-only day — a weather report with gaps nobody
    announced is worse than one that admits an outage, because the gap reads as
    "nothing happened".

    Retries because the commit endpoint returns 500s under load; the GPU-job
    scripts learned this at 540 commits on one sweep, which is also why this
    uploads a FOLDER in one commit rather than a file at a time.
    """
    import time

    ok, why = available()
    if not ok:
        raise SystemExit(f"cannot push: {why}")

    repo = repo or DATASET
    api = _api()
    stage = Path(f"/tmp/vetoworld-occasion-log/{provider}/{date}")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "row.json").write_text(json.dumps(rows, indent=2, sort_keys=True)
                                    + "\n")
    for c in cells:
        (stage / c.name).write_text(Path(c).read_text())

    last = None
    for attempt in range(attempts):
        try:
            api.upload_folder(
                folder_path=str(stage), repo_id=repo, repo_type="dataset",
                path_in_repo=f"{provider}/{date}",
                commit_message=f"{provider} {date}: {len(rows)} channel row(s)")
            return f"https://huggingface.co/datasets/{repo}/tree/main/{provider}/{date}"
        except Exception as e:                       # noqa: BLE001
            last = e
            if attempt < attempts - 1:
                time.sleep(3 * (attempt + 1))
    raise SystemExit(f"push failed after {attempts} attempts: {last}")


def read_rows(provider: str = "together", repo: str | None = None) -> list[dict]:
    """Every row on record for a provider, oldest first.

    Reads through the SDK because the log is private-capable and paginated;
    `corpus.py`'s unauthenticated urllib path deliberately cannot do that.
    """
    ok, why = available()
    if not ok:
        raise SystemExit(f"cannot read the log: {why}")

    from huggingface_hub import hf_hub_download
    api = _api()
    repo = repo or DATASET
    files = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f.startswith(f"{provider}/") and f.endswith("/row.json")]
    out: list[dict] = []
    for f in sorted(files):
        local = hf_hub_download(repo, f, repo_type="dataset")
        out.extend(json.loads(Path(local).read_text()))
    return out
