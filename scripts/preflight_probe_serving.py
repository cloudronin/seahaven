"""Pre-flight that SERVES. The one Monday's could not be.

Monday's pre-flight passed and proved nothing about serving: it ran
`--dry-run`, which returns before the first episode. So it verified install,
pin, secrets and grid derivation, and could not have discovered that opening a
compiled world raises `No module named 'textworld'`. The scheduled job found
that out instead, 22 times, and called it PARTIAL.

**A pre-flight for a serving job has to serve.** Note that the ordinary
`vworld probe <endpoint>` pre-flight would NOT have caught this either: it
sends one chat completion and never opens a world. Only `--daily` runs an
episode, so only `--daily` exercises the thing that broke.

**The DeepInfra column only, five cells, ~$2.40.** A full two-column day is
~$10 and would prove nothing the smaller column does not: both go through the
same `run_fidelity` -> `world/loader` path, and DeepInfra is the newer transport
and the likelier one to surprise. Cheap enough to be worth running before every
schedule change.

Writes to a scratch directory and does NOT push, so the published log stays
untouched by a test. The cells are discarded — this buys confidence, not data.
"""
import os
import sys

from huggingface_hub import fetch_job_logs, inspect_job, run_job

VERSION = "0.3.1"

SCRIPT = f"""
set -u
echo '=== toolchain present? jericho has no wheels and must compile ==='
command -v make gcc || {{ echo '** no build toolchain — jericho cannot build **'; exit 1; }}

echo
echo '=== install the published release WITH the serving extra ==='
pip install --no-cache-dir --quiet 'vetoworld[probe]=={VERSION}' || exit 1
vworld --version

echo
echo '=== the serving deps must actually be present ==='
python - <<'PY'
from vetoworld.commands._serving_deps import missing
gone = missing()
print("missing serving modules:", gone or "none")
import seahaven.world.loader
print("world loader imports — a compiled world can be opened")
raise SystemExit(1 if gone else 0)
PY
[ $? -eq 0 ] || exit 1

echo
echo '=== SERVE THE DEEPINFRA COLUMN FOR REAL (5 cells, ~$2.40) ==='
# --no-push: a pre-flight must never write to the published log.
rc=0
mkdir -p /tmp/preflight/results && cd /tmp/preflight
vworld probe deepinfra --daily --no-push || rc=$?
echo "deepinfra exit=$rc"

echo
echo "cells written:"
ls -1 /tmp/preflight/results/probe-*.json 2>/dev/null | wc -l
echo "expected 5"

echo "PREFLIGHT_RC=$rc"
exit $rc
"""


def main():
    token = os.environ["HF_TOKEN"]
    job = run_job(
        #: **NOT -slim.** `jericho` ships an sdist and NO wheels for any
        #: platform, so pip compiles it from source and needs make + a C
        #: toolchain. `python:3.12-slim` has neither, and the slim image is
        #: what the 0.3.0 schedule used. Found by this pre-flight, which is
        #: the point of it serving rather than dry-running.
        image="python:3.12",
        command=["bash", "-c", SCRIPT],
        secrets={
            "TOGETHER_API_KEY": os.environ["TOGETHER_API_KEY"],
            "HF_TOKEN": token,
            "DEEPINFRA_API_KEY": os.environ["DEEPINFRA_API_KEY"],
        },
        flavor="cpu-basic",
        timeout="3h",
        token=token,
    )
    print(f"job {job.id}", flush=True)
    for line in fetch_job_logs(job_id=job.id, follow=True, token=token):
        print(line, flush=True)
    print("FINAL:", inspect_job(job_id=job.id, token=token).status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
