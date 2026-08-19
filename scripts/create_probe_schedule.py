"""Create the daily probe schedule, with every parameter DERIVED FROM THE PIN.

Nothing here is retyped. `schedule`, `flavor`, `timeout`, `secrets` and the
column order all come out of `probe.SCHEDULE_JOB`, which is in the hashed
payload — so the live job cannot quietly disagree with what was pre-registered.
Retyping them would be one more "two copies drift" instance, on the one object
that spends money unattended.

The version is read from `pyproject.toml` and PINNED in the install, so a later
PyPI release cannot silently change what the schedule runs.
"""
import os
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

#: The repo root, derived from this file's location — not a machine path.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from huggingface_hub import (  # noqa: E402
    create_scheduled_job,
    inspect_scheduled_job,
    HfApi,
)

from seahaven.eden import probe as PB  # noqa: E402

VERSION = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
JOB = PB.SCHEDULE_JOB


def build_command() -> list[str]:
    """Install the pinned release, then each column in the PINNED order.

    **Each column runs independently.** Chaining with `&&` would let a Together
    outage cancel the DeepInfra day, which turns one provider's bad afternoon
    into a missing row for BOTH columns — and the coincidence read is precisely
    a statement about the two columns on the SAME day. The job still exits
    non-zero if either failed, so a partial day is visible rather than silent.
    """
    lines = [
        "set -u",
        f"pip install --no-cache-dir --quiet 'vetoworld[probe]=={VERSION}' || exit 1",
        "vworld --version",
        #: Refuse loudly if the container cannot serve, rather than discovering
        #: it once per cell. The CLI guards this too; belt and braces because
        #: this is the unattended path.
        "python -c 'from vetoworld.commands._serving_deps import require;"
        " require(\"scheduled day\")' || exit 1",
        "rc=0",
    ]
    for column in JOB["columns_in_order"]:
        lines.append(f"vworld probe {column} --daily || rc=1")
    lines += ['echo "DAY_RC=$rc"', "exit $rc"]
    return ["bash", "-c", "\n".join(lines)]


def main() -> int:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    existing = list(api.list_scheduled_jobs())
    if existing:
        print(f"REFUSING: {len(existing)} scheduled job(s) already exist.")
        for j in existing:
            print("  ", j)
        print("The plan says DELETE AND RECREATE, never edit. Delete first.")
        return 1
    print("pre-condition OK: zero scheduled jobs")

    command = build_command()
    print("\n--- command ---")
    print(command[2])
    print("--- spec (all from the pin) ---")
    for k in ("schedule", "flavor", "timeout"):
        print(f"  {k:9} {JOB[k]!r}")
    print(f"  image     {JOB['image']}")
    print(f"  secrets   {sorted(JOB['secrets'])}")
    print(f"  version   {VERSION}")

    secrets = {name: os.environ[name] for name in JOB["secrets"]}
    missing = [n for n, v in secrets.items() if not v]
    if missing:
        print(f"REFUSING: empty secret(s) {missing}")
        return 1

    #: **Derived from the pin like everything else, since 0.3.2.** It was the
    #: one job parameter still retyped here — and the 0.3.0 outage was caused
    #: by exactly that: the image could not build `jericho`, which publishes an
    #: sdist and no wheels, so pip compiles it and needs make plus a C
    #: toolchain. Pinning the PACKAGE never protected against it, because the
    #: failure was in what the package was installed INTO.
    job = create_scheduled_job(
        image=JOB["image"],
        command=command,
        schedule=JOB["schedule"],
        flavor=JOB["flavor"],
        timeout=JOB["timeout"],
        secrets=secrets,
        token=token,
    )
    print(f"\ncreated: {job.id}")

    info = inspect_scheduled_job(scheduled_job_id=job.id, token=token)
    print("\n--- verification, read back from the API ---")
    for field in ("id", "status", "schedule", "suspend", "next_job_run_at",
                  "last_job_run_at"):
        print(f"  {field:16} {getattr(info, field, '(absent)')!r}")

    jobs = list(api.list_scheduled_jobs())
    print(f"\nscheduled jobs now: {len(jobs)}")

    #: The pinned cron must be what the API echoes back, or the schedule that
    #: runs is not the schedule that was pre-registered.
    echoed = getattr(info, "schedule", None)
    if echoed != JOB["schedule"]:
        print(f"MISMATCH: pinned {JOB['schedule']!r}, API says {echoed!r}")
        return 1
    print(f"cron matches the pin: {echoed!r}")

    #: **The image must be digest-pinned, or the container is the one input
    #: nothing records.** `python:3.12` is a FLOATING tag: Docker rebuilds it
    #: for point releases and security patches, so a container that passes a
    #: pre-flight is not the one that fires tomorrow.
    if "@sha256:" not in JOB["image"]:
        print(f"REFUSING: the pinned image {JOB['image']!r} is a floating tag. "
              "Pin it by digest — every other input to this job is hashed or "
              "content-addressed, and the container should not be the "
              "exception.")
        return 1
    print(f"image is digest-pinned: {JOB['image'].split('@')[1][:23]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
