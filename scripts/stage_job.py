"""Stage a GPU job payload from the code's OWN declared dependencies, then
execute the job's preflight against the staged copy before anything launches.

**Three jobs have now died on an incomplete or stale payload**, each in a way
that looked different and was the same:

1. `gpu_job31` — a stale bucket mount reused an old directory.
2. `gpu_job33` — `seahaven/` staged before `worldspec.SETTINGS` had the
   E-worlds, so every cell died AFTER a 150s model load.
3. `gpu_job34` — `docs/` never staged at all, so `axis2_prereg` could not hash
   its own artifacts and the module failed at import.

The pattern is not carelessness about a particular file. It is that the payload
was assembled from **what I remembered the job needing**, and checked against
**the same memory**. A shasum diff of the files I thought to list cannot find
the file I never thought of.

So this does two things memory cannot:

* **Derives the artifact list from the code.** `axis2_prereg.ARTIFACTS` already
  names every file the pre-registration hashes; it is read, not retyped.
* **Runs the preflight against the staged directory.** The staged copy is put on
  `sys.path` with its own working directory, and the same imports and assertions
  the container will make are made here first. A payload that cannot preflight
  locally cannot preflight remotely, and finding that out costs nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Python packages every job needs. Everything else is derived.
PACKAGES = ("seahaven", "worlds")


def required_artifacts() -> list[str]:
    """Read the hashed artifact list from the module that defines it."""
    sys.path.insert(0, str(ROOT))
    from seahaven.dimensional import axis2_prereg as A

    return list(A.ARTIFACTS)


def stage(job: str, extra: tuple[str, ...] = ()) -> Path:
    dest = ROOT / "scripts" / job
    for pkg in PACKAGES:
        tgt = dest / pkg
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(ROOT / pkg, tgt)
    for rel in list(required_artifacts()) + list(extra):
        src, tgt = ROOT / rel, dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
    for pyc in dest.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    return dest


PREFLIGHT = r"""
import sys, os
from seahaven.dimensional import axis2_prereg as A
from seahaven.dimensional import seal as S
from seahaven.eaxis.levels import assert_level_runnable
from seahaven.fidelity.worldspec import load

S.assert_sealed()
S.assert_not_held_out(S.EXPLORATION)
A.assert_prereg()
print(f"  SEAL   {S.SEAL_HASH[:16]}  {len(S.EXPLORATION)} models")
print(f"  PREREG {A.PREREG_HASH[:16]}")
for w in ("world_ea", "world_eb", "world_ea_E3", "world_eb_E3"):
    spec = load(w)
    print(f"  world {w:<14} start={spec.start_room!r} rooms={len(spec.rooms)}")
for w in ("world_ea", "world_eb"):
    for lv in ("E0", "E1", "E2"):
        assert_level_runnable(w, lv)
for w in ("world_ea_E3", "world_eb_E3"):
    assert_level_runnable(w, "E3")
print("  PROOFS every level/world pair committed")
"""


def verify(dest: Path) -> bool:
    """Run the container's own preflight against the staged copy.

    **Reproduce the container faithfully, which means NOT running from the
    payload directory.** The first version of this ran with `cwd=dest`, which
    seemed right and was the opposite of right: the container sets
    `PYTHONPATH=/app` and never `cd`s there, so a module reading a bare relative
    path works under `cwd=dest` and fails under `/app`. That is precisely what
    happened -- `levels.assert_level_runnable` defaulted to
    `results/e_world_proofs.json`, passed this check, and refused in the
    container.

    So: `PYTHONPATH` points at the payload and the working directory is
    deliberately somewhere else. A check that cannot fail on CWD dependence
    cannot catch CWD dependence.
    """
    env = dict(os.environ, PYTHONPATH=str(dest))
    r = subprocess.run([sys.executable, "-c", PREFLIGHT], cwd=tempfile.gettempdir(),
                       capture_output=True, text=True, env=env)
    print(r.stdout.rstrip() or "(no output)")
    if r.returncode != 0:
        print(r.stderr.rstrip()[-1500:])
    return r.returncode == 0


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else "gpu_job34"
    extra = tuple(sys.argv[2:])
    dest = stage(job, extra)
    print(f"staged {job}: {sum(1 for _ in dest.rglob('*') if _.is_file())} files")
    for rel in required_artifacts():
        print(f"  artifact {rel}  {'OK' if (dest / rel).exists() else 'MISSING'}")
    ok = verify(dest)
    print(f"\nPREFLIGHT {'PASS — safe to launch' if ok else 'FAIL — do not launch'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
