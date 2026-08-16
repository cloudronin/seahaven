"""Refuse to run at all in an interpreter that cannot play a world.

**This exists because 83 tests silently did not run, and nobody noticed for a
whole session.**

`textworld` and `jericho` are not declared in `pyproject.toml` — the package
deliberately ships `dependencies = []` so `vworld corpus fetch` installs
anywhere. They live in `env/vetoworld-dev.yml` and are installed by
`scripts/setup_dev_env.sh`. Run the suite in any other interpreter and the seven
modules that open a compiled world fail to import.

Without this file, that presents as **83 collection errors reading
`ModuleNotFoundError: No module named 'textworld'`** — which looks exactly like
an optional dependency nobody needs. It was read that way three times in three
separate reports, and the "full suite" those reports certified was missing a
third of itself. Two commits and a published retraction went out on it.

The gap was not cosmetic: the same missing module means the machine cannot serve
a single episode. That is how it was finally caught — a paid probe run failed on
all 17 cells, having spent $0.00.

So this does not skip, warn, or mark xfail. **A partial suite that looks whole is
worse than no suite**, because it produces a green number that certifies
nothing, and the whole point of this repo's discipline is not to do that.
"""

from __future__ import annotations

import shutil
import sys

import pytest

#: The environment `scripts/setup_dev_env.sh` builds. Named here so the error
#: message can say what to do rather than only what is wrong.
ENV_NAME = "vetoworld-dev"

#: Imported by `seahaven/world/loader.py` and `worlds/build_eden_worlds.py`.
#: Both are unavoidable for anything that opens a `.z8`.
REQUIRED = ("textworld", "jericho")


def _missing() -> list[str]:
    import importlib.util
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def pytest_collection(session):
    """Hard-stop before collection, so the failure is one legible message
    instead of dozens of import tracebacks that read as ignorable."""
    missing = _missing()
    if not missing:
        return

    conda = shutil.which("conda") or "conda"
    raise pytest.UsageError(
        "\n"
        "  WRONG INTERPRETER — the test suite cannot run here.\n"
        "\n"
        f"  Missing: {', '.join(missing)}\n"
        f"  Using  : {sys.executable}\n"
        "\n"
        "  These are not optional. Without them every module that opens a "
        "compiled world\n"
        "  fails to import, ~83 tests never execute, and the run still reports "
        "a pass\n"
        "  for a suite that is missing a third of itself. The same gap means "
        "this\n"
        "  interpreter cannot serve an episode either.\n"
        "\n"
        "  Run the suite the way the README does:\n"
        "\n"
        f"      {conda} run -n {ENV_NAME} python -m pytest\n"
        "\n"
        f"  If {ENV_NAME} does not exist yet, build it with the script — the\n"
        "  CONDA_SUBDIR=osx-arm64 invariant is load-bearing and the README says "
        "not\n"
        "  to create the environment by hand:\n"
        "\n"
        "      bash scripts/setup_dev_env.sh\n")


def test_the_interpreter_can_open_a_world():
    """Belt and braces: if the hook above is ever removed or bypassed, this
    still fails loudly rather than letting a partial suite report green."""
    assert not _missing(), (
        f"{_missing()} unavailable — see tests/conftest.py; run with "
        f"`conda run -n {ENV_NAME} python -m pytest`")
