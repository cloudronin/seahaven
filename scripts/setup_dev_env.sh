#!/usr/bin/env bash
# Create the local development environment.
#
# CONDA_SUBDIR=osx-arm64 is load-bearing and is the reason this script exists.
#
# The miniconda install on this machine is x86_64. Without the override, conda
# creates an x86_64 environment, and then:
#
#   - jericho compiles libfrotz.so from C using the *system* clang, which targets
#     arm64 and ignores the interpreter's architecture;
#   - the resulting arm64 .so cannot be dlopen'd by the x86_64 interpreter.
#
# The failure is silent at import time, because jericho loads libfrotz lazily via
# ctypes. `import jericho` succeeds. The mismatch only surfaces when a world is
# first opened, with "incompatible architecture (have 'arm64', need 'x86_64')".
#
# Forcing the environment to arm64 makes interpreter and library agree, and
# confines Rosetta to where it belongs: the x86_64 Inform 7 binaries (`ni`,
# `inform6`), which are invoked as subprocesses during world compilation only.
set -euo pipefail

CONDA="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="seahaven-dev"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$CONDA" ]; then
    echo "conda not found at $CONDA — set CONDA_BIN" >&2
    exit 1
fi

if ! /usr/bin/pgrep -q oahd; then
    echo "warning: Rosetta 2 does not appear to be running." >&2
    echo "         Playing compiled worlds will still work; building them will not." >&2
    echo "         Install with: softwareupdate --install-rosetta --agree-to-license" >&2
fi

echo "Creating $ENV_NAME as osx-arm64 ..."
CONDA_SUBDIR=osx-arm64 "$CONDA" env create -f "$REPO_ROOT/env/seahaven-dev.yml" --yes

PY="$("$CONDA" info --base)/envs/$ENV_NAME/bin/python"

echo
echo "Verifying architecture invariants ..."
"$PY" - <<'PYCHECK'
import ctypes, os, platform, subprocess, sys

def arch_of(path):
    out = subprocess.run(["file", path], capture_output=True, text=True).stdout
    return "arm64" if "arm64" in out else "x86_64" if "x86_64" in out else "?"

ok = True

if platform.machine() != "arm64":
    print(f"  FAIL interpreter is {platform.machine()}, expected arm64")
    ok = False
else:
    print("  ok   interpreter is arm64")

import jericho
lib = os.path.join(os.path.dirname(jericho.__file__), "libfrotz.so")
try:
    ctypes.CDLL(lib)
    print(f"  ok   libfrotz.so ({arch_of(lib)}) loads")
except OSError as e:
    print(f"  FAIL libfrotz.so ({arch_of(lib)}) will not load: {e}")
    ok = False

import textworld
ni = os.path.join(os.path.dirname(textworld.__file__),
                  "thirdparty", "inform7-6M62", "share", "inform7", "Compilers", "ni")
if os.path.exists(ni):
    print(f"  ok   Inform 7 `ni` present ({arch_of(ni)}; runs under Rosetta)")
else:
    print("  WARN Inform 7 `ni` missing — world compilation unavailable")

sys.exit(0 if ok else 1)
PYCHECK

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
