#!/usr/bin/env bash
# **The runner's exit code IS pytest's, by construction.**
#
# Twice in one session a suite run was reported green off a shell exit code
# that was not pytest's. `pytest ...; echo "EXIT=$?"` leaves the SHELL's status
# as the process result, so a trailing `echo` — or any command after it —
# returns 0 no matter what pytest did. The second time, one real failure was
# reported as "exit code 0".
#
# A rule ("remember to check pytest's code, not the shell's") is a rule that
# gets forgotten under exactly the conditions that matter: a long run, late,
# when the answer is expected to be green. So this is not a rule.
#
# `exec` REPLACES this shell with pytest. There is no shell left to have its
# own exit code, and nothing can run after pytest to overwrite one. The
# guarantee is structural: it holds even if someone appends a line below.
set -euo pipefail
exec "${PYTEST_PYTHON:-python}" -m pytest "$@"
