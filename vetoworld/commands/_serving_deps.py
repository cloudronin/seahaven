"""One place that answers: can this install actually serve an episode?

**0.3.0's scheduled job could not, and said PARTIAL.** Every cell raised
`No module named 'textworld'` inside the serve loop, was caught by the
per-cell `except Exception` as a SERVE_FAIL, and the day reported success with
an empty push. Twenty-two identical failures, each individually "handled".

A missing dependency is not a serving failure. It is a broken install, it is
the same for every cell, and it is knowable BEFORE the first request. Checking
once and refusing is the difference between an actionable message and
twenty-two stack-trace fragments that look like provider trouble.
"""

from __future__ import annotations

#: Imported by the serving path, absent from the base install. `textworld.gym`
#: is what `seahaven/world/loader.py` needs to open a compiled world.
SERVING_MODULES = ("textworld", "jericho")


def missing() -> list[str]:
    import importlib.util

    return [m for m in SERVING_MODULES
            if importlib.util.find_spec(m) is None]


def require(verb: str) -> None:
    """Refuse before serving anything, with the command that fixes it."""
    gone = missing()
    if not gone:
        return
    raise SystemExit(
        f"CANNOT SERVE: {', '.join(gone)} not installed.\n"
        f"  `{verb}` opens a compiled world, which needs the serving extra:\n"
        "      pip install 'vetoworld[serve]'\n"
        "  The scheduled job installs `vetoworld[probe]`, which includes it.\n"
        "  Refusing here rather than failing once per cell: a missing "
        "dependency is a broken install, not a provider outage, and reporting "
        "it as the latter is how 0.3.0 pushed an empty day and called it "
        "PARTIAL.")
