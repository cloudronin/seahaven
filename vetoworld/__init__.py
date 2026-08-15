"""VetoWorld — a benchmark of expedience under terminal stakes.

An agent is put in a world where survival requires food, told it may not eat one
particular item, and the item is the only thing that keeps it alive. The measure
is whether it eats anyway. The name names **the rule**: a world with a veto in it.

The public surface. The construct is **expedience**; the metric is a verified
reach rate, behaviourally defined, and it is deliberately not named after a
mental state the design refuses to claim.

The internal library keeps its own name: its paths are hashed into the round
pins, so renaming it would invalidate every freeze. Nothing in this package's
user-facing output names it. `docs/naming.md` records why each candidate name was
rejected, so a future contributor does not relitigate it.
"""

def _version() -> str:
    """**Derived, never declared twice.**

    This was a literal `"0.1.0"` while `pyproject.toml` said `0.1.2`, so
    `vworld --version` reported a release that had not existed for two
    publishes. Harmless until the scheduled probe pins
    `vetoworld[probe]==<version>` — at which point a job would install one
    version and a row would record another.

    Installed: the package metadata, which is what `pip` actually resolved.
    From a source checkout: `pyproject.toml`, the single place the number is
    written. Two sources, one fact, and neither is a second copy to maintain.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("vetoworld")
    except PackageNotFoundError:
        import tomllib
        from pathlib import Path
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject.exists():          # installed oddly, no metadata
            return "0+unknown"
        return tomllib.loads(pyproject.read_text())["project"]["version"]


__version__ = _version()
