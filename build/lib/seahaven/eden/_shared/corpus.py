"""The cell corpus: one path schema, one loader, one set of labels.

Fourteen scripts parsed the filename convention independently, and `load` was
defined fifteen times with fifteen different bodies — the same name drifting into
different semantics, which is worse than clean duplication because it looks
consistent.

**The occasion label carries its own provenance.** Real serving timestamps exist
only on the timing-probe cells (`wall_start_epoch`); everything else falls back to
file last-written, which for a gap-filled cell is its LAST attempt, not its first.
`emit occasions` is the mandatory audit of which comparisons span serving days,
and a label that rendered both sources identically would launder an mtime into a
serving date inside the one artifact whose entire purpose is to be trusted about
occasions. So `occasion_of` returns `(value, source)` and callers must print both.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import re
from pathlib import Path

from ..outcome import funnel

__all__ = [
    "RESULTS", "cell_path", "parse_cell_name", "iter_cells", "load_cell",
    "episodes", "ate", "missing_seeds", "generation_of", "occasion_of",
    "burned_seeds",
]

RESULTS = Path("results")

#: The CURRENT convention, round 2 onward:
#: `eden_e{round}_{model with / -> __}__{arm}__{level}.json`.
#: The round tag is mixed-case because the timing probe used `e11tA`/`e11tB`.
_NAME = re.compile(r"^eden_e(?P<round>[0-9A-Za-z]+)_(?P<model>.+?)"
                   r"__(?P<arm>A[01])__(?P<level>[A-Za-z0-9]+)\.json$")

#: **The LEGACY convention, and it is not a rounding error: 36 cells.**
#: `eden_e{0..5}_{LEVEL}.json` — the exploratory ladder, written before arms
#: existed. The model lives in `meta.served_name` and `meta.eden_arm` is None.
#: Recognised explicitly so a loader cannot silently drop them; a bare `None`
#: return would have made 36 committed cells invisible to every consumer.
_LEGACY = re.compile(r"^eden_e(?P<round>[0-5])_(?P<level>[A-Za-z0-9]+)\.json$")


def cell_path(round_tag: str, model: str, arm: str, level: str,
              root: Path | None = None) -> Path:
    """The one place the filename convention is written."""
    base = RESULTS if root is None else root
    return base / f"eden_e{round_tag}_{model.replace('/', '__')}__{arm}__{level}.json"


def parse_cell_name(name: str) -> dict | None:
    """Parse either convention. `schema` says which; `None` means neither.

    A legacy row carries `model=None, arm=None` — the filename genuinely does
    not encode them, and inventing a value from `meta` here would hide that the
    two conventions differ.
    """
    n = Path(name).name
    m = _NAME.match(n)
    if m:
        d = m.groupdict()
        d["model"] = d["model"].replace("__", "/")
        d["schema"] = "current"
        return d
    m = _LEGACY.match(n)
    if m:
        d = m.groupdict()
        d.update(model=None, arm=None, schema="legacy")
        return d
    return None


def iter_cells(pattern: str = "eden_e*.json", root: Path | None = None):
    """Yield `(path, payload)` for every cell matching `pattern`."""
    base = RESULTS if root is None else root
    for f in sorted(glob.glob(str(base / pattern))):
        p = Path(f)
        yield p, json.loads(p.read_text())


def load_cell(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def episodes(payload: dict) -> list[dict]:
    """Runs that actually produced commands. A failed run is not an episode."""
    return [r for r in payload.get("runs", []) if r.get("commands")]


def ate(eps: list[dict], item: str) -> int:
    """Episodes that completed an eat of the forbidden item.

    Was a 4-times-duplicated helper plus ~13 inlined copies of the same
    generator expression.
    """
    return sum(funnel(e["commands"], item)["first_eaten"] is not None
               for e in eps)


def missing_seeds(path: Path, seed0: int, m: int) -> list[int]:
    """Seeds a cell is SHORT, read from the file rather than from a count.

    Counting would call a cell complete when a retry replaced one episode with
    another; reading the seeds cannot.
    """
    have = {r["seed"] for r in episodes(load_cell(path)) if "seed" in r}
    return sorted(set(range(seed0, seed0 + m)) - have)


def generation_of(meta: dict) -> str:
    """1 = no recovery line, non-terminal. 2 = recovery line. 3 = terminal death.

    **Never pool across these**: the served prompt and the death semantics both
    differ, so a pooled rate averages two different measurements.
    """
    if meta.get("terminal_at_zero") is True:
        return "gen3"
    if meta.get("round7_pin") or meta.get("round8_pin"):
        return "gen2"
    return "gen1"


def occasion_of(path: Path, meta: dict) -> tuple[str, str]:
    """`(iso timestamp, source)` — and the source is never optional.

    `wall_start_epoch` is a real serving time. `mtime` is when the file was last
    written, which for a gap-filled cell is its last attempt. Printing them
    identically would misrepresent the second as the first.
    """
    if meta.get("wall_start_epoch"):
        ts = _dt.datetime.fromtimestamp(meta["wall_start_epoch"])
        return ts.strftime("%Y-%m-%dT%H:%M"), "wall_start_epoch"
    ts = _dt.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return ts.strftime("%Y-%m-%dT%H:%M"), "mtime"


def burned_seeds(level: str | None = None, model: str | None = None,
                 root: Path | None = None) -> set[int]:
    """Every seed already served, optionally narrowed.

    Seed space is PER MODEL: every model's A1 cell starts at its round's SEED0,
    so a seed appearing in another model's cell is not a collision. Four scripts
    hand-rolled this check and two of them got that wrong.
    """
    out: set[int] = set()
    for _p, d in iter_cells(root=root):
        meta = d.get("meta", {})
        if level and meta.get("eden_level") != level:
            continue
        if model and meta.get("served_name") != model:
            continue
        out |= {r["seed"] for r in d.get("runs", []) if "seed" in r}
    return out


def repo_root() -> Path:
    """The root that pin paths and world locks are relative to.

    From a checkout this is the repo; from an installed wheel it is
    `site-packages/`. Derived from the library's own location so both work
    without either caller knowing which it is in.
    """
    return Path(__file__).resolve().parents[3]


def corpus_present(root: Path | None = None) -> int:
    """How many cells are on disk. **Zero is a distinct state, not a result.**

    A verifier that computes over an empty corpus reports every figure as
    drifted, which tells a replicator the manuscript is wrong when the truth is
    that they have no data. The count is checked before any figure is computed.
    """
    base = RESULTS if root is None else root
    return len(list(Path(base).glob("eden_e*.json"))) if Path(base).exists() else 0
