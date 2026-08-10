"""Shared loader for the V-P cells.

`analyse_phrasing.py` pools both worlds because G-P was specified pooled. The
flag rule is per **(model, world)**, so it needs the same corpus split rather
than summed. One loader for both, so the two analyses cannot drift into
disagreeing about what a cell contains.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path


def _is_cell(parts: list[str]) -> bool:
    """`vp_<lab>_<phrasing>_world_<v>_<seed>` and nothing else.

    Analysis output lands in the same directory and matches the glob; this has
    already bitten twice (`v3_narration.json`, `vp_phrasing.json`).
    """
    return len(parts) >= 6 and parts[-1].isdigit()


def load_cells(pattern: str = "results/vp_*.json"):
    """(lab, phrasing, world) -> {seed: [violations, commands]}.

    Counts rather than rates, so callers pool by summing — mean-of-ratios moved
    a correlation from 0.964 to 0.679 on identical data once already (TRAP 30).
    """
    from seahaven.fidelity.adherence import classify

    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in sorted(glob.glob(pattern)):
        parts = Path(f).stem.split("_")
        if not _is_cell(parts):
            continue
        lab, ph, world, seed = parts[1], parts[2], f"{parts[3]}_{parts[4]}", parts[5]
        d = json.loads(Path(f).read_text())
        cell = out[(lab, ph, world)][seed]
        for run in d["runs"]:
            for c in run.get("commands", []):
                cell[0] += classify(c["command"]) == "violation"
                cell[1] += 1
    return out


def commands_for(world: str, phrasing: str = "p1",
                 pattern: str = "results/vp_*.json") -> list[str]:
    """Every command issued in one world under one phrasing.

    The C-MIMIC fit corpus. **All** commands as issued, not only the legal ones:
    the anchor is an imitator of the model population's surface statistics, and
    filtering to legal commands would make it an imitator of something no model
    actually produced.
    """
    out = []
    for f in sorted(glob.glob(pattern)):
        parts = Path(f).stem.split("_")
        if not _is_cell(parts):
            continue
        if parts[2] != phrasing or f"{parts[3]}_{parts[4]}" != world:
            continue
        d = json.loads(Path(f).read_text())
        out += [c["command"] for run in d["runs"] for c in run.get("commands", [])]
    return out


def episodes_for(pattern: str = "results/vp_*.json"):
    """(lab, phrasing, world) -> [[violations, commands], ...], one per episode.

    The cluster unit for the possibility bar's seed-stability criterion, which
    the phase entry specifies as an **episode** bootstrap. Commands within an
    episode share room, inventory and conversation state and are not independent
    draws — the same dependence the design-effect sizing refused to assume away.

    Kept beside `load_cells` rather than derived from it: cells are already
    summed over episodes, and re-splitting a sum is not possible.
    """
    from seahaven.fidelity.adherence import classify

    out = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        parts = Path(f).stem.split("_")
        if not _is_cell(parts):
            continue
        lab, ph, world = parts[1], parts[2], f"{parts[3]}_{parts[4]}"
        d = json.loads(Path(f).read_text())
        for run in d["runs"]:
            cmds = run.get("commands", [])
            if cmds:
                out[(lab, ph, world)].append(
                    [sum(classify(c["command"]) == "violation" for c in cmds),
                     len(cmds)])
    return out


def pooled(counts) -> float:
    bad, n = counts
    return 100.0 * (1 - bad / n) if n else float("nan")


def per_world_adherence(cells):
    """(lab, world) -> {phrasing: adherence}, pooled over seeds."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (lab, ph, world), seeds in cells.items():
        tot = agg[(lab, world)][ph]
        for c in seeds.values():
            tot[0] += c[0]
            tot[1] += c[1]
    return {k: {ph: pooled(c) for ph, c in v.items()} for k, v in agg.items()}
