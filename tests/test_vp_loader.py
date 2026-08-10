"""The per-world loader must not disagree with the pooled one it splits.

`analyse_phrasing.py` pools both worlds because G-P was specified pooled;
`_vp_data.py` keeps them apart because the flag rule is per (model, world). Two
loaders over one corpus is exactly the setup where a filename-parsing difference
or a divergent skip rule silently moves a published number, so the split one is
held to reproducing the pooled one on summation.

This regression is on the phase-switch entry's do-not-relax list. It reads the
committed `results/vp_*.json` corpus and skips if that corpus is absent, so a
fresh checkout without results still runs the suite.
"""

from __future__ import annotations

import glob
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

PHRASINGS = ("p1", "p2", "p3", "p4", "p5")

pytestmark = pytest.mark.skipif(
    not glob.glob("results/vp_*.json"),
    reason="V-P corpus not present in this checkout")


@pytest.fixture(scope="module")
def loaders():
    from _vp_data import load_cells

    import analyse_phrasing

    pooled, _ = analyse_phrasing.load()
    return pooled, load_cells()


def test_per_world_counts_sum_to_the_pooled_table(loaders):
    """Summing worlds and seeds must land on the pooled counts exactly.

    Counts, not rates. Comparing rates would let a compensating pair of errors
    pass, and pooling rates is the mistake that moved a correlation from 0.964
    to 0.679 on identical data (TRAP 30).
    """
    pooled, cells = loaders

    summed = defaultdict(lambda: [0, 0])
    for (lab, ph, _world), seeds in cells.items():
        for counts in seeds.values():
            summed[(lab, ph)][0] += counts[0]
            summed[(lab, ph)][1] += counts[1]

    assert {k: tuple(v) for k, v in summed.items()} == \
           {k: tuple(v) for k, v in pooled.items()}, (
        "the per-world loader and the pooled loader disagree about what the "
        "corpus contains. Check the filename skip rule before touching either.")


def test_the_corpus_is_the_shape_the_flag_rule_assumes(loaders):
    """Every (model, world) carries all five phrasings.

    `margin_for` raises on a missing phrasing rather than taking the minimum
    over whichever cells completed. That refusal is only useful if the loader
    would actually surface the gap, so assert the corpus is complete here where
    the failure names the missing cell.
    """
    _, cells = loaders

    have = defaultdict(set)
    for lab, ph, world in cells:
        have[(lab, world)].add(ph)

    missing = {k: sorted(set(PHRASINGS) - v) for k, v in have.items()
               if set(PHRASINGS) - v}
    assert not missing, f"incomplete (model, world) cells: {missing}"


def test_fit_corpus_is_p1_only(loaders):
    """`commands_for` is the C-MIMIC fit corpus and must not pool phrasings.

    Pooling would make the anchor's height depend on the phrasing mix: P5-heavy
    data carries more violations, weakening the imitator and flattering every
    real model against it.
    """
    from _vp_data import commands_for

    _, cells = loaders
    worlds = sorted({w for _, _, w in cells})

    for world in worlds:
        n_p1 = sum(c[1] for (lab, ph, w), seeds in cells.items()
                   if ph == "p1" and w == world for c in seeds.values())
        assert len(commands_for(world, "p1")) == n_p1
        assert n_p1 < sum(c[1] for (lab, ph, w), seeds in cells.items()
                          if w == world for c in seeds.values()), \
            "P1 alone should be a strict subset of the world's commands"
