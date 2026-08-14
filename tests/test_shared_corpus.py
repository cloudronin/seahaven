"""`_shared.corpus` — the schema, and the labels that must not mislead."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seahaven.eden import outcome as O
from seahaven.eden._shared import corpus as C

_ROOT = Path(__file__).resolve().parents[1]


def test_every_cell_on_disk_is_RECOGNISED_by_one_schema_or_the_other():
    """**Two conventions exist and the loader must not silently drop either.**

    36 cells use the legacy `eden_e{0..5}_{LEVEL}.json` form from before arms
    existed. A parser that returned None for them would have made 36 committed
    cells invisible to every consumer — which the first version of this test
    caught, by failing.
    """
    current = legacy = 0
    for p, _d in C.iter_cells(root=_ROOT / "results"):
        got = C.parse_cell_name(p.name)
        assert got, f"{p.name} matches NEITHER schema"
        if got["schema"] == "current":
            rebuilt = C.cell_path(got["round"], got["model"], got["arm"],
                                  got["level"], root=p.parent)
            assert rebuilt.name == p.name
            current += 1
        else:
            assert got["model"] is None and got["arm"] is None
            legacy += 1
    assert current > 200 and legacy == 36, (current, legacy)


def test_the_MIXED_CASE_round_tag_parses():
    """The timing probe wrote `e11tA`/`e11tB`. A lowercase-only tag pattern
    dropped both, and they are the only cells with real serving timestamps."""
    got = C.parse_cell_name(
        "eden_e11tA_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json")
    assert got and got["round"] == "11tA" and got["arm"] == "A1"


def test_ate_matches_the_inlined_expression_it_replaces():
    """The helper existed 4 times and the raw generator ~13 more."""
    item = O.level_state(O.load_level("world_eden_LAT"))["forbidden"]
    p = _ROOT / "results/eden_e12_deepcogito__cogito-v2-1-671b__A1__LAT.json"
    eps = C.episodes(C.load_cell(p))
    inlined = sum(O.funnel(e["commands"], item)["first_eaten"] is not None
                  for e in eps)
    assert C.ate(eps, item) == inlined == 16


def test_missing_seeds_reads_SEEDS_not_a_count():
    """Counting calls a cell complete when a retry swapped one episode for
    another. Reading the seeds cannot."""
    p = _ROOT / "results/eden_e13_gpt-5.6-terra__A1__LAT.json"
    assert C.missing_seeds(p, 19000, 48) == []
    assert C.missing_seeds(p, 19000, 50) == [19048, 19049]


def test_generation_labels_and_the_refusal_to_pool():
    assert C.generation_of({"terminal_at_zero": True}) == "gen3"
    assert C.generation_of({"round7_pin": "x"}) == "gen2"
    assert C.generation_of({"round8_pin": "x"}) == "gen2"
    assert C.generation_of({"round2_pin": "x"}) == "gen1"
    assert C.generation_of({}) == "gen1"


def test_occasion_label_CARRIES_ITS_SOURCE_and_the_two_differ():
    """**The audit-integrity check.** Only the timing-probe cells have a real
    serving timestamp; everything else is file mtime, which for a gap-filled
    cell is its LAST attempt. Rendering both as "served on" would launder one
    into the other inside the one artifact meant to be trusted about occasions.
    """
    probe = _ROOT / "results/eden_e11tA_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json"
    # Round 13 records wall_start_epoch, so it is NOT an mtime example. Round 12
    # does not — that is the fallback path.
    plain = _ROOT / "results/eden_e12_deepcogito__cogito-v2-1-671b__A1__LAT.json"
    v1, s1 = C.occasion_of(probe, C.load_cell(probe)["meta"])
    v2, s2 = C.occasion_of(plain, C.load_cell(plain)["meta"])
    assert s1 == "wall_start_epoch"
    assert s2 == "mtime"
    assert s1 != s2, "the two sources must be distinguishable at the call site"
    assert v1 and v2


def test_occasion_source_is_NEVER_omitted():
    """A caller cannot accidentally get a bare timestamp: the function returns a
    2-tuple, so dropping the provenance has to be deliberate."""
    p = _ROOT / "results/eden_e12_deepcogito__cogito-v2-1-671b__A1__LAT.json"
    got = C.occasion_of(p, C.load_cell(p)["meta"])
    assert isinstance(got, tuple) and len(got) == 2


def test_burned_seeds_is_PER_MODEL_because_seed_space_is():
    """Every model's A1 cell starts at its round's SEED0, so a seed in another
    model's cell is not a collision. Two hand-rolled versions got this wrong."""
    everywhere = C.burned_seeds(level="LAT", root=_ROOT / "results")
    just_terra = C.burned_seeds(level="LAT", model="gpt-5.6-terra",
                                root=_ROOT / "results")
    assert 15000 in everywhere
    assert 15000 not in just_terra, "15000 is round 10's block, not Terra's"
    assert just_terra <= everywhere
    assert 19000 in just_terra
