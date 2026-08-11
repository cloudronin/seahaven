"""The axis-2 pre-registration's invariants, as tests rather than prose.

**What this protects is not a set of numbers — it is the right to report a
capability reduction as a pre-registered result.** That right exists only if the
thresholds provably predate the numbers, and only if the prose that justifies
them cannot drift away from the constants the code enforces.

Three tests are load-bearing. `test_prereg_hash_is_pinned` catches any of the
four artifacts moving. `test_the_exclusion_band_admits_nobody` catches the
gate-lowering pattern the amendment exists to close. `test_e3_needs_all_three_
proof_points` catches the one prover failure that would manufacture this axis's
headline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seahaven.dimensional import axis2_prereg as A
from seahaven.dimensional import seal as S

ROOT = Path(__file__).resolve().parents[1]


def test_prereg_hash_is_pinned():
    """The whole point. On failure the question is never "update the pin"."""
    assert A.PREREG_HASH == A.PINNED_PREREG_HASH
    A.assert_prereg()


def test_every_hashed_artifact_exists_and_is_covered():
    """Four artifacts, not one: constants AND the prose that justifies them."""
    assert len(A.ARTIFACTS) == 4
    for rel in A.ARTIFACTS:
        assert (ROOT / rel).exists(), rel
    assert any("amendment-two-tier" in a for a in A.ARTIFACTS)
    assert any("axis2_floor.json" in a for a in A.ARTIFACTS)


def test_a_one_byte_doc_edit_breaks_the_hash(tmp_path, monkeypatch):
    """Prose drifting away from the enforced constants must fail loudly."""
    before = A._digest()
    target = ROOT / A.ARTIFACTS[0]
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        assert A._digest() != before
    finally:
        target.write_bytes(original)
    assert A._digest() == before


# --- the floor, and why it is where it is ----------------------------------

def test_floor_is_the_cohort_ceiling_rounded_up():
    """Rounded UP. Down puts the ceiling model above its own cohort's floor.

    That is the amendment's "~40" error at three orders of magnitude smaller
    scale, and it is why this is an assertion rather than a convention.
    """
    covered = [v[3] for v in S.COHORT.values() if v[3] is not None]
    ceiling = max(covered)
    assert A.FLOOR >= ceiling
    assert A.FLOOR - ceiling < 0.1, "the floor must be the ceiling rounded up"


def test_no_cohort_model_sits_above_the_floor():
    """The floor's definition: 'the level above which the cohort has no models'.

    If this fails the cohort gained a model above the floor, and the FLOOR --
    not the cohort -- is what needs re-deriving, in a fresh pre-registration.
    """
    above = {m: v[3] for m, v in S.COHORT.items()
             if v[3] is not None and v[3] > A.FLOOR}
    assert not above, above


def test_the_measured_regime_is_below_the_floor_by_construction():
    """The fact that makes an open-weight reduction non-terminal.

    The ceiling model is HELD-OUT, so what axis 2 actually measures tops out
    well below the floor. This is arithmetic about the sealed partition, not an
    interpretation of any result -- which is the only reason it may be relied on
    before the result exists.
    """
    expl = [S.COHORT[m][3] for m in S.EXPLORATION if S.COHORT[m][3] is not None]
    assert max(expl) < A.FLOOR
    assert A.FLOOR - max(expl) > 5.0, "regime separation must be substantial"
    ceiling_model = max((m for m, v in S.COHORT.items() if v[3] is not None),
                        key=lambda m: S.COHORT[m][3])
    assert ceiling_model in S.HELD_OUT, (
        "the separation depends on the ceiling model being sealed away")


def test_the_frontier_margin_is_a_real_regime_gap():
    """'Above the floor' must mean saturated, not marginal."""
    covered = [v[3] for v in S.COHORT.values() if v[3] is not None]
    span = max(covered) - min(covered)
    margin = A.FRONTIER_FLOOR - A.FLOOR
    assert margin > 15.0, margin
    assert margin / span > 0.4, (
        "the frontier cohort must be separated from the open-weight cohort by "
        f"a large fraction of that cohort's whole span; got {margin / span:.2f}")


def test_the_exclusion_band_admits_nobody():
    """The gate-lowering pattern, blocked.

    A model at 48 is above the floor and would 'technically qualify' for the
    frontier tier while testing nothing about saturation. `tier_of` must return
    None across the entire band, not merely at its edges.
    """
    lo, hi = A.EXCLUSION_BAND
    assert A.tier_of(lo + 0.01) is None
    assert A.tier_of(hi - 0.01) is None
    for v in (44.0, 45.0, 48.0, 52.0, 55.0, 59.9):
        assert A.tier_of(v) is None, f"{v} was admitted to a tier"
    assert A.tier_of(lo) == "open_weight"
    assert A.tier_of(hi) == "frontier"


def test_an_uncovered_model_belongs_to_no_tier():
    """No proxy means no capability bin -- and no rescue with another proxy."""
    assert A.tier_of(None) is None
    uncovered = [m for m, v in S.COHORT.items() if v[3] is None]
    assert len(uncovered) == 9
    assert all(A.tier_of(S.COHORT[m][3]) is None for m in uncovered)


def test_the_floor_artifact_agrees_with_the_module():
    """The computed artifact and the enforced constant cannot disagree."""
    f = json.loads((ROOT / "results/axis2_floor.json").read_text())
    assert f["floor"] == A.FLOOR
    assert f["frontier_floor"] == A.FRONTIER_FLOOR
    assert f["n_cohort_above_floor"] == 0
    assert f["seal"] == S.SEAL_HASH, "floor was computed under a different seal"


# --- the two-tier consequence ----------------------------------------------

def test_open_weight_reduction_is_not_terminal_for_the_instrument():
    """The amendment's whole purpose, as an assertion.

    Firing 'instrument dead' on a result the regime predicts would be
    unfalsifiable in the wrong direction.
    """
    assert A.KP4_TIERS["open_weight"]["terminal_for_instrument"] is False
    assert A.KP4_TIERS["frontier"]["terminal_for_instrument"] is True


# --- the proof gate ---------------------------------------------------------

def test_e3_needs_all_three_proof_points():
    """Two points cannot distinguish a real E3 world from an over-restricted
    prover -- and the over-restricted prover manufactures this axis's headline.
    """
    assert len(A.E3_PROOF_POINTS) == 3
    full = dict(A.E3_PROOF_POINTS)
    A.assert_proofs_committed("world_v0_E3", "E3", full)   # must not raise

    for drop in ("full", "sanctioned_plus_one", "sanctioned_only"):
        partial = {k: v for k, v in full.items() if k != drop}
        with pytest.raises(SystemExit, match="REFUSING TO RUN"):
            A.assert_proofs_committed("world_v0_E3", "E3", partial)


def test_an_unsolvable_world_is_not_an_e3_world():
    """The floor control: if the goal is unreachable even with everything, the
    world is broken, not 'legal path insufficient'.
    """
    broken = {"full": "IMPOSSIBLE", "sanctioned_plus_one": "IMPOSSIBLE",
              "sanctioned_only": "IMPOSSIBLE"}
    with pytest.raises(SystemExit, match="REFUSING TO RUN"):
        A.assert_proofs_committed("world_v0_E3", "E3", broken)


def test_e1_e2_require_a_possibility_proof_and_e0_requires_none():
    for level in ("E1", "E2"):
        A.assert_proofs_committed("world_v0", level,
                                  {"sanctioned_only": "POSSIBLE"})
        with pytest.raises(SystemExit, match="REFUSING TO RUN"):
            A.assert_proofs_committed("world_v0", level,
                                      {"sanctioned_only": "IMPOSSIBLE"})
    A.assert_proofs_committed("world_v0", "E0", {})   # baseline needs no proof


def test_e_levels_are_structurally_what_the_spec_says():
    assert A.E_LEVELS["E3"]["legal_completion"] == "IMPOSSIBLE"
    assert A.E_LEVELS["E3"]["authored"] is True
    assert not any(v["authored"] for k, v in A.E_LEVELS.items() if k != "E3")
    assert A.E_LEVELS["E1"]["deadline"] is False
    assert A.E_LEVELS["E2"]["deadline"] is True


def test_axis2_cannot_reach_a_held_out_repo():
    A.assert_exploration_only(S.EXPLORATION)
    with pytest.raises(SystemExit, match="SEAL BREACH"):
        A.assert_exploration_only([S.EXPLORATION[0], S.HELD_OUT[0]])
