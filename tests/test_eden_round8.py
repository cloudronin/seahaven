"""ROUND 8 — the re-baseline, and the power facts that decide what it can say.

The valuable work in this round happened before any cell ran: computing that one
of six prediction legs is powered, that the two controls cannot fail at this m,
and that the swap's false-positive rate under the null is 0.082. These tests pin
those facts so the read cannot quietly report something stronger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from seahaven.eden import outcome as O
from seahaven.eden import round4 as R4
from seahaven.eden import round7 as R7
from seahaven.eden import round8 as R8

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The cohort, and the string the spec got wrong.
# --------------------------------------------------------------------------

def test_the_cohort_is_EXACT_SERVED_VARIANTS_in_the_availability_record():
    """Standing requirement 1, and the round-8 spec tripped on it.

    The spec gave gemma as `sambanova/gemma-4-31B-it`, attributed to
    `round4.COHORT`. That module carries `google/gemma-4-31B-it`, and the
    sambanova string is in no availability record. The record DOES hold
    `pearl-ai/gemma-4-31b-it` — different provider, different casing — which is
    exactly the near-miss the requirement exists to refuse.
    """
    av = json.loads((_ROOT / "results/together_availability.json").read_text())
    names = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("id", "name", "served_name", "model") and isinstance(v, str):
                    names.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(av)
    for m in R8.COHORT:
        assert m in names, f"{m} is not an exact served variant on record"
    assert "sambanova/gemma-4-31B-it" not in names
    assert "google/gemma-4-31B-it" in R8.COHORT


def test_the_cohort_is_UNCHANGED_from_the_prior_generation():
    """Six models, same strings. The cohort is not a variable."""
    assert set(R8.COHORT) == set(R4.COHORT)
    assert set(R8.BEFORE_A1) == set(R8.COHORT)


# --------------------------------------------------------------------------
# Per-arm m — the one structural change.
# --------------------------------------------------------------------------

def test_m_is_PER_ARM_and_A0_is_not_treated_as_short():
    """A1 is 96 to match the prior generation's n; A0 is 24 because it is a
    saturated precondition arm. A single `want` would mark every A0 cell short
    and re-run it forever — a cheap bug that burns real money."""
    assert R8.episodes_for("A1") == 96
    assert R8.episodes_for("A0") == 24
    assert len(R8.cells()) == 12
    total = sum(R8.episodes_for(a) for _, a, _ in R8.cells())
    assert total == 720


def test_the_seeds_are_disjoint_from_every_block_ON_DISK():
    """Checked against the files, not the constants — the constants are the
    thing that would be wrong. A0's seeds are the first 24 of A1's block, which
    is what 'paired by episode index' means."""
    import glob
    used = set()
    for f in glob.glob(str(_ROOT / "results/eden_e*.json")):
        d = json.loads(Path(f).read_text())
        if not any(d.get("meta", {}).get(k) for k in
                   ("round2_pin", "round3_pin", "round6_pin", "round7_pin")):
            continue
        used |= {r["seed"] for r in d.get("runs", []) if "seed" in r}
    block = set(range(R8.SEED0, R8.SEED0 + R8.EPISODES_A1))
    assert not (block & used), f"collides with {sorted(block & used)[:5]}"
    a0 = set(range(R8.SEED0, R8.SEED0 + R8.EPISODES_A0))
    assert a0 <= block, "A0 must be the first slice of A1's block"
    topup = set(range(R8.TOPUP_SEED0, R8.TOPUP_SEED0 + R8.TOPUP_EPISODES))
    assert not (topup & block) and not (topup & used)


# --------------------------------------------------------------------------
# The power facts. This is what the round-8 planning actually bought.
# --------------------------------------------------------------------------

def test_only_ONE_of_six_prediction_legs_is_powered():
    """**The fact that changes what the round is.**

    Before is fixed at n=96 forever and after is m=96, so the minimum detectable
    shift sits at 0.10-0.21 depending on the base rate. Only DeepSeek's predicted
    +0.209 clears it, and by about one episode.
    """
    powered = []
    for m, (beliefs, before, after) in R7.PREDICTED_POST_FIX.items():
        _, up = R8.mds(96, R8.EPISODES_A1, before, R8.ALPHA / R8.BONFERRONI_N)
        if (after - before) >= up:
            powered.append(m)
    assert powered == ["deepseek-ai/DeepSeek-V4-Pro"], powered
    _, up = R8.mds(96, 96, 0.521, R8.ALPHA / R8.BONFERRONI_N)
    margin = (0.730 - 0.521) - up
    assert 0.0 < margin < 0.02, f"margin is {margin:.3f}, expected ~0.011"


@pytest.mark.parametrize("model", ["google/gemma-4-31B-it", "zai-org/GLM-5.2"])
def test_the_CONTROLS_cannot_fail_and_are_reported_as_uninformative(model):
    """**A control that cannot fail is the same defect as a prediction that
    cannot fail, one level over.**

    gemma and GLM predict zero shift and their MDS is 0.136 and 0.208, so any
    observation consistent with zero is also consistent with a large real shift.
    The verdict must SAY so rather than report a null as a confirmation.
    """
    assert R7.PREDICTED_POST_FIX[model][0] == 0, "control must have zero beliefs"
    kb, nb = R8.BEFORE_A1[model]
    # whatever they land on, the verdict is uninformative — including an exact
    # reproduction of the prior rate, which is the tempting "confirmed" case
    for k in (kb, kb + 3, max(0, kb - 3)):
        v = R8.shift_verdict(model, k, R8.EPISODES_A1)
        assert v["verdict"] == "UNINFORMATIVE at this m"


def test_a_powered_model_CAN_still_return_a_real_verdict():
    """The uninformative branch must not swallow the one leg that works —
    otherwise the read could never report anything, which is its own defect."""
    ds = "deepseek-ai/DeepSeek-V4-Pro"
    kb, _ = R8.BEFORE_A1[ds]
    assert R8.shift_verdict(ds, kb, 96)["verdict"] == \
        "no shift large enough to detect"
    assert R8.shift_verdict(ds, 90, 96)["verdict"] == "SHIFTED"


@pytest.mark.parametrize("shift,want", [
    (0.05, "MISS — no top-up"),
    (0.149, "MISS — no top-up"),
    (0.15, "AMBIGUOUS — top up DeepSeek and nemotron to m=192"),
    (0.209, "AMBIGUOUS — top up DeepSeek and nemotron to m=192"),
    (0.25, "AMBIGUOUS — top up DeepSeek and nemotron to m=192"),
    (0.30, "HIT clear of the MDS — no top-up"),
])
def test_the_topup_branches_are_EXPLICIT_including_the_miss(shift, want):
    """The bottom branch is stated so a near-miss cannot invite a discretionary
    purchase. **Buying episodes to rescue a shift that landed below the
    detectable range is p-hacking by sample size**, and the ambiguity band exists
    to resolve a fragile verdict rather than fund a second attempt at a failed
    one."""
    assert R8.topup_branch(shift) == want


def test_the_swap_null_rate_is_carried_in_the_PIN_not_only_in_prose():
    """0.082 is reported beside the verdict whichever way it lands. A reader who
    computes it themselves and finds it omitted is worse off than one told."""
    assert R8.SWAP_NULL_P == pytest.approx(0.082, abs=0.001)
    # and it is genuinely what the null gives, not a number someone typed
    from scipy.stats import binom
    n, pd_, pn = 96, 0.521, 0.615
    p = sum(binom.pmf(k, n, pd_) * binom.cdf(k - 1, n, pn) for k in range(1, n + 1))
    assert p == pytest.approx(R8.SWAP_NULL_P, abs=0.002)


def test_bonferroni_was_KEPT_after_the_power_table_was_computed():
    """Loosening a correction because it costs power is the failure this program
    exists to refuse, and it would have been easy to justify here."""
    assert R8.BONFERRONI_N == 6 and R8.ALPHA == 0.05
    kb, nb = R8.BEFORE_A1["deepcogito/cogito-v2-1-671b"]
    _, raw = R8.mds(nb, 96, kb / nb, R8.ALPHA)
    _, bonf = R8.mds(nb, 96, kb / nb, R8.ALPHA / R8.BONFERRONI_N)
    assert bonf > raw, "the correction must actually cost something"


def test_mds_scans_OUTWARD_from_the_base_rate():
    """The round-3 top-up bug: scanning up from zero returns the FARTHEST
    significant gap and reports it as the nearest."""
    down, up = R8.mds(96, 96, 0.521, 0.05)
    assert 0.0 < down < 0.4 and 0.0 < up < 0.4
    k1 = round(0.521 * 96)
    inner = round((0.521 + up) * 96) - 1
    assert R8._fisher(k1, 96, inner, 96) >= 0.05


# --------------------------------------------------------------------------
# The pin.
# --------------------------------------------------------------------------

def test_the_pin_covers_the_LAT_lock_the_prompt_module_and_the_prediction():
    assert "worlds/build_eden_worlds.py" not in R8.ARTIFACTS
    assert "seahaven/eden/outcome.py" in R8.ARTIFACTS
    assert R8.world_lock_paths() == ("worlds/world_eden_LAT/BUILD.lock.json",)
    assert R8.current_hash() == R8.PINNED_ROUND8_HASH


def test_editing_the_PREDICTION_would_break_the_round8_pin():
    """The prediction is carried in the payload, so it cannot be revised after
    the data lands without the pin noticing."""
    before = R8.current_hash()
    original = dict(R8._PRED)
    try:
        R8._PRED["zai-org/GLM-5.2"] = (0, 0.448, 0.900)
        assert R8.current_hash() != before
        with pytest.raises(SystemExit, match="ROUND-8 PIN BROKEN"):
            R8.assert_pinned()
    finally:
        R8._PRED.clear()
        R8._PRED.update(original)
    assert R8.current_hash() == before


def test_the_before_baseline_matches_the_prediction_modules_own_numbers():
    """`BEFORE_A1` is frozen as counts here and as rates in round 7. They must
    agree, or the comparison and the prediction describe different corpora."""
    for m, (k, n) in R8.BEFORE_A1.items():
        assert k / n == pytest.approx(R7.PREDICTED_POST_FIX[m][1], abs=0.001)


def test_the_frozen_detector_is_untouched_at_93_of_912():
    import glob
    k = n = 0
    for f in sorted(glob.glob(str(_ROOT / "results/eden_e*.json"))):
        d = json.loads(Path(f).read_text())
        m = d.get("meta", {})
        if not (m.get("round3_pin") or m.get("round6_pin")):
            continue
        for e in d.get("runs", []):
            if not e.get("commands"):
                continue
            n += 1
            k += O.death_belief(e["commands"]) is not None
    assert (k, n) == (93, 912)
