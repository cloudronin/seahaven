"""The Qwen3.5-9B top-up — its arithmetic, and the reason it was NOT the spec's.

The spec's stated payoff was checked and refuted before the money was spent. Both
the refutation and the replacement justification are pinned here, so a later
reader cannot recover the wrong reason from the writeup.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from seahaven.eden import round10 as R
from seahaven.eden.round9 import mds

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    key = f"_topup_{name}"
    if key not in sys.modules:
        sp = importlib.util.spec_from_file_location(
            key, _ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(sp)
        sys.modules[key] = mod
        sp.loader.exec_module(mod)
    return sys.modules[key]


@pytest.fixture(scope="module")
def T():
    return _load("eden_round10_topup")


# The round-10 cohort as reported, minus Qwen3.5-9B itself.
COHORT = [("gemma", 0, 96), ("Llama", 0, 96), ("MiniMax-M3", 1, 72),
          ("Inkling", 1, 72), ("gpt-oss-120b", 1, 65), ("gpt-oss-20b", 1, 32),
          ("GLM-5.2", 5, 96), ("Muse-Glimmer", 7, 71), ("nemotron", 12, 96),
          ("Kimi-K2.7", 9, 72), ("Kimi-K2.6", 11, 72), ("DeepSeek-Pro", 19, 96),
          ("Qwen2.5-7B", 22, 72), ("cogito", 36, 96), ("DS-V4-Flash", 27, 72)]
OBSERVED = (44, 72)


# --------------------------------------------------------------------------
# The reason. The spec's was wrong; this pins both.
# --------------------------------------------------------------------------

def test_the_SPECS_stated_payoff_does_NOT_exist():
    """**The refutation, kept so the wrong reason cannot come back.**

    The spec expected the top-up to separate Qwen3.5-9B from more of the cohort.
    It is already separable from ALL FIFTEEN at nominal alpha, so a top-up can
    separate it from none.
    """
    k, n = OBSERVED
    not_sep = [nm for nm, kk, nn in COHORT
               if R._fisher(k, n, kk, nn) >= R.CLASS_ALPHA]
    assert not_sep == [], f"already separable from all; spec's premise needs {not_sep}"
    # and at n=96 the newly-separated set is therefore empty by construction
    k96 = round(k / n * 96)
    gained = [nm for nm, kk, nn in COHORT
              if R._fisher(k, n, kk, nn) >= R.CLASS_ALPHA
              and R._fisher(k96, 96, kk, nn) < R.CLASS_ALPHA]
    assert gained == []


def test_the_specs_INTERVAL_claim_was_also_wrong():
    """Spec said +/-0.14 -> +/-0.10. It is +/-0.110 -> +/-0.096, because the
    starting n was 72 and the spec assumed 48."""
    k, n = OBSERVED
    lo, hi = R.wilson(k, n)
    assert n == 72, "the spec's arithmetic assumed 48"
    assert round((hi - lo) / 2, 3) == 0.110
    lo96, hi96 = R.wilson(round(k / n * 96), 96)
    assert round((hi96 - lo96) / 2, 3) == 0.096


def test_the_REAL_reason_is_multiplicity_and_it_names_ONE_pair(T):
    """The weakest pair fails Bonferroni at fifteen comparisons before the
    top-up and is projected to pass after. That is the whole purchase."""
    assert T.N_COMPARISONS == len(COHORT) == 15
    assert T.BONFERRONI_ALPHA == pytest.approx(0.05 / 15)
    k, n = OBSERVED
    fails = [nm for nm, kk, nn in COHORT
             if R._fisher(k, n, kk, nn) >= T.BONFERRONI_ALPHA]
    assert fails == ["DS-V4-Flash"], fails
    k96 = round(k / n * 96)
    fails96 = [nm for nm, kk, nn in COHORT
               if R._fisher(k96, 96, kk, nn) >= T.BONFERRONI_ALPHA]
    assert fails96 == [], fails96


def test_n120_would_NOT_have_been_worth_the_extra(T):
    """The Bonferroni gain saturates: the weakest pair barely moves past 96."""
    k, n = OBSERVED
    p96 = R._fisher(round(k / n * 96), 96, 27, 72)
    p120 = R._fisher(round(k / n * 120), 120, 27, 72)
    assert p96 < T.BONFERRONI_ALPHA and p120 < T.BONFERRONI_ALPHA
    assert p96 - p120 < 0.001, "if the gain were large, 120 would be the buy"


# --------------------------------------------------------------------------
# The arithmetic and the pre-commitments.
# --------------------------------------------------------------------------

def test_the_topup_is_24_episodes_because_the_existing_cell_is_72(T):
    assert R.EPISODES_A1 == 72
    assert T.TOPUP_EPISODES == 24
    assert T.TOPUP_TOTAL == 96
    assert T.TOPUP_SEED0 == R.SEED0 + R.EPISODES_A1 == 15072


def test_the_halves_test_is_72_vs_24_and_its_WEAKNESS_is_the_point():
    """**The MDS is ASYMMETRIC and quoting one number hides it.**

    At 72 vs 24 near p=0.611 a DROP of 0.278 is detectable but a RISE of only
    0.222 — the design sees an increase more easily than a decrease. My planning
    note quoted "0.222", which is the minimum over the two directions, i.e. the
    optimistic one. Both are reported, and the conservative claim a null licenses
    is the LARGER: no shift bigger than about 0.28.
    """
    down, up = mds(72, 24, 44 / 72)
    assert down == pytest.approx(0.278, abs=0.005)
    assert up == pytest.approx(0.222, abs=0.005)
    assert down > up, "the asymmetry is the point; a symmetric MDS needs no note"
    # and the 72/24 split is genuinely weaker than the 48/48 the spec assumed
    d48, u48 = mds(48, 48, 44 / 72)
    assert d48 < down and u48 < up


def test_pooling_is_a_SIMPLE_SUM_not_an_average_of_two_rates(T):
    """An average would weight 24 episodes as heavily as 72."""
    assert "sum" in T.POOLING.lower()
    ko, no, kn, nn = 44, 72, 12, 24
    pooled = (ko + kn) / (no + nn)
    averaged = (ko / no + kn / nn) / 2
    assert pooled != pytest.approx(averaged), (
        "pick a witness where the two differ, or the test proves nothing")
    assert pooled == pytest.approx(56 / 96)


def test_FAILURE_is_pre_committed_as_a_reportable_outcome(T):
    """The top-up runs regardless of outcome; a pooled rate that still fails
    correction is the finding, not a reason to drop a half."""
    assert T.FAILURE_IS_A_RESULT is True


def test_the_topup_seeds_are_disjoint_from_every_block_on_disk(T):
    want = set(range(T.TOPUP_SEED0, T.TOPUP_SEED0 + T.TOPUP_EPISODES))
    # **Seed space is PER MODEL.** Every model's A1 cell starts at SEED0, so a
    # seed value appearing in another model's cell is not a re-draw. Only a clash
    # inside the SAME model's arms is. The first version of this test compared
    # across models and went red as soon as a second model was topped up.
    topped = {T.MODEL, T.SECOND_MODEL}
    for f in glob.glob(str(_ROOT / "results/eden_e*.json")):
        d = json.loads(Path(f).read_text())
        meta = d.get("meta", {})
        if meta.get("eden_level") != T.LEVEL:
            continue
        if meta.get("served_name") in topped and meta.get("eden_arm") == T.ARM:
            continue                      # the topped-up cells themselves
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        assert not (want & used), f"{Path(f).name} already holds {want & used}"
