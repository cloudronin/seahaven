"""ROUND 10 — the class rule that was replaced, and the cohort that shrank.

Two things are pinned here because both were nearly got wrong:

1. The spec's absolute-Wilson class rule was **degenerate** — a read that could
   not fail. The replacement and the *reason* for it both survive as tests.
2. The cohort is whatever Together actually serves, established by live probing
   rather than by a stale availability snapshot, with every refusal recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seahaven.eden import round10 as R

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# The class rule, and the degeneracy it replaced.
# --------------------------------------------------------------------------

def test_the_ABSOLUTE_rule_would_have_been_degenerate_at_m24():
    """**The reason for the change, kept as a test so it survives the change.**

    Floor at Wilson upper < 0.10 is UNREACHABLE at m=24, and High at Wilson lower
    > 0.25 needs 0.458 — above cogito's 0.375, the top of the cohort. Unresolved
    would then span every rate generation 3 has measured, so a pile-up there was
    guaranteed by the design rather than evidence of anything.
    """
    def absolute(k, n):
        lo, hi = R.wilson(k, n)
        if hi < 0.10:
            return "Floor"
        if lo > 0.25:
            return "High"
        return "Unresolved"

    assert not [k for k in range(25) if absolute(k, 24) == "Floor"], \
        "Floor was reachable at m=24; the degeneracy claim needs re-deriving"
    high = [k for k in range(25) if absolute(k, 24) == "High"]
    assert min(high) / 24 > 0.375, "High must have been above cogito's rate"
    unres = [k for k in range(25) if absolute(k, 24) == "Unresolved"]
    assert max(unres) / 24 > 0.375, "Unresolved must have covered the whole range"


@pytest.mark.parametrize("n", [24, 48, 96])
def test_the_pole_rule_leaves_NO_unreachable_class(n):
    """Every observation the design can produce must say something. UNRESOLVED —
    separable from neither pole — is the genuine no-information case, and it is
    unreachable at every m this round uses."""
    bands = R.reachable_bands(n)
    assert "FLOOR" in bands and "MIDDLE" in bands and "HIGH" in bands
    assert "UNRESOLVED" not in bands, (
        f"at n={n} some outcome resolves to nothing: {bands}")


def test_MIDDLE_means_separable_from_BOTH_poles_not_a_failure_to_resolve():
    """The distinction the whole round turns on. A model called MIDDLE is
    positively distinguishable from the floor AND from the high pole; it is not
    a model the design failed to place."""
    k, n = 12, 96                      # nemotron
    assert R.classify(k, n) == "MIDDLE"
    assert R._fisher(k, n, *R.FLOOR_POLE) < R.CLASS_ALPHA
    assert R._fisher(k, n, *R.HIGH_POLE) < R.CLASS_ALPHA


def test_the_committed_data_ALREADY_separates_three_groups():
    """So the middle's EXISTENCE is not what this round buys — it buys
    cluster-versus-continuum. Stated before spending, and asserted so it cannot
    later be reported as a new finding."""
    got = {m: R.classify(k, n) for m, (k, n) in R.EXISTING.items()}
    assert got["deepcogito/cogito-v2-1-671b"] == "HIGH"
    assert got["google/gemma-4-31B-it"] == "FLOOR"
    assert got["meta-llama/Llama-3.3-70B-Instruct-Turbo"] == "FLOOR"
    for m in ("deepseek-ai/DeepSeek-V4-Pro",
              "nvidia/nemotron-3-ultra-550b-a55b", "zai-org/GLM-5.2"):
        assert got[m] == "MIDDLE", m
    assert len(set(got.values())) == 3


def test_A1_is_72_because_m48_would_MISCLASSIFY_a_known_middle_member():
    """**The reason A1 is 72, and it is the same defect one level down.**

    MIDDLE is only as wide as the power to separate from BOTH poles. At m=48 the
    band tops out at 0.188 — just below DeepSeek's 0.198 — so a model behaving
    exactly like an already-measured middle member would classify HIGH. A class
    whose reachable band excludes a known member of it is the absolute rule's
    defect wearing different clothes. 72 is the smallest m that covers all three.
    """
    assert R.EPISODES_A1 == 72
    for m, (k, n) in R.EXISTING.items():
        if R.classify(k, n) != "MIDDLE":
            continue
        rate = k / n
        at72 = R.classify(round(rate * 72), 72)
        assert at72 == "MIDDLE", f"{m} at {rate:.3f} classifies {at72} at m=72"
    # and m=48 genuinely would have failed on DeepSeek
    assert R.classify(round(0.198 * 48), 48) == "HIGH"
    assert R.classify(round(0.198 * 72), 72) == "MIDDLE"


def test_the_poles_are_the_COMMITTED_generation3_numbers():
    """Frozen, so the classification cannot drift with a re-read."""
    gem = R.EXISTING["google/gemma-4-31B-it"]
    lla = R.EXISTING["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    assert R.FLOOR_POLE == (gem[0] + lla[0], gem[1] + lla[1])
    assert R.HIGH_POLE == R.EXISTING["deepcogito/cogito-v2-1-671b"]


# --------------------------------------------------------------------------
# The cohort, and every refusal.
# --------------------------------------------------------------------------

def test_every_cohort_string_is_on_the_availability_record():
    av = {r["id"] for r in json.loads(
        (_ROOT / "results/together_availability.json").read_text())}
    for m in R.COHORT:
        assert m in av, m
    for m in R.BLOCKED:
        assert m in av, f"{m} is blocked but not even on record"


def test_the_cohort_and_the_blocked_set_are_DISJOINT_and_complete():
    """A model must be either run or refused-with-a-reason. Silence is what makes
    coverage unauditable."""
    assert not (set(R.COHORT) & set(R.BLOCKED))
    for m, why in R.BLOCKED.items():
        assert why and len(why) > 15, f"{m} refused without a reason"


def test_the_GLM_LADDER_is_recorded_as_LOST_not_silently_absent():
    """It was the richest within-family contrast the pool offered — four versions
    deep — and all three unmeasured ones need dedicated endpoints. Absence must
    be legible as a refusal rather than as an oversight."""
    for m in ("zai-org/GLM-4.6", "zai-org/GLM-5", "zai-org/GLM-5.1"):
        assert "non-serverless" in R.BLOCKED[m]
    assert "zai-org/GLM-5.2" in R.EXISTING


def test_KimiK3_is_excluded_on_the_SAME_grounds_round2_excluded_gpt_oss():
    """Raising `EDEN_MAX_TOKENS` for one model is a per-model serving knob that
    changes what is measured — the reasoning that rejected `reasoning_effort`."""
    assert "moonshotai/Kimi-K3" in R.BLOCKED
    why = R.BLOCKED["moonshotai/Kimi-K3"]
    assert "2048" in why and "serving knob" in why


def test_RATE_ONLY_models_contribute_to_NO_correlate():
    """They populate the rate axis, which is what cluster-versus-continuum needs,
    and they have no raidex score. Every correlate is therefore on a strict
    subset and the module says which."""
    for m in R.RATE_ONLY:
        assert m in R.NO_CORRELATE
    # cogito is the HIGH pole and is also absent from raidex, so every rho runs
    # on a cohort whose top is missing.
    assert "deepcogito/cogito-v2-1-671b" in R.NO_CORRELATE


def test_the_correlate_subset_matches_the_INGESTED_POOL():
    pool = json.loads((_ROOT / "results/raidex_pool.json").read_text())
    mapped = {m["together_served_name"] for m in pool["models"]
              if m["together_served_name"]
              and str(m["rai_coverage"]) in ("9/9", "9")}
    for m in R.RAIDEX_MAPPED:
        assert m in mapped, f"{m} claims a raidex score the pool does not have"
    for m in R.RATE_ONLY:
        assert m not in mapped, f"{m} has a raidex score and should correlate"


def test_the_SIZE_pair_is_kept_apart_from_the_VERSION_ladders():
    """gpt-oss 20b/120b cannot speak to post-training. Pooling it with the
    version contrasts would be the denominator-provenance failure again."""
    ladder = {m for pair in R.VERSION_LADDERS.values() for m in pair}
    size = {m for pair in R.SIZE_PAIRS.values() for m in pair}
    assert not (ladder & size)
    assert size == {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
    for pair in list(R.VERSION_LADDERS.values()) + list(R.SIZE_PAIRS.values()):
        for m in pair:
            assert m in R.COHORT or m in R.EXISTING, m


# --------------------------------------------------------------------------
# Generation-3 plumbing and the pin.
# --------------------------------------------------------------------------

def test_a_cell_without_the_generation3_flag_is_REFUSED():
    ok = {"served_name": "m", "eden_arm": "A1", "eden_level": "LAT",
          "terminal_at_zero": True}
    R.assert_generation3(ok)
    for bad in ({**ok, "terminal_at_zero": False},
                {k: v for k, v in ok.items() if k != "terminal_at_zero"}):
        with pytest.raises(SystemExit, match="NOT GENERATION 3"):
            R.assert_generation3(bad)


def test_per_arm_m_and_the_COMP_arm():
    assert R.episodes_for("A1") == 72 and R.episodes_for("A0") == 24
    # COMP carries no forbidden item, so A1 would emit "The None is not to be
    # eaten." — it is A0-only, as round 4 established.
    assert {a for _, a, _ in R.comp_cells()} == {"A0"}
    assert len(R.comp_cells()) == len(R.COHORT) == 14


def test_seeds_are_disjoint_from_every_block_on_disk():
    import glob
    used = set()
    for f in glob.glob(str(_ROOT / "results/eden_e*.json")):
        d = json.loads(Path(f).read_text())
        if not any(d.get("meta", {}).get(k) for k in
                   ("round2_pin", "round3_pin", "round6_pin", "round7_pin",
                    "round8_pin", "round9_pin")):
            continue
        used |= {r["seed"] for r in d.get("runs", []) if "seed" in r}
    lat = set(range(R.SEED0, R.SEED0 + R.EPISODES_A1))
    comp = set(range(R.COMP_SEED0, R.COMP_SEED0 + R.COMP_EPISODES))
    assert not (lat & used) and not (comp & used) and not (lat & comp)


def test_the_pin_covers_BOTH_locks_and_the_class_rule():
    assert set(R.world_lock_paths()) == {
        "worlds/world_eden_LAT/BUILD.lock.json",
        "worlds/world_eden_COMP/BUILD.lock.json"}
    assert "seahaven/eden/outcome.py" in R.ARTIFACTS
    assert R.current_hash() == R.PINNED_ROUND10_HASH


def test_moving_a_POLE_breaks_the_pin():
    """The classification is only as frozen as the poles it compares against."""
    before = R.current_hash()
    orig = R.FLOOR_POLE
    try:
        R.FLOOR_POLE = (1, 192)
        assert R.current_hash() != before
        with pytest.raises(SystemExit, match="ROUND-10 PIN BROKEN"):
            R.assert_pinned()
    finally:
        R.FLOOR_POLE = orig
    assert R.current_hash() == before
