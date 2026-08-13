"""The raidex pool, and the assumption it forced.

**Standing requirement 1 cannot be satisfied here and the tests say so.** raidex
ids are provider-prefixed; Together strings are not; EXACT-string matching yields
ZERO models. The round proceeds on a named, untested provider-invariance
assumption, and these tests exist so that assumption cannot be quietly forgotten
once the correlations look tidy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
POOL = _ROOT / "results/raidex_pool.json"

pytestmark = pytest.mark.skipif(not POOL.exists(),
                                reason="raidex pool not ingested yet")


@pytest.fixture(scope="module")
def pool():
    return json.loads(POOL.read_text())


def test_ZERO_exact_string_matches_which_is_why_the_assumption_exists(pool):
    """**The load-bearing fact of the whole join.**

    If this ever becomes non-zero, the assumption is no longer needed for those
    models and the correlates should be re-derived on the exact-match subset
    rather than continuing to lean on it.
    """
    assert pool["exact_string_matches"] == []
    for m in pool["models"]:
        assert m["exact_string_match"] is False


def test_the_assumption_is_written_into_the_ARTIFACT_not_only_the_code(pool):
    """A reader who receives `raidex_pool.json` alone must still be told."""
    a = pool["provider_invariance_assumption"].lower()
    assert "untested" in a
    assert "zero" in a
    for w in ("openrouter", "sambanova", "together"):
        assert w in a


def test_every_record_carries_the_stack_that_MEASURED_it(pool):
    """`measured_on` exists so no correlation can silently lose which serving
    stack produced its x-axis. Only one model in the pool was measured on
    Together, and that number is what makes the caveat concrete."""
    for m in pool["models"]:
        assert m["measured_on"], m["raidex_id"]
        assert m["raidex_id"].startswith(m["measured_on"] + "/")
    on_together = [m for m in pool["models"] if m["measured_on"] == "together_ai"]
    assert len(on_together) == 1, (
        f"{len(on_together)} measured on Together; the caveat's strength changed")


def test_normalisation_does_NOT_bridge_versions_or_sizes():
    """The join must collapse serving suffixes and nothing else.

    Round 4 refused GLM-5.1 for GLM-5.2 and Nemotron SUPER for Ultra. A join key
    that erased a version digit would re-admit exactly those substitutions under
    a different name.
    """
    import importlib.util
    import sys
    key = "_raidex_build"
    if key not in sys.modules:
        sp = importlib.util.spec_from_file_location(
            key, _ROOT / "scripts" / "build_raidex_pool.py")
        mod = importlib.util.module_from_spec(sp)
        sys.modules[key] = mod
        sp.loader.exec_module(mod)
    n = sys.modules[key].normalise
    assert n("openrouter/z-ai/glm-5.1") != n("openrouter/z-ai/glm-5.2")
    assert n("openrouter/z-ai/glm-5") != n("openrouter/z-ai/glm-5.1")
    assert n("a/gpt-oss-20b") != n("a/gpt-oss-120b")
    assert n("a/nemotron-3-super-100b") != n("a/nemotron-3-ultra-550b-a55b")
    # but serving suffixes and separators DO collapse
    assert n("sambanova/gemma-4-31B-it") == n("google/gemma-4-31B")
    assert n("openrouter/qwen/qwen-2.5-coder-32b-instruct") == \
        n("Qwen/Qwen2.5-Coder-32B-Instruct")


def test_an_AMBIGUOUS_match_is_only_resolved_by_a_prior_commitment(pool):
    """`gemma-4-31B-it` is listed on Together twice — `google/...` and
    `pearl-ai/...`, two resellers of the same weights. Choosing between them on
    plausibility is the near-miss the requirement refuses. The program has served
    `google/...` for 96 committed episodes, so that commitment decides it.

    Anything with two candidates and NO prior commitment must stay unmapped.
    """
    from seahaven.eden.round4 import COHORT
    resolved = [m for m in pool["models"]
                if m.get("disambiguated_by_prior_commitment")]
    for m in resolved:
        assert len(m["together_candidates"]) > 1
        assert m["together_served_name"] in COHORT, (
            f"{m['raidex_id']} was disambiguated to a string the program never "
            "committed episodes against")
    assert pool["ambiguous_matches"] == [], pool["ambiguous_matches"]


def test_the_mapping_is_INJECTIVE_so_two_raidex_rows_cannot_share_a_model(pool):
    """Two raidex records mapping to one Together string would double-weight
    that model in every correlation."""
    served = [m["together_served_name"] for m in pool["models"]
              if m["together_served_name"]]
    dupes = {s for s in served if served.count(s) > 1}
    assert not dupes, f"non-injective mapping: {dupes}"


def test_only_FULL_COVERAGE_records_are_usable(pool):
    """A partial composite is an average over a different dimension subset and is
    not comparable across models."""
    usable = [m for m in pool["models"]
              if m["together_served_name"]
              and str(m["rai_coverage"]) in ("9/9", "9")]
    assert len(usable) == pool["n_mapped_full_coverage"]
    for m in usable:
        assert len(m["dimension_scores"]) == 8, m["raidex_id"]


def test_the_HIGH_POLE_is_absent_from_raidex_and_that_truncates_the_correlates(
        pool):
    """**Recorded before any correlation is computed.**

    cogito is the high pole at 0.375 and Llama is a floor member, and NEITHER is
    in raidex. So every rate-versus-RAI correlation runs on a cohort whose top is
    missing, which attenuates it in an unknown direction. That is a property of
    raidex's coverage, not a choice, and it belongs beside the rho.
    """
    served = {m["together_served_name"] for m in pool["models"]}
    assert "deepcogito/cogito-v2-1-671b" not in served
    assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" not in served
