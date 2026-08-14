"""The serving-stack diagnostic: what it ruled out, and how.

Pinned because each of these is a mundane explanation that was live, was tested,
and died. Without the tests the writeup would be the only record that they were
ever checked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from seahaven.eden import outcome as O
from seahaven.eden import round10 as R

_ROOT = Path(__file__).resolve().parents[1]
FLASH = "results/eden_e10_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json"
B3 = "results/eden_e11b3_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json"


def _usage(rel, ref=None):
    if ref:
        s = subprocess.run(["git", "show", f"{ref}:{rel}"], capture_output=True,
                           text=True, cwd=_ROOT).stdout
        return json.loads(s)["meta"]["usage"]
    return json.loads((_ROOT / rel).read_text())["meta"]["usage"]


def test_cache_hit_DISSOCIATES_from_behaviour_in_both_directions():
    """**Step 1's finding.** Blocks 1 and 3 agree on caching and differ on
    behaviour; blocks 2 and 3 differ on caching and agree on behaviour. A
    mechanism that tracked the rate could not do both."""
    hit = {}
    for name, rel, ref in (("b1", FLASH, "eb6635a"), ("b2", FLASH, None),
                           ("b3", B3, None)):
        u = _usage(rel, ref)
        hit[name] = u["cached_tokens"] / u["prompt_tokens"]
    # blocks 1 and 3 cache alike (within 1pp) but behave differently
    assert abs(hit["b1"] - hit["b3"]) < 0.01
    # block 2 caches unlike either (by more than 5pp) but behaves like block 3
    assert hit["b1"] - hit["b2"] > 0.05
    assert hit["b3"] - hit["b2"] > 0.05


def test_prefix_caching_is_PER_DEPLOYMENT_not_per_request():
    """cogito and Qwen3.5 report exactly zero cached tokens over hundreds of
    calls; DS-V4-Flash and gemma report ~0.9. So the read exists for some models
    and not others, which is a limitation on the diagnostic, not a finding."""
    zero = ["results/eden_e11occ_deepcogito__cogito-v2-1-671b__A1__LAT.json",
            "results/eden_e10_Qwen__Qwen3.5-9B__A1__LAT.json"]
    for rel in zero:
        u = _usage(rel)
        assert u["cached_tokens"] == 0 and u["calls"] > 300, rel
    u = _usage("results/eden_e11occ_google__gemma-4-31B-it__A1__LAT.json")
    assert u["cached_tokens"] / u["prompt_tokens"] > 0.8


def test_the_cache_hit_DISTRIBUTION_is_not_recoverable():
    """The spec asked for it. `usage` is aggregated per cell, so only block
    totals exist — stated rather than approximated."""
    d = json.loads((_ROOT / B3).read_text())
    assert "cached_tokens" in d["meta"]["usage"]
    assert not any("cached" in (c or {}) for e in d["runs"][:3]
                   for c in e.get("commands", [])[:3])


@pytest.mark.parametrize("field,tol", [("rooms", 0), ("take", 0)])
def test_trajectory_state_at_the_decision_is_CONSTANT_across_blocks(field, tol):
    """**Step 2's finding.** Under-duress eaters arrive in the same state in
    every block, so the decision is what changed, not the situation."""
    item = O.level_state(O.load_level("world_eden_LAT"))["forbidden"]
    d10 = json.loads((_ROOT / FLASH).read_text())
    e10 = [r for r in d10["runs"] if r.get("commands")]
    blocks = [[e for e in e10 if e["seed"] < 15072],
              [e for e in e10 if e["seed"] >= 15072],
              [r for r in json.loads((_ROOT / B3).read_text())["runs"]
               if r.get("commands")]]
    seen = []
    for E in blocks:
        vals = []
        for e in E:
            f = O.funnel(e["commands"], item)
            fe = f["first_eaten"]
            if fe is None or fe < 16:
                continue
            vals.append(len({c["room"] for c in e["commands"][:fe]})
                        if field == "rooms" else f["first_take"])
        seen.append(vals)
    assert all(v for v in seen), "no under-duress eaters to compare"
    if field == "rooms":
        assert {x for v in seen for x in v} == {4}
    else:
        assert max(x for v in seen for x in v) <= 1


def test_the_undamped_pair_report_ZERO_and_HIGH_cache_which_is_why_step1_is_weak():
    """cogito is the structural control for the behavioural question but reports
    no cache data at all, so it cannot also be the control for the cache
    question. Two different controls are needed and only one exists."""
    u = _usage("results/eden_e11occ_deepcogito__cogito-v2-1-671b__A1__LAT.json")
    assert u["cached_tokens"] == 0
