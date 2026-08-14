"""The possibility bar's four criteria, on inputs whose answers are known.

**Written and passing before the survey's numbers existed.** A criterion
debugged after seeing which rung it rejects is not a criterion, and this project
has spent enough entries on the difference. Each case below is constructed so
the right answer is arithmetic rather than judgement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The one-off scripts moved to archive/ when the CLI absorbed the live
# family. Both roots are on the path so a test need not know which side
# of that move its subject landed on.
for _d in ("scripts", "archive"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _d))

from possibility_bar import (WORLDS, criterion_1, criterion_2,  # noqa: E402
                             criterion_3, criterion_4)

from seahaven.fidelity.flag import PHRASING_IDS  # noqa: E402


def make_models(spec: dict[str, tuple[bool, bool]], margins=None) -> dict:
    """{lab: (flagged_on_v0, flagged_on_v2)} -> the survey's `models` shape."""
    out = {}
    for lab, flags in spec.items():
        m = (margins or {}).get(lab)
        per = [{"world": w, "flagged": f, "worst_phrasing": "p1",
                "worst_adherence": 0.0,
                "margin": m if m is not None else (-1.0 if f else 1.0)}
               for w, f in zip(WORLDS, flags)]
        label = ("FLAG" if all(flags) else "PASS" if not any(flags) else "UNSTABLE")
        out[lab] = {"label": label, "per_world": per}
    return out


def flat_episodes(adherence: float, n: int = 20, per: int = 100):
    """`n` identical episodes at a given adherence — zero bootstrap variance."""
    bad = round(per * (1 - adherence / 100.0))
    return [[bad, per] for _ in range(n)]


def split_episodes(lo: float, hi: float, n: int = 20, per: int = 100):
    """Half at `lo`, half at `hi` — pooled midpoint with real spread."""
    return (flat_episodes(lo, n // 2, per) + flat_episodes(hi, n // 2, per))


def test_the_fixtures_mean_what_they_say():
    """Guards the guards: a fixture that quietly drifts invalidates every case."""
    assert flat_episodes(90.0) == [[10, 100]] * 20
    eps = split_episodes(85.0, 97.0, 8)
    assert len(eps) == 8
    bad, n = sum(e[0] for e in eps), sum(e[1] for e in eps)
    assert 100.0 * (1 - bad / n) == pytest.approx(91.0)


# --- criterion 1: non-trivial split -----------------------------------------

def test_c1_needs_two_on_the_minority_side():
    two = make_models({"A": (True, True), "B": (True, True),
                       "C": (False, False), "D": (False, False),
                       "E": (False, False)})
    assert criterion_1(two)[0]


def test_c1_rejects_a_lone_flag():
    one = make_models({"A": (True, True), "B": (False, False),
                       "C": (False, False)})
    ok, note = criterion_1(one)
    assert not ok and "minority side has 1" in note


def test_c1_rejects_a_degenerate_split():
    """Everything on one side is the failure mode the criterion exists for."""
    assert not criterion_1(make_models({l: (False, False) for l in "ABCDE"}))[0]
    assert not criterion_1(make_models({l: (True, True) for l in "ABCDE"}))[0]


def test_c1_counts_the_smaller_side_whichever_it_is():
    """Six FLAG and two PASS is as non-trivial as the reverse."""
    lopsided = make_models({**{l: (True, True) for l in "ABCDEF"},
                            **{l: (False, False) for l in "GH"}})
    ok, note = criterion_1(lopsided)
    assert ok and "minority side has 2" in note


# --- criterion 2: cross-world -----------------------------------------------

def test_c2_passes_when_worlds_agree():
    assert criterion_2(make_models({"A": (True, True), "B": (False, False)}))[0]


def test_c2_fails_on_a_single_disagreement():
    ok, note = criterion_2(make_models({"A": (True, False), "B": (False, False)}))
    assert not ok and "differ on ['A']" in note


# --- criterion 3: seed stability under a joint bootstrap --------------------

def _c3_inputs(model_adherence: dict, anchor: float = 90.0, spread=None):
    """`spread` maps a lab to an (lo, hi, n) episode split; others are flat."""
    anchors = {w: {"adherence": anchor,
                   "episode_counts": flat_episodes(anchor, n=40)}
               for w in WORLDS}
    eps = {}
    for lab, adh in model_adherence.items():
        s = (spread or {}).get(lab)
        for w in WORLDS:
            for p in PHRASING_IDS:
                eps[(lab, p, w)] = (split_episodes(*s) if s else flat_episodes(adh))
    models = make_models(
        {lab: (adh <= anchor, adh <= anchor) for lab, adh in model_adherence.items()},
        margins={lab: adh - anchor for lab, adh in model_adherence.items()})
    return models, anchors, eps


def test_c3_passes_when_every_margin_is_far_from_the_boundary():
    models, anchors, eps = _c3_inputs({"A": 70.0, "B": 99.0})
    ok, note, detail = criterion_3(models, anchors, eps)
    assert ok, note
    assert detail["A|world_v0"]["flag_share"] == 1.0
    assert detail["B|world_v0"]["flag_share"] == 0.0


def test_c3_fails_when_a_model_straddles_the_boundary():
    """The fragile model sits just ABOVE the anchor, not at it.

    `min` over five noisy per-phrasing estimates is biased low — the minimum of
    five draws lands below the minimum of their five means — so the worst-case
    rule pushes resampled margins downward. A model *at* the anchor therefore
    flags robustly rather than straddling, and the cells that genuinely straddle
    are the ones whose point margin is small and positive.

    That asymmetry is a property of the worst-case-over-phrasings rule, not of
    the bootstrap: **FLAGs are sticky and PASSes are fragile.**
    """
    models, anchors, eps = _c3_inputs(
        {"A": 70.0, "B": 91.0}, spread={"B": (85.0, 97.0, 8)})
    ok, note, detail = criterion_3(models, anchors, eps)
    assert not ok
    assert "B|world_v0" in note
    lo, hi = detail["B|world_v0"]["ci90"]
    assert lo < 0 < hi, f"B's interval should span the boundary, got [{lo}, {hi}]"
    assert detail["A|world_v0"]["stable"], "A is far away and must stay stable"


def test_c3_a_model_at_the_anchor_is_stable_not_borderline():
    """Documents the surprising half of the asymmetry above.

    Written down because the intuitive expectation — margin zero means maximum
    instability — is wrong here, and a later reader finding `flag_share` near 1
    for a model sitting exactly on the line should not conclude the bootstrap is
    broken.
    """
    models, anchors, eps = _c3_inputs(
        {"B": 90.0}, spread={"B": (85.0, 95.0, 20)})
    ok, _, detail = criterion_3(models, anchors, eps)
    assert ok
    assert detail["B|world_v0"]["flag_share"] > 0.9, \
        "the min over five phrasings should keep it below the line"


def test_c3_resamples_the_anchor_too():
    """A noisy anchor must be able to destabilise an otherwise clean margin.

    Holding the anchor fixed would treat it as known exactly — the assumption
    the SE target exists to deny — and would make this criterion look far
    stronger than it is.
    """
    models, anchors, eps = _c3_inputs({"A": 92.0})
    steady = criterion_3(models, anchors, eps)[2]["A|world_v0"]["ci90"]

    for w in WORLDS:                      # same anchor location, real spread
        anchors[w]["episode_counts"] = split_episodes(80.0, 100.0, n=40)
    noisy = criterion_3(models, anchors, eps)[2]["A|world_v0"]["ci90"]

    assert (noisy[1] - noisy[0]) > (steady[1] - steady[0]), \
        "anchor uncertainty must widen the margin interval"


# --- criterion 4: leave-one-phrasing-out ------------------------------------

def _c4_inputs(per_model: dict, anchor: float = 90.0):
    anchors = {w: {"adherence": anchor} for w in WORLDS}
    per_phrasing = {f"{lab}|{w}": dict(zip(PHRASING_IDS, vals))
                    for lab, vals in per_model.items() for w in WORLDS}
    models = make_models({lab: (min(v) - anchor <= 0,) * 2
                          for lab, v in per_model.items()})
    return models, anchors, per_phrasing


def test_c4_passes_when_a_flag_does_not_rest_on_one_phrasing():
    models, anchors, per = _c4_inputs({
        "A": [80.0] * 5,                       # flags under every phrasing
        "C": [99.0] * 5,                       # never flags
    })
    ok, note, _ = criterion_4(models, anchors, per)
    assert ok, note


def test_c4_fails_when_one_phrasing_carries_the_flag():
    models, anchors, per = _c4_inputs({
        "A": [80.0] * 5,
        "B": [95.0, 95.0, 95.0, 95.0, 85.0],   # flags only because of p5
        "C": [99.0] * 5,
    })
    ok, note, detail = criterion_4(models, anchors, per)
    assert not ok
    assert "drop p5" in note and "'B'" in note
    assert detail["drop_p5|world_v0"] == ["A"], "B should drop out"
    assert detail["drop_p1|world_v0"] == ["A", "B"], "and stay in otherwise"


def test_c4_is_one_directional():
    """Leave-one-out takes a min over four, so margins can only rise.

    A model can lose a flag under LOO and can never gain one. If a test ever
    shows the reverse, the recomputation is not taking a minimum.
    """
    models, anchors, per = _c4_inputs({
        "A": [80.0] * 5, "B": [95.0, 95.0, 95.0, 95.0, 85.0], "C": [99.0] * 5,
    })
    base = {lab for lab, m in models.items() if m["per_world"][0]["flagged"]}
    _, _, detail = criterion_4(models, anchors, per)
    for key, got in detail.items():
        assert set(got) <= base, f"{key} gained a flag under leave-one-out: {got}"
