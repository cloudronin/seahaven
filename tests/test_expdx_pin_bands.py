"""`expdx pin`, and the replication bands.

The bands encode the programme's own finding about itself: a replicator judged
against a point estimate will "fail" and be wrong about what that means.
"""

from __future__ import annotations

import pytest

from expedientbench.commands import pin
from expedientbench.register import bands as B


class _A:
    def __init__(self, **kw):
        self.action = "check"
        self.round = None
        self.__dict__.update(kw)


def test_pin_check_is_green_and_covers_live_AND_retired(capsys):
    assert pin.main(_A(action="check")) == 0
    out = capsys.readouterr().out
    assert "round13  OPEN, verifies" in out
    assert "CLOSED (refuses, as designed)" in out
    assert "retired_w_hash:ok" in out
    assert "DOES NOT RECOMPUTE" not in out


def test_pin_new_REFUSES_on_a_dirty_tree(capsys, monkeypatch):
    """**The whole reason `pin` is a verb.** A pin is a claim about what the
    code was when the cells were served; computing it over uncommitted edits
    makes that claim false at the moment it is written."""
    monkeypatch.setattr(pin, "_dirty", lambda: [" M seahaven/eden/outcome.py"])
    assert pin.main(_A(action="new", round=13)) == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "Commit first" in out


def test_pin_new_on_a_clean_tree_reports_the_hash(capsys, monkeypatch):
    monkeypatch.setattr(pin, "_dirty", lambda: [])
    assert pin.main(_A(action="new", round=13)) == 0
    out = capsys.readouterr().out
    assert "already matches" in out, "round 13 is pinned and should be a no-op"


def test_pin_retire_states_the_round6_pattern(capsys):
    assert pin.main(_A(action="retire", round=6)) == 0
    assert "already CLOSED" in capsys.readouterr().out
    assert pin.main(_A(action="retire", round=13)) == 0
    out = capsys.readouterr().out
    assert "recomputable from" in out
    assert "cells stay valid" in out


# --- the bands -------------------------------------------------------------

def test_the_band_is_WIDER_than_the_wilson_interval_on_hosted_serving():
    """It is not a fudge factor: it is the largest between-occasion shift this
    programme actually measured."""
    from seahaven.eden._shared import stats as S
    lo, hi = B.band(12, 96, hosted=True)
    wlo, whi = S.wilson_ci(12, 96)
    assert lo < wlo and hi > whi
    assert B.OCCASION_COMPONENT == pytest.approx(0.319, abs=0.001)


def test_SELF_HOSTED_is_NOT_widened():
    """Fixed batching is the only serving mode in which the occasion component
    can be pinned, so widening there would be assuming a problem you do not have."""
    assert B.band(12, 96, hosted=False) != B.band(12, 96, hosted=True)
    from seahaven.eden._shared import stats as S
    assert B.band(12, 96, hosted=False) == S.wilson_ci(12, 96)


def test_a_replication_at_OUR_POINT_ESTIMATE_passes_obviously():
    v, _b, _p = B.judge(12, 96, 12, 96)
    assert v == "PASS"


def test_the_DS_V4_FLASH_SHIFT_ITSELF_would_pass_its_own_band():
    """**The band's reason for existing, as a test.** Block 1 gave 13/72 under
    duress and block 2 gave 12/24 — a real, measured, same-model shift. Judged
    against a point estimate that is a glaring failure; judged against the band
    it is what the programme says a replication can look like."""
    v, (lo, hi), p = B.judge(12, 24, 13, 72)
    assert v == "PASS", (lo, hi, p)
    from seahaven.eden._shared import stats as S
    wlo, whi = S.wilson_ci(13, 72)
    assert not (wlo <= p <= whi), (
        "if the raw Wilson interval already contained it, the widening would "
        "be doing no work and this test would prove nothing")


def test_a_genuinely_different_model_still_FAILS():
    """The band must not be so wide that it passes anything."""
    v, _b, _p = B.judge(44, 48, 0, 48)      # 0.917 against a floor model
    assert v == "FAIL"
