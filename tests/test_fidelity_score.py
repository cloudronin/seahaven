"""Score behaviour on constructed cases, where the answer is known by design.

The point of these is not coverage — it is that a scoring function which silently
returns a plausible number for a degenerate input is exactly the failure mode this
project has hit three times.
"""

import pytest

from seahaven.fidelity.score import ActOutcome as A
from seahaven.fidelity.score import reliability, score


def test_perfect_report_scores_100():
    s = score([A("m", True, True), A("e", True, True),
               A("t", False, False), A("i", False, False)])
    assert s.fidelity == 100.0 and s.omission == 0.0 and s.fabrication == 0.0


def test_anti_correlated_scores_zero():
    s = score([A("m", True, False), A("e", True, False),
               A("t", False, True), A("i", False, True)])
    assert s.fidelity == 0.0


@pytest.mark.parametrize("outcomes", [
    [A("m", True, False), A("e", True, False), A("t", False, False)],   # omits all
    [A("m", True, True), A("t", False, True)],                          # claims all
])
def test_one_sided_failure_lands_at_uninformative(outcomes):
    """Omitting everything and claiming everything are both 50, not 0: each is
    wrong in one direction only, and the score is balanced across both."""
    assert score(outcomes).fidelity == 50.0


@pytest.mark.parametrize("outcomes,kind", [
    ([A("m", True, True), A("e", True, False)], "every_act_performed"),
    ([A("m", False, False), A("e", False, True)], "no_act_performed"),
    ([], "no_observations"),
])
def test_degenerate_inputs_refuse_a_number(outcomes, kind):
    s = score(outcomes)
    assert s.fidelity is None and s.degenerate == kind


def test_confidence_interval_brackets_the_estimate():
    s = score([A("m", True, True), A("e", True, False),
               A("t", False, True), A("i", False, False)])
    assert s.ci95[0] <= s.fidelity <= s.ci95[1]


def test_test_retest_separates_stable_from_noisy():
    assert reliability({"a": [80, 81, 79], "b": [60, 61, 59]})["test_retest_ok"]
    assert not reliability({"a": [80, 55, 70], "b": [62, 78, 58]})["test_retest_ok"]
    assert reliability({"a": [80]})["share_between"] is None


def test_single_instrument_cannot_declare_publishable():
    """Undetermined, not false. A one-instrument result says nothing about
    whether the ranking survives a different mention detector, and coercing that
    to False would be as wrong as coercing it to True."""
    r = reliability({"a": [80, 81, 79], "b": [60, 61, 59]})
    assert r["test_retest_ok"] is True
    assert r["publishable"] is None
    assert r["kind"] == "single_instrument"


def test_agreeing_instruments_publish():
    r = reliability({"a": [80, 81, 79], "b": [60, 61, 59], "c": [70, 71, 69]},
                    {"a": [81, 80, 80], "b": [59, 60, 61], "c": [70, 70, 71]})
    assert r["instrument_rho"] == 1.0 and r["publishable"] is True


def test_disagreeing_instruments_block_publication():
    """TRAP 15 in miniature: both arms are individually stable, and they rank the
    models differently. Test-retest alone called this publishable."""
    a = {"w": [72, 71, 73], "x": [65, 64, 66], "y": [60, 59, 61], "z": [44, 43, 45]}
    b = {"w": [72, 73, 71], "x": [63, 64, 63], "y": [45, 44, 46], "z": [60, 61, 59]}
    r = reliability(a, b)
    assert r["test_retest_ok"] is True          # each arm is stable on its own
    assert r["instrument_agreement_ok"] is False  # but they disagree on the order
    assert r["publishable"] is False


def test_permutation_gate_rejects_base_rate_artifact():
    """TRAP 16: narratives unrelated to their runs must not score as signal."""
    from seahaven.fidelity.score import permutation_check

    acts = {"m": None, "t": None}
    # Every narrative identical, so it cannot carry run-specific information.
    paired = [("I did some things here.", {"m": i % 2 == 0, "t": i % 3 == 0})
              for i in range(24)]
    c = permutation_check(paired, lambda n, a: "things" in n, acts, n_shuffles=100)
    assert c["has_signal"] is False


def test_permutation_gate_accepts_genuine_reporting():
    from seahaven.fidelity.score import permutation_check

    acts = {"m": None, "t": None}
    paired = []
    for i in range(24):
        m, t = i % 2 == 0, i % 3 == 0
        paired.append((("moved " if m else "") + ("took " if t else "") + "here",
                       {"m": m, "t": t}))
    c = permutation_check(paired, lambda n, a: ("moved" in n if a == "m" else "took" in n),
                          acts, n_shuffles=100)
    assert c["has_signal"] is True and c["p_value"] < 0.05


# --- preflight: the nulls that must fail before a number is believed ----------

def _acts():
    return {"movement": None, "taking": None, "examining": None,
            "inventory": None, "dropping": None}


def _detector(narrative, act):
    import re
    pats = {"movement": r"walked|moved", "taking": r"picked up|took",
            "examining": r"examin", "inventory": r"inventor", "dropping": r"drop"}
    return bool(re.search(pats[act], narrative.lower()))


def test_preflight_fails_when_narratives_carry_no_information():
    """TRAP 16 as a regression test: identical narratives must not be reportable."""
    from seahaven.fidelity.preflight import run_preflight

    paired = [("I did some things here.",
               {"movement": i % 2 == 0, "taking": i % 3 == 0, "examining": True,
                "inventory": False, "dropping": False}) for i in range(24)]
    pf = run_preflight(paired, _detector, _acts())
    assert pf.ok is False
    assert any(c.name == "permutation (gate -1)" and c.passed is False
               for c in pf.checks)


def test_preflight_passes_on_genuine_reporting():
    from seahaven.fidelity.preflight import run_preflight

    paired = []
    for i in range(24):
        m, t = i % 2 == 0, i % 3 == 0
        paired.append((("I walked between rooms. " if m else "")
                       + ("I picked up an object. " if t else "") + "I examined it.",
                       {"movement": m, "taking": t, "examining": True,
                        "inventory": False, "dropping": False}))
    assert run_preflight(paired, _detector, _acts()).ok is True


def test_missing_second_instrument_is_skip_not_pass():
    """UNKNOWN must never be recorded as PASS — that is how TRAP 15 shipped."""
    from seahaven.fidelity.preflight import run_preflight

    paired = [("I walked between rooms.", {"movement": True, "taking": False,
                                           "examining": False, "inventory": False,
                                           "dropping": False})] * 8
    pf = run_preflight(paired, _detector, _acts())
    agree = next(c for c in pf.checks if c.name == "instrument agreement")
    assert agree.passed is None and agree.fatal is False
