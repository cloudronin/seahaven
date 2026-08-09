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
    """TRAP 16: narratives that vary but describe the WRONG run must not score
    as signal. Identical narratives would make the test vacuous rather than
    negative, which is a different (and separately handled) case."""
    from seahaven.fidelity.score import permutation_check

    acts = {"m": None, "t": None}
    truths = [{"m": i % 2 == 0, "t": i % 3 == 0} for i in range(24)]
    texts = [("moved " if x["m"] else "") + ("took " if x["t"] else "") + "here"
             for x in truths]
    paired = list(zip(texts[1:] + texts[:1], truths))       # rotated: all mispaired
    c = permutation_check(paired, lambda n, a: ("moved" in n if a == "m" else "took" in n),
                          acts, n_shuffles=200)
    assert c["has_signal"] is False


def test_permutation_reports_vacuous_rather_than_negative():
    """No variation in ground truth means the test cannot discriminate. That is
    UNKNOWN, not 'no signal' — blaming the measurement for a property of the
    sample would be wrong."""
    from seahaven.fidelity.score import permutation_check

    acts = {"m": None, "t": None}
    paired = [(f"narrative {i}", {"m": True, "t": False}) for i in range(8)]
    c = permutation_check(paired, lambda n, a: True, acts, n_shuffles=50)
    assert c["has_signal"] is None
    assert c["kind"] == "no_variation_in_ground_truth"


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

    truths = [{"movement": i % 2 == 0, "taking": i % 3 == 0, "examining": True,
               "inventory": False, "dropping": False} for i in range(24)]
    texts = [("I walked between rooms. " if x["movement"] else "")
             + ("I picked up an object. " if x["taking"] else "") + "I examined it."
             for x in truths]
    paired = list(zip(texts[1:] + texts[:1], truths))       # every pairing wrong
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


# --- endpoint capability negotiation -----------------------------------------

class _FakeEndpoint:
    """Records which request forms an endpoint accepts, to test the fallback."""

    def __init__(self, reject_kwargs=False, reject_system=False):
        self.reject_kwargs, self.reject_system = reject_kwargs, reject_system
        self.attempts = []

    def post(self, payload):
        has_kwargs = "chat_template_kwargs" in payload
        has_system = any(m["role"] == "system" for m in payload["messages"])
        self.attempts.append(("kwargs" if has_kwargs else
                              "plain" if has_system else "merged"))
        if has_kwargs and self.reject_kwargs:
            raise RuntimeError("HTTP 400: Bad Request")
        if has_system and self.reject_system:
            raise RuntimeError("HTTP 400: Bad Request")
        return {"choices": [{"message": {"content": "ok"}}]}


def _wire(ep, fake):
    ep._post = lambda path, payload: fake.post(payload)
    return ep


def test_endpoint_falls_back_when_template_kwargs_rejected():
    """Mistral-7B rejects chat_template_kwargs with a 400. Adding that key to fix
    a Qwen problem broke two other models in a real sweep."""
    from seahaven.fidelity.endpoint import Endpoint

    fake = _FakeEndpoint(reject_kwargs=True)
    ep = _wire(Endpoint("http://x/v1", "m"), fake)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert ep.chat(msgs) == "ok"
    assert fake.attempts == ["kwargs", "plain"]


def test_endpoint_falls_back_to_merged_when_system_rejected():
    """Gemma-2 rejects a system role outright."""
    from seahaven.fidelity.endpoint import Endpoint

    fake = _FakeEndpoint(reject_kwargs=True, reject_system=True)
    ep = _wire(Endpoint("http://x/v1", "m"), fake)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert ep.chat(msgs) == "ok"
    assert fake.attempts == ["kwargs", "plain", "merged"]


def test_endpoint_remembers_what_worked():
    """Renegotiating every call would triple the request count."""
    from seahaven.fidelity.endpoint import Endpoint

    fake = _FakeEndpoint(reject_kwargs=True)
    ep = _wire(Endpoint("http://x/v1", "m"), fake)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    ep.chat(msgs); fake.attempts.clear()
    ep.chat(msgs)
    assert fake.attempts == ["plain"]        # no retry of the rejected form


def test_merged_fallback_preserves_system_text():
    """Dropping the system turn would silently remove framing that carries real
    effects; it must be folded into the first user turn instead."""
    from seahaven.fidelity.endpoint import Endpoint

    seen = {}

    class F(_FakeEndpoint):
        def post(self, payload):
            if any(m["role"] == "system" for m in payload["messages"]):
                raise RuntimeError("HTTP 400: Bad Request")
            seen["msgs"] = payload["messages"]
            return {"choices": [{"message": {"content": "ok"}}]}

    fake = F(reject_kwargs=True, reject_system=True)
    ep = _wire(Endpoint("http://x/v1", "m"), fake)
    ep.chat([{"role": "system", "content": "SYSTEXT"},
             {"role": "user", "content": "USERTEXT"}])
    assert "SYSTEXT" in seen["msgs"][0]["content"]
    assert "USERTEXT" in seen["msgs"][0]["content"]


# --- gate -1 stratification (TRAP 17) ----------------------------------------

def _length_confounded(n_per_len=3, lengths=(2, 10)):
    """Narratives that name entities in proportion to episode LENGTH only, with
    no correspondence to which entities a given run actually touched.

    Unstratified shuffling credits this as signal, because a long narrative
    paired with a short run's ground truth produces fabrications. Stratified
    shuffling should not.
    """
    paired, strata = [], []
    keys = [f"e{i}" for i in range(6)]
    for L in lengths:
        for r in range(n_per_len):
            # long runs perform (and name) more entities; WHICH ones is rotated
            # so the pairing carries no entity-level information
            k = 4 if L > 5 else 1
            perf = {key: (i < k) for i, key in enumerate(keys)}
            named = keys[(r + 1) % len(keys):][:k]     # rotated: wrong entities
            paired.append(("I did " + ", ".join(named) + ".", perf))
            strata.append(L)
    return paired, strata, keys


def test_unstratified_null_credits_length_as_signal():
    from seahaven.fidelity.score import permutation_check

    paired, strata, keys = _length_confounded()
    det = lambda nar, k: k in nar
    un = permutation_check(paired, det, keys, n_shuffles=300)
    st = permutation_check(paired, det, keys, n_shuffles=300, strata=strata)
    # The unstratified null must be the more permissive of the two.
    assert un["lift"] >= st["lift"]
    assert st["stratified"] is True and un["stratified"] is False


def test_all_singleton_strata_refuse_rather_than_pass():
    """One run per stratum makes within-stratum shuffling the identity. That is
    UNKNOWN, not signal — and not a failure of the measurement either."""
    from seahaven.fidelity.score import permutation_check

    paired = [(f"narrative {i}", {"a": i % 2 == 0, "b": True}) for i in range(6)]
    c = permutation_check(paired, lambda n, k: True, ["a", "b"],
                          n_shuffles=50, strata=list(range(6)))
    assert c["has_signal"] is None and c["kind"] == "strata_all_singleton"


def test_step_schedule_keeps_three_runs_per_length():
    """Two per stratum caps the achievable p at 0.059 and can never reach
    significance; three is the minimum workable design."""
    from collections import Counter

    from seahaven.fidelity.runner import STEP_SCHEDULE, _steps_for

    counts = Counter(_steps_for(i, 30) for i in range(len(STEP_SCHEDULE)))
    assert min(counts.values()) >= 3, counts
    assert len(counts) >= 3, "need several distinct lengths for varying ground truth"


def test_strata_length_mismatch_is_an_error_not_silent():
    from seahaven.fidelity.score import permutation_check

    paired = [("x", {"a": True})] * 4
    try:
        permutation_check(paired, lambda n, k: True, ["a"], strata=[1, 2])
    except ValueError as e:
        assert "strata" in str(e)
    else:
        raise AssertionError("mismatched strata length must raise")
