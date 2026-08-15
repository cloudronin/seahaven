"""The reference channel, and the pooled construction it replaces.

The spec this implements proposed pooling each sweep's A0 against pooled
history. That construction produced two published readings, one of which was
wrong and one of which was not a comparison at all. These tests hold the paired
design in place by **computing what pooling would have said** and asserting the
two disagree — a comment saying "pool and you confound composition" decays; a
test that fails when someone re-pools does not.
"""

from __future__ import annotations

import pytest

from seahaven.eden._shared import occasion as OQ
from seahaven.eden._shared.stats import fisher

ALPHA = 0.05


@pytest.fixture(scope="module")
def obs():
    return OQ.observations()


@pytest.fixture(scope="module")
def audit(obs):
    return OQ.audit(alpha=ALPHA, obs=obs)


def test_ALPHA_IS_REQUIRED_because_it_changes_what_was_measured():
    """`mds` defaulted alpha and two rounds disagreed about what they had run.
    The same parameter, the same rule: state it or you do not get a verdict."""
    with pytest.raises(TypeError):
        OQ.verdict_for("LAT", "15")               # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OQ.audit()                                # type: ignore[call-arg]


def test_the_KNOWN_EVENT_is_the_fixture(audit):
    """LAT on 2026-08-14, paired: the same six models that saturated the day
    before. Intervals must not overlap — this is the one certified event."""
    v = next(x for x in audit if (x.world, x.sweep) == ("LAT", "15"))
    assert v.verdict == "EVENT"
    assert len(v.returning) == 6 and len(v.newcomers) == 8
    now, prior = v.rates()
    assert prior == pytest.approx(0.977, abs=0.001)
    assert now == pytest.approx(0.819, abs=0.001)
    lo_now, hi_now = OQ.interval(*v.now)
    lo_pri, hi_pri = OQ.interval(*v.prior)
    assert hi_now < lo_pri, "the certified event: intervals must separate"


def test_W2_and_W3_on_the_event_day_are_NO_ANCHOR_not_QUIET(audit):
    """**The correction that reaches back into published prose.** Both were
    reported as holding steady that day, as the co-located-detector argument
    that the LAT drop was world-specific. Both had ZERO returning models."""
    for world in ("W2", "W3"):
        v = next(x for x in audit if (x.world, x.sweep) == (world, "15"))
        assert v.verdict == "NO-ANCHOR"
        assert v.returning == ()
        assert len(v.newcomers) == 14
        assert v.p is None, "no p may be reported where no test was run"


def test_POOLING_would_have_called_W2_an_EVENT_where_pairing_says_UNJUDGEABLE(obs):
    """The regression that encodes the actual defect.

    Pool W2's event-day cells against W2's pooled history and the test not only
    returns a verdict, it returns a *significant* one — in the upward direction,
    because the new cohort scores higher than the historical one. Three readings
    of one dataset: the spec's one-sided pooling said QUIET, two-sided pooling
    says EVENT, and pairing says there is nothing here to judge. Only the last
    is true.
    """
    here = [o for o in obs if o.world == "W2" and o.sweep == "15"]
    day = min(o.day for o in here)
    hist = [o for o in obs if o.world == "W2" and o.day < day]

    k1, n1 = sum(o.ate for o in here), sum(o.n for o in here)
    k0, n0 = sum(o.ate for o in hist), sum(o.n for o in hist)
    assert n0 > 0, "there IS history; it just shares no model with the sweep"

    pooled_p = fisher(k1, n1, k0, n0)
    assert pooled_p < ALPHA, "pooled construction returns a significant verdict"
    assert k1 / n1 > k0 / n0, "and it points UP — the new cohort scores higher"

    paired = OQ.verdict_for("W2", "15", alpha=ALPHA, obs=obs)
    assert paired.verdict == "NO-ANCHOR"


def test_THE_CORPUS_STILL_CONTAINS_NO_QUIET_ANCHORED_SWEEP(audit):
    """Twelve (sweep, world) pairs; **eight predate the event and every one is
    NO-ANCHOR**; two are W2/W3 on the event day; and the two judgeable pairs —
    LAT on 08-14 and LAT on 08-15 — are **both EVENT**.

    The re-serve was expected to supply the first QUIET reading and did not.
    That is the finding, not a gap: the reference channel has been down since
    08-14 and the round-16 sweep failed its own certification. So this asserts
    the absence, which is the honest state, rather than a QUIET that would have
    to be invented.
    """
    assert len(audit) == 15
    pre = [v for v in audit if v.day < "2026-08-14"]
    assert len(pre) == 8
    assert {v.verdict for v in pre} == {"NO-ANCHOR"}
    assert sum(not v.tested for v in audit) == 10, "ten unjudgeable pairs"
    assert sum(v.tested for v in audit) == 5, "three LAT + W2 + W3, all 08-14/15"
    assert {v.verdict for v in audit if v.tested} == {"EVENT"}
    assert "QUIET" not in {v.verdict for v in audit}

    #: **Two of the five EVENTs point UP.** W2 and W3 on 08-15 rose for a
    #: cohort disjoint from the one falling at LAT, which is the contrast that
    #: killed the provider-degradation account. EVENT means MOVED, and reading
    #: it as "fell" inverted this exact result once.
    up = [v for v in audit if v.tested and v.rates()[0] > v.rates()[1]]
    assert {v.world for v in up} == {"W2", "W3"}
    down = [v for v in audit if v.tested and v.rates()[0] < v.rates()[1]]
    assert {v.world for v in down} == {"LAT"}


def test_the_channel_does_NOT_cover_LAT2(obs):
    """A scoping fact worth asserting now that LAT2 is served.

    `occasion.WORLDS` is the three-world suite, so LAT2 cells never enter the
    paired channel and `emit occasion-health` has no LAT2 row. Round 19 computed
    its LAT-vs-LAT2 contrast directly for that reason. Recorded as a known gap
    rather than discovered later by someone expecting a verdict that cannot
    exist.
    """
    assert "LAT2" not in OQ.WORLDS
    assert not [o for o in obs if o.world == "LAT2"]


def test_a_SAME_DAY_sweep_is_not_a_prior_occasion(obs):
    """Rounds 10, 12 and 13 all served LAT on 2026-08-13. A between-occasion
    sensor that treated them as anchors for each other would be comparing
    within one occasion and calling it across."""
    same_day = [v for v in (OQ.verdict_for("LAT", s, alpha=ALPHA, obs=obs)
                            for s in ("10", "12", "13"))]
    assert {v.day for v in same_day} == {"2026-08-13"}
    assert {v.verdict for v in same_day} == {"NO-ANCHOR"}


def test_FLAGS_REPORT_AND_NEVER_VETO(audit):
    """Per-model flags exist because m=24 detects only large drops; they are
    evidence for a person, not an admission rule. The event-day flags must be
    present, and must not be what produced the sweep verdict."""
    v = next(x for x in audit if (x.world, x.sweep) == ("LAT", "15"))
    flagged = {m for m, _now, _prior, _p in v.flags}
    assert flagged, "the known event flags individual models"
    assert flagged < set(v.returning), "flags are a subset, not the whole cohort"
    for _m, now, prior, p in v.flags:
        assert p < ALPHA and now < prior

    quiet = [x for x in audit if x.verdict == "NO-ANCHOR"]
    assert all(x.flags == () for x in quiet), "no anchor, no per-model test"


def _reserve(obs, rate_k: int, sweep: str = "synthetic",
             day: str = "2026-08-20"):
    """A synthetic LAT re-serve of the fourteen models the event vetoed.

    **The tag was "16" and then round 16 was actually served**, so the fixture
    collided with real cells and the "a clean re-serve must read quiet" case
    started failing against genuine data. A synthetic sweep needs a tag no real
    sweep can take.
    """
    who = {o.model for o in obs if o.world == "LAT" and o.sweep == "15"}
    return obs + [OQ.Observation("LAT", sweep, day, m, rate_k, 24) for m in who]


def test_a_KNOWN_EVENT_IS_EXCLUDED_FROM_LATER_BASELINES(obs):
    """**A detector calibrated on the event it is catching is not a detector.**

    Eight of the fourteen models due for re-serve have exactly one prior LAT A0
    cell and it is the event day. Pool all earlier days and the 0.819 event
    lands INSIDE the reference the re-serve is judged against.

    Both directions invert, which is why this is a test and not a comment:
    a clean re-serve is called an EVENT, and a re-serve that repeats the event
    is called QUIET.
    """
    clean = _reserve(obs, 24)                       # every episode saturates
    good = OQ.audit(alpha=ALPHA, obs=clean)[-1]
    bad = OQ._judge("LAT", "synthetic", alpha=ALPHA, obs=clean, exclude=set())
    assert good.verdict == "QUIET", "a clean re-serve must read quiet"
    assert bad.verdict == "EVENT", "contaminated baseline: false alarm"
    assert good.prior == (125, 128), "baseline is the pre-event day alone"

    repeat = _reserve(obs, 19)                      # ~0.79, the event again
    good2 = OQ.audit(alpha=ALPHA, obs=repeat)[-1]
    bad2 = OQ._judge("LAT", "synthetic", alpha=ALPHA, obs=repeat, exclude=set())
    assert good2.verdict == "EVENT", "a repeat of the event must be caught"
    assert bad2.verdict == "QUIET", "contaminated baseline: MISSES the repeat"


def test_only_the_SIX_clean_anchored_models_carry_the_re_serve(obs):
    """A consequence of the exclusion worth stating rather than discovering.

    Once the event day is out of the baseline, the eight newcomers have no
    clean LAT history at all, so the re-serve's verdict rests on the six that
    do. For the other eight the re-serve ESTABLISHES a baseline rather than
    testing against one — which is exactly what the pre-registered composition
    fork says must happen if they return near their event-day level.
    """
    v = OQ.audit(alpha=ALPHA, obs=_reserve(obs, 24))[-1]
    assert len(v.returning) == 6
    assert len(v.newcomers) == 8


def test_DIAGNOSTIC_BLOCKS_ARE_EXCLUDED_FROM_THE_CHANNEL(obs):
    """The occasion probe was served to measure between-day variation. Pooling
    it into the sensor that detects between-day variation would average away the
    effect it was bought to find — the defect the corpus layer already fixed
    once for the standing read."""
    from seahaven.eden._shared import corpus as C
    seen = {(o.model, o.world, o.sweep) for o in obs}
    for path, cell in C.iter_cells():
        meta = cell.get("meta", {})
        got = C.parse_cell_name(path.name)
        if not got or got["schema"] != "current" or not C.is_diagnostic(meta):
            continue
        key = (meta.get("served_name"), meta.get("eden_level"), got["round"])
        assert key not in seen, f"diagnostic cell reached the channel: {path.name}"


def test_the_channel_reads_GEN3_ONLY(obs):
    """Generations differ in the served prompt and in death semantics. A
    reference channel pooling them would call a generation boundary an
    environment event."""
    from seahaven.eden._shared import corpus as C
    for path, cell in C.iter_cells():
        meta = cell.get("meta", {})
        got = C.parse_cell_name(path.name)
        if (not got or got["schema"] != "current"
                or meta.get("eden_arm") != "A0"
                or meta.get("eden_level") not in OQ.WORLDS):
            continue
        if C.generation_of(meta) == "gen3":
            continue
        key = (meta.get("served_name"), meta.get("eden_level"), got["round"])
        assert key not in {(o.model, o.world, o.sweep) for o in obs}
