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


def test_the_ONCE_CERTIFIED_EVENT_IS_RETRACTED(audit):
    """**The certified event is withdrawn. Issue #113.**

    It read: LAT on 2026-08-14, six returning models, 0.977 -> 0.819, intervals
    separating, p=1.7e-05. Every "returning model" was a genuine model on the
    earlier day and cogito on the later one — six different models before, one
    model after. It was never a paired comparison.

    Corrected, the sweep has ONE returning model and reads QUIET at p=0.222.
    The instrument did not fail; it judged the identities it was given.
    """
    v = next(x for x in audit if (x.world, x.sweep) == ("LAT", "15"))
    assert v.verdict == "QUIET"
    assert v.returning == ("deepcogito/cogito-v2-1-671b",)
    now, prior = v.rates()
    assert prior == pytest.approx(0.875, abs=0.001)
    assert now == pytest.approx(0.756, abs=0.001)
    assert v.p == pytest.approx(0.222, abs=0.005)

    lo_now, hi_now = OQ.interval(*v.now)
    lo_pri, hi_pri = OQ.interval(*v.prior)
    assert hi_now > lo_pri, "the intervals now overlap; nothing separates"


def test_W2_and_W3_on_the_event_day_HAVE_an_anchor_and_read_QUIET(audit):
    """**A correction on top of a correction, and the second one is subtler.**

    These were first published as "held steady" — the co-located-detector
    argument. That was withdrawn to NO-ANCHOR on the finding that both had ZERO
    returning models: fourteen new names each, scored against a historical
    cohort of different models.

    Corrected, the fourteen names were one model that HAD been served at both
    worlds before. So there is an anchor, and both read QUIET — arriving back at
    the original word by a completely different route, and meaning something
    else by it. QUIET on one model is not the co-located-detector argument; it
    cannot be, because one detector is not two.
    """
    for world in ("W2", "W3"):
        v = next(x for x in audit if (x.world, x.sweep) == (world, "15"))
        assert v.verdict == "QUIET"
        assert v.returning == ("deepcogito/cogito-v2-1-671b",)
        assert v.newcomers == ()
        assert v.p == pytest.approx(1.0)


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

    #: Pairing still disagrees with pooling, which is the point of the test.
    #: It now says QUIET rather than NO-ANCHOR — one model, matched to itself,
    #: unmoved — where pooling says a significant rise. The pooled reading is
    #: an artefact of cohort composition either way.
    paired = OQ.verdict_for("W2", "15", alpha=ALPHA, obs=obs)
    assert paired.verdict == "QUIET"
    assert pooled_p < ALPHA < paired.p, (
        "pooling manufactures significance that pairing does not see")


def test_THE_CORPUS_CONTAINS_NO_SURVIVING_EVENT(audit):
    """**The corrected catalogue, asserted. Issue #113.**

    The channel used to report five EVENTs and no QUIET. Corrected it reports
    seven QUIET and one EVENT, and that one does not survive inspection either:
    it is LAT e18, whose eight cells are one model on one 24-seed set, so its
    n=192 counts the same seeds eight times. Judged per replicate, none of the
    eight fires. It is also QUIET under the Bonferroni pass alpha the
    retrospective read uses (0.05/7 = 0.00714 > 0.0336 is false).

    So: **no LAT event, on any reading.** The seismograph's confirmed catalogue
    is the pre-defect Flash step and nothing else.
    """
    #: **Counts are derived, the CLAIM is asserted.** This hardcoded
    #: `len(audit) == 15` and broke the moment round 20 added three rows — a
    #: test failing because the programme made progress teaches nothing. What
    #: must hold is the claim in the name: no surviving event.
    pre = [v for v in audit if v.day < "2026-08-14"]
    assert len(pre) == 8
    assert {v.verdict for v in pre} == {"NO-ANCHOR"}
    assert sum(v.tested for v in audit) + sum(not v.tested for v in audit) \
        == len(audit)

    events = [v for v in audit if v.verdict == "EVENT"]
    assert [(v.world, v.sweep) for v in events] == [("LAT", "18")]
    assert events[0].p == pytest.approx(0.0336, abs=0.001)
    bonferroni = ALPHA / sum(v.tested for v in audit)
    assert events[0].p > bonferroni, (
        "the sole EVENT does not clear the retrospective pass alpha "
        f"({bonferroni:.5f}); at the Bonferroni standard the corpus has none")

    #: Every returning cohort is now a single model, which is the whole of the
    #: retraction in one line: there is no multi-model comparison left anywhere.
    assert {len(v.returning) for v in audit if v.tested} == {1}

    #: **A first measurement reads NO-ANCHOR, and that is the design.** Round
    #: 20 measured GLM-5.2 at three worlds where it had no gen-3 history, so
    #: all three rows admit with a flag rather than being vetoed — the reductio
    #: in `_shared.occasion`: vetoing NO-ANCHOR would make the first
    #: observation of everything permanently inadmissible.
    new = [v for v in audit if v.sweep == "20"]
    assert len(new) == 3
    assert {v.verdict for v in new} == {"NO-ANCHOR"}
    assert all(v.p is None and v.returning == () for v in new)


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


def test_FLAGS_REPORT_AND_NEVER_VETO(obs):
    """Per-model flags exist because m=24 detects only large drops; they are
    evidence for a person, not an admission rule.

    **Driven synthetically rather than off the corpus.** It used to read the
    real event's flags — and the real event was retracted, which is precisely
    the failure mode `event_pairs` exists to prevent one layer down. A property
    of the machinery should not depend on a finding surviving.
    """
    hist = [OQ.Observation("LAT", "h", "2026-01-01", m, 24, 24)
            for m in ("a/one", "a/two", "a/three")]
    now = [OQ.Observation("LAT", "s", "2026-01-02", "a/one", 6, 24),
           OQ.Observation("LAT", "s", "2026-01-02", "a/two", 24, 24),
           OQ.Observation("LAT", "s", "2026-01-02", "a/three", 24, 24)]
    v = OQ.verdict_for("LAT", "s", alpha=ALPHA, obs=hist + now)

    flagged = {m for m, _now, _prior, _p in v.flags}
    assert flagged == {"a/one"}, "only the model that actually moved"
    assert flagged < set(v.returning), "flags are a subset, not the whole cohort"
    for _m, rate_now, rate_prior, p in v.flags:
        assert p < ALPHA and rate_now < rate_prior

    no_anchor = [x for x in OQ.audit(alpha=ALPHA, obs=obs) if not x.tested]
    assert no_anchor and all(x.flags == () for x in no_anchor), (
        "no anchor, no per-model test")


#: **A closed synthetic world, owing nothing to the corpus.**
#:
#: This fixture used to be built from the real event's cohort. When that event
#: was retracted (#113) the fixture went with it — a machinery test that depends
#: on a finding is a machinery test that dies when the finding does, which is
#: the same mistake as calibrating a detector on the event it is catching, one
#: level up. Six models, three days, and every number stated here.
_MODELS = tuple(f"synth/m{i}" for i in range(6))


def _world(reserve_k: int):
    """The structure that made the taint law necessary, in miniature.

    **Only ONE model has a clean prior day.** The other five appear first on the
    event day, so a baseline built from "all earlier days" is mostly the event
    itself — which is exactly the real shape that prompted this: eight of the
    fourteen models due for re-serve had one prior LAT cell and it was the event.

    s0  m0 alone, saturated        24/24
    s1  all six, dropped           12/24 each
    s2  all six, re-served         `reserve_k`/24 each
    """
    out = [OQ.Observation("LAT", "s0", "2026-01-01", _MODELS[0], 24, 24)]
    for m in _MODELS:
        out.append(OQ.Observation("LAT", "s1", "2026-01-02", m, 12, 24))
        out.append(OQ.Observation("LAT", "s2", "2026-01-03", m, reserve_k, 24))
    return out


def test_a_KNOWN_EVENT_IS_EXCLUDED_FROM_LATER_BASELINES():
    """**A detector calibrated on the event it is catching is not a detector.**

    `s1` is a real drop. If it stays in the baseline that `s2` is judged
    against, both directions invert: a clean re-serve reads EVENT, and a
    re-serve that repeats the drop reads QUIET.
    """
    clean = _world(24)                     # a full recovery
    good = OQ.verdict_for("LAT", "s2", alpha=ALPHA, obs=clean)
    bad = OQ._judge("LAT", "s2", alpha=ALPHA, obs=clean, exclude=set())
    assert good.verdict == "QUIET", "a clean re-serve must read quiet"
    assert bad.verdict == "EVENT", "contaminated baseline: false alarm"
    assert good.prior == (24, 24), "baseline is the pre-event day alone"

    repeat = _world(12)                    # the drop again
    good2 = OQ.verdict_for("LAT", "s2", alpha=ALPHA, obs=repeat)
    bad2 = OQ._judge("LAT", "s2", alpha=ALPHA, obs=repeat, exclude=set())
    assert good2.verdict == "EVENT", "a repeat of the event must be caught"
    assert bad2.verdict == "QUIET", "contaminated baseline: MISSES the repeat"


def test_TAINT_PROPAGATES_OLDEST_FIRST_so_the_decision_is_well_founded():
    """The ordering property the taint law rests on: `s1` must be judged before
    `s2` needs to know whether to exclude it. Asserted directly, because a
    detector that resolved this in the wrong order would be circular."""
    tainted = OQ.event_pairs(alpha=ALPHA, obs=_world(24))
    assert ("LAT", "s1") in tainted
    assert ("LAT", "s0") not in tainted, "the first day has nothing to move from"
    assert ("LAT", "s2") not in tainted, "a clean re-serve is not itself an event"


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
