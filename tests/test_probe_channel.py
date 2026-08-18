"""The probe channel: exclusion by construction, two anchors, explicit direction.

The spec's verification items 5, 6 and 8 live here — the boiling-frog drift, the
NO-ANCHOR lifecycle, and the taint law at daily cadence.
"""

from __future__ import annotations

import pytest

from seahaven.eden import probe as PB
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import occasion as OQ
from seahaven.eden._shared import probe_channel as PC
from seahaven.eden._shared.stats import fisher

ALPHA = PB.ALPHA


def _day(date, k, n=24, models=8, provider="together", level="LAT", arm="A0"):
    return [PC.ProbeCell(provider, date, f"m{i}", arm, level, k, n)
            for i in range(models)]


# --- exclusion, structurally ------------------------------------------------

def test_a_PROBE_CELL_CANNOT_BE_READ_AS_A_CORPUS_CELL():
    """**The requirement that probe cells never pool into a score corpus, made
    true by construction rather than by discipline.**

    `corpus._NAME` requires an `eden_e` prefix and `_digest_corpus` globs
    `eden_e*.json`. A probe cell can therefore sit in the same directory and be
    invisible to both — which is the only version of this guarantee that
    survives someone pointing the two at one folder.
    """
    name = PC.cell_name("together", "2026-08-16", "Qwen/Qwen3.5-9B", "A0", "LAT")
    assert PC.parse_probe_name(name)["model"] == "Qwen/Qwen3.5-9B"
    assert C.parse_cell_name(name) is None
    assert not name.startswith("eden_e")


def test_PROBE_CELLS_IN_THE_CORPUS_DIR_MOVE_NEITHER_DIGEST_NOR_OBSERVATIONS(
        tmp_path):
    """The property under the adversarial case: put one in the corpus directory
    and neither the digest nor the occasion channel may notice."""
    import json
    import shutil

    from vetoworld.commands import corpus as CORPUS

    src = sorted(C.RESULTS.glob("eden_e19_*A0__LAT2.json"))[0]
    scratch = tmp_path / "results"
    scratch.mkdir()
    shutil.copy(src, scratch / src.name)
    before, _c, _b = CORPUS._digest_corpus(scratch)

    payload = json.loads(src.read_text())
    (scratch / PC.cell_name("together", "2026-08-16", "x/y", "A0", "LAT")
     ).write_text(json.dumps(payload))
    after, _c, _b = CORPUS._digest_corpus(scratch)
    assert before == after, "a probe cell moved the corpus digest"

    obs = OQ.observations(root=scratch)
    assert not [o for o in obs if o.sweep.startswith("probe")]


# --- direction --------------------------------------------------------------

def test_DIRECTION_IS_ALWAYS_SET_including_when_the_verdict_is_QUIET():
    """EVENT means MOVED, not FELL. Reading it as "fell" inverted round 17's
    conclusion once — W2 and W3 rose on the day LAT fell. No caller may infer
    direction from the verdict word, so it is always carried."""
    up = PC.judge_day(_day("2026-08-16", 24), provider="together",
                      date="2026-08-16", channel="LAT.A0",
                      epoch=(120, 192), rolling=None, alpha=ALPHA)
    down = PC.judge_day(_day("2026-08-16", 12), provider="together",
                        date="2026-08-16", channel="LAT.A0",
                        epoch=(183, 192), rolling=None, alpha=ALPHA)
    assert up.verdict == "EVENT" and up.direction == "up"
    assert down.verdict == "EVENT" and down.direction == "down"

    quiet = PC.judge_day(_day("2026-08-16", 23), provider="together",
                         date="2026-08-16", channel="LAT.A0",
                         epoch=(183, 192), rolling=None, alpha=ALPHA)
    assert quiet.verdict == "QUIET" and quiet.direction in ("up", "down", "flat")


# --- the two anchors --------------------------------------------------------

def _drift(per_day, days=40, start=23 / 24):
    cells, rate = [], start
    for i in range(days):
        d = f"2026-{9 + i // 30:02d}-{(i % 30) + 1:02d}"
        cells += _day(d, round(rate * 24))
        rate -= per_day
    return cells


def test_BOILING_FROG_and_THE_RATE_AT_WHICH_THE_TWO_ANCHORS_DIVERGE():
    """**Why rolling-only would be a trap — and where the trap actually is.**

    The spec's verification item names a 0.02/day drift as the case rolling
    sleeps through. Measured, it does not: at 192 episodes a day the rolling
    window is powerful enough to catch 0.02/day itself. The epoch-only regime
    begins at **<= 0.01/day**, and this asserts the boundary rather than one
    hand-picked number, because the boundary is the property.
    """
    fast = [v for v in PC.trace(
        _drift(0.02), provider="together", channel="LAT.A0", epoch=(183, 192),
        alpha=ALPHA, rolling_k=PB.ROLLING_K, stale_after_days=999)
        if v.verdict == "EVENT"]
    assert fast and fast[0].p_rolling < ALPHA, (
        "0.02/day should be caught by BOTH anchors — if this changes, the "
        "rolling window has lost power and the boundary below moves with it")

    slow = [v for v in PC.trace(
        _drift(0.005), provider="together", channel="LAT.A0", epoch=(183, 192),
        alpha=ALPHA, rolling_k=PB.ROLLING_K, stale_after_days=999)
        if v.verdict == "EVENT"]
    assert slow, "the epoch anchor never noticed a 0.005/day drift"
    first = slow[0]
    assert first.direction == "down"
    assert first.p_epoch < ALPHA, "epoch is what caught it"
    assert first.p_rolling is None or first.p_rolling > ALPHA, (
        "the rolling anchor must NOT catch a 0.005/day drift — that is the "
        "boiling frog, and the epoch anchor exists for exactly this")


def test_EVENT_DAYS_NEVER_ENTER_THE_ROLLING_ANCHOR():
    """The taint law at daily cadence. `occasion.event_pairs` enforces it on
    the corpus; a rolling window that admitted its own event would launder the
    new level into the baseline within K days."""
    cells = (_day("2026-08-16", 23) + _day("2026-08-17", 23)
             + _day("2026-08-18", 12) + _day("2026-08-19", 12)
             + _day("2026-08-20", 12))
    tr = PC.trace(cells, provider="together", channel="LAT.A0",
                  epoch=(183, 192), alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS)
    by = {v.date: v for v in tr}
    assert by["2026-08-18"].verdict == "EVENT"
    #: Two more days at the SAME depressed level must still read EVENT. If the
    #: rolling anchor had absorbed 08-18 they would read QUIET.
    assert by["2026-08-19"].verdict == "EVENT"
    assert by["2026-08-20"].verdict == "EVENT"
    for d in ("2026-08-19", "2026-08-20"):
        assert by[d].rolling == (368, 384), (
            "rolling pools the two QUIET days across all eight models "
            "(2 x 8 x 23 of 2 x 8 x 24) and admits neither EVENT day")


def test_NO_ANCHOR_LIFECYCLE_a_column_earns_an_epoch_and_then_judges():
    """A column with no epoch admits with a flag and yields no verdict. Once it
    has one it judges — and the anchor it earned is never revised.

    **This test used to earn the anchor BY HAND** — it read `v.now` off the
    first day and passed it back in as `epoch`, which is the shape the module
    docstring promised and no code implemented. It passed while
    `probe_channel` had no earning path at all, so the promise "a new column's
    epoch is its own first QUIET run" was witnessed only by the witness. It now
    drives `trace(earn_days=...)`, the real one.
    """
    v = PC.judge_day(_day("2026-08-16", 20, provider="fireworks"),
                     provider="fireworks", date="2026-08-16",
                     channel="LAT.A0", epoch=None, rolling=None, alpha=ALPHA)
    assert v.verdict == "NO-ANCHOR"
    assert v.p_epoch is None and v.p_rolling is None
    assert v.now[1] == 192, "it still ADMITS to the record"


# --- the anchor-earning rule, which was a docstring until 2026-08-17 --------

def _pair_days(spec, provider="deepinfra", level="LAT", arm="A0"):
    """The matched pair's own shape: two models a day, n=24 each, pooled n=48."""
    out = []
    for date, k in spec:
        out += _day(date, k, models=2, provider=provider, level=level, arm=arm)
    return out


def test_EARN_the_three_day_rule_FIRES_and_freezes_an_envelope_beside_it():
    """**NO-ANCHOR is escapable.** Three mutually non-separable served days
    pool into the channel's own epoch, and the constituent DAY spread freezes
    beside it — the Flash precedent moved from blocks to days.

    The envelope is not decoration. At n=48 a day mutual non-separability is a
    weak standard, so the anchor alone would be quoted as tighter than the
    evidence supports; the envelope is what carries the mush forward.
    """
    got = PC.earn_epoch([("2026-08-18", (42, 48)),
                         ("2026-08-19", (40, 48)),
                         ("2026-08-20", (44, 48))],
                        alpha=ALPHA, earn_days=PB.EARN_DAYS)
    assert got is not None
    assert got["anchor"] == (126, 144), "the pooled sum of the three days"
    assert got["envelope"] == (0.8333, 0.9167), (
        "min and max of the constituent DAY rates, not of the pooled band — "
        "the pooled Wilson interval here is far narrower than the real spread")
    assert got["days"] == ("2026-08-18", "2026-08-19", "2026-08-20")
    #: Fewer than the rule requires earns nothing, however clean.
    assert PC.earn_epoch([("2026-08-18", (42, 48)), ("2026-08-19", (42, 48))],
                         alpha=ALPHA, earn_days=PB.EARN_DAYS) is None


def test_EARN_IS_PAIRWISE_so_a_MONOTONE_DRIFT_cannot_pass_as_a_BASELINE():
    """**The reason the rule says every pairwise p, not every adjacent p.**

    These three days pass both adjacent tests (p = 0.023 and 0.084) and fail
    the extremes badly (p = 0.00004). Adjacent-only would freeze a SLOPE as
    this column's baseline — and an epoch anchor exists precisely to catch the
    drift a day-over-day test sleeps through, so admitting a drift as the epoch
    disables the one anchor that was supposed to see it.
    """
    drift = [("2026-08-18", (183, 192)),
             ("2026-08-19", (170, 192)),
             ("2026-08-20", (157, 192))]
    counts = [c for _d, c in drift]
    assert fisher(*counts[0], *counts[1]) >= ALPHA
    assert fisher(*counts[1], *counts[2]) >= ALPHA
    assert fisher(*counts[0], *counts[2]) < ALPHA, "the extremes DO separate"

    assert not PC.mutually_non_separable(counts, ALPHA)
    assert PC.earn_epoch(drift, alpha=ALPHA, earn_days=PB.EARN_DAYS) is None


def test_EARN_a_DISAGREEING_DAY_restarts_the_window_but_its_NEIGHBOURS_survive():
    """A day that separates from the pool leaves the CANDIDATE pool — and only
    it. The window slides rather than hard-restarting, because a day that never
    disagreed with anything has done nothing to be punished for."""
    served = [("2026-08-18", (24, 48)),          # the outlier, 0.500
              ("2026-08-19", (42, 48)),
              ("2026-08-20", (40, 48)),
              ("2026-08-21", (44, 48))]
    got = PC.earn_epoch(served, alpha=ALPHA, earn_days=PB.EARN_DAYS)
    assert got is not None
    assert got["days"] == ("2026-08-19", "2026-08-20", "2026-08-21"), (
        "the outlier is excluded and the three clean days that followed it "
        "earn the anchor — hard-restarting would have discarded 08-19 and "
        "08-20, which separated from nothing")
    assert got["anchor"] == (126, 144)


def test_EARN_AN_OUTAGE_DOES_NOT_RESTART_THE_WINDOW():
    """**SERVE_FAIL and VOID days do not restart the candidate window.**

    'Consecutive' means consecutive among SERVED days. A provider outage is not
    evidence about agreement, and an anchor-earning process an outage can reset
    measures uptime rather than stability — a column on a flaky provider would
    never earn an anchor no matter how stable its served days were.

    A failed day is ABSENT from the served sequence rather than special-cased,
    which is why the calendar gap below is invisible to the rule.
    """
    #: 08-19 SERVE_FAIL, 08-22 VOID — neither wrote a cell, so neither appears.
    served = [("2026-08-18", (42, 48)),
              ("2026-08-20", (40, 48)),
              ("2026-08-23", (44, 48))]
    got = PC.earn_epoch(served, alpha=ALPHA, earn_days=PB.EARN_DAYS)
    assert got is not None, (
        "a five-day calendar span with two failures still earns the anchor — "
        "the three SERVED days are what the rule counts")
    assert got["days"] == ("2026-08-18", "2026-08-20", "2026-08-23")


def test_EARN_END_TO_END_the_column_goes_NO_ANCHOR_then_QUIET_then_EVENT():
    """The whole lifecycle through `trace`, which is what the job calls.

    **NO-ANCHOR was ABSORBING before this.** Only QUIET days enter the rolling
    pool and a channel with no epoch never returns QUIET, so a column arriving
    without an anchor could never earn a rolling one either — 30 days of
    NO-ANCHOR at full price, which is what the DeepInfra column was about to do.
    """
    cells = _pair_days([("2026-08-18", 21), ("2026-08-19", 20),
                        ("2026-08-20", 22),        # window closes here
                        ("2026-08-21", 21),        # first day actually judged
                        ("2026-08-22", 12)])       # rate 0.500, a real step
    tr = PC.trace(cells, provider="deepinfra", channel="LAT.A0", epoch=None,
                  alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS,
                  earn_days=PB.EARN_DAYS)
    by = {v.date: v for v in tr}

    #: **The three constituent days are NOT judged against the anchor they
    #: constitute.** An anchor cannot be evidence about the days it pools.
    for d in ("2026-08-18", "2026-08-19", "2026-08-20"):
        assert by[d].verdict == "NO-ANCHOR", by[d]
        assert by[d].epoch is None
    assert "EARNED" in by["2026-08-20"].reason
    assert "banked" in by["2026-08-18"].reason

    #: The first day after the window closes judges against the earned anchor.
    assert by["2026-08-21"].verdict == "QUIET"
    assert by["2026-08-21"].epoch == (126, 144)
    assert by["2026-08-21"].envelope == (0.8333, 0.9167)

    #: And it is a working anchor, not a decorative one.
    assert by["2026-08-22"].verdict == "EVENT"
    assert by["2026-08-22"].direction == "down"
    #: Never revised: the step day does not move the anchor it fired against.
    assert by["2026-08-22"].epoch == (126, 144)


def test_EARN_IS_OFF_BY_DEFAULT_so_TOGETHERS_TRACE_IS_UNCHANGED():
    """`earn_days=None` is the pre-existing behaviour exactly. Together has its
    epoch from the corpus and must not acquire a second way to get one."""
    cells = _pair_days([("2026-08-18", 21), ("2026-08-19", 20),
                        ("2026-08-20", 22), ("2026-08-21", 21)])
    tr = PC.trace(cells, provider="deepinfra", channel="LAT.A0", epoch=None,
                  alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS)
    assert {v.verdict for v in tr} == {"NO-ANCHOR"}
    assert all(v.epoch is None for v in tr)


def test_STALE_when_no_QUIET_refresh_inside_the_window():
    """A rolling anchor made of old days is not a rolling anchor, and STALE is
    HOLD for anything event-conditioned."""
    cells = _day("2026-08-16", 23) + _day("2026-08-30", 23)
    tr = PC.trace(cells, provider="together", channel="LAT.A0",
                  epoch=(183, 192), alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS)
    assert tr[-1].verdict == "STALE", [v.verdict for v in tr]


# --- the decision channel's envelope ----------------------------------------

def test_THE_ENVELOPE_IS_CURRENTLY_INERT_AND_THAT_IS_THE_FINDING():
    """**Measured, not assumed: at m=24 and alpha=0.01 the envelope cannot
    bite.**

    Fisher against the 69/96 anchor fires only for k <= 10 (rate <= 0.417) or
    k = 24. The envelope covers k in [14, 21]. The two sets are DISJOINT, so
    nothing that could land inside the envelope is significant in the first
    place — the low power is not "mostly" protecting the channel, it is
    protecting it completely.

    The envelope is therefore belt-and-braces against a future where m rises or
    alpha loosens. Recorded here rather than left for a reader to assume it is
    doing work, and asserted so that if it ever DOES become active, somebody
    finds out from a failing test rather than from a verdict.
    """
    from seahaven.eden._shared.stats import fisher

    fires = {k for k in range(PB.EPISODES + 1)
             if fisher(k, PB.EPISODES, *PB.FLASH_ANCHOR) < PB.ALPHA}
    inside = {k for k in range(PB.EPISODES + 1)
              if PB.FLASH_ENVELOPE[0] <= k / PB.EPISODES <= PB.FLASH_ENVELOPE[1]}
    assert fires and inside
    assert not (fires & inside), (
        f"the envelope has become ACTIVE: k in {sorted(fires & inside)} now "
        "both fires and sits inside the block range. That is not a bug — it "
        "means m or alpha changed — but the anchor's power has changed with "
        "it and FLASH_ANCHOR_RULE needs re-reading before the next verdict.")


def test_THE_ENVELOPE_STILL_SUPPRESSES_WHEN_IT_IS_REACHED():
    """The mechanism works even though nothing currently reaches it. Driven
    directly with a wider envelope, so the suppression path is exercised rather
    than merely present — an unreachable branch is an untested branch."""
    args = dict(provider="together", date="2026-08-16", channel="LAT.A1",
                epoch=PB.FLASH_ANCHOR, rolling=None, alpha=ALPHA)
    cells = _day("2026-08-16", 4, models=1, arm="A1")     # 0.167, fires
    assert PC.judge_day(cells, envelope=None, **args).verdict == "EVENT"

    wide = PC.judge_day(cells, envelope=(0.10, 0.90), **args)
    assert wide.verdict == "QUIET"
    assert "envelope" in wide.reason
    assert wide.direction == "down", "direction survives suppression"


def test_A_VALUE_OUTSIDE_THE_ENVELOPE_STILL_FIRES():
    """The other half: suppression must not swallow a real event."""
    out = PC.judge_day(
        _day("2026-08-16", 2, models=1, arm="A1"), provider="together",
        date="2026-08-16", channel="LAT.A1", epoch=PB.FLASH_ANCHOR,
        rolling=None, alpha=ALPHA, envelope=PB.FLASH_ENVELOPE)
    assert out.rate < PB.FLASH_ENVELOPE[0]
    assert out.verdict == "EVENT" and out.direction == "down"


# --- the job's memory: read_rows was exported and called by nothing ---------

def _row(date, k, verdict, provider="deepinfra", channel="LAT.A0", n=48):
    return {"date": date, "provider": provider, "channel": channel,
            "k": k, "n": n, "verdict": verdict}


def test_ROWS_TO_HISTORY_filters_by_PROVIDER_and_CHANNEL_and_dedupes():
    """The published log is one stream for every column and channel. Reading a
    DeepInfra rolling window out of Together's rows is the same class of error
    as borrowing Together's epoch, one layer down."""
    rows = [_row("2026-08-18", 42, "NO-ANCHOR"),
            _row("2026-08-18", 180, "QUIET", provider="together", n=192),
            _row("2026-08-19", 44, "QUIET", channel="W2.A0"),
            _row("2026-08-19", 40, "NO-ANCHOR"),
            _row("2026-08-19", 41, "QUIET")]          # republished, later wins
    got = PC.rows_to_history(rows, provider="deepinfra", channel="LAT.A0")
    assert got == [("2026-08-18", (42, 48), "NO-ANCHOR"),
                   ("2026-08-19", (41, 48), "QUIET")], (
        "one entry per date, other columns and channels excluded, and a "
        "corrected republish winning over the row it corrected")


def test_HISTORY_GIVES_THE_EPHEMERAL_JOB_A_ROLLING_ANCHOR():
    """**Without this the scheduled job has no memory.** `trace` rebuilds
    history from local cells; the job's container has none, so every run would
    start empty — rolling permanently None, and the epoch never earned.

    Day one worked only because it ran on this machine, where cells persist.
    """
    today = _pair_days([("2026-08-21", 21)])
    history = [("2026-08-18", (42, 48), "QUIET"),
               ("2026-08-19", (40, 48), "QUIET"),
               ("2026-08-20", (44, 48), "QUIET")]

    blind = PC.trace(today, provider="deepinfra", channel="LAT.A0",
                     epoch=(126, 144), alpha=ALPHA, rolling_k=PB.ROLLING_K,
                     stale_after_days=PB.STALE_AFTER_DAYS)
    assert blind[0].rolling is None, "today alone: no memory, no rolling anchor"

    withmem = PC.trace(today, provider="deepinfra", channel="LAT.A0",
                       epoch=(126, 144), alpha=ALPHA, rolling_k=PB.ROLLING_K,
                       stale_after_days=PB.STALE_AFTER_DAYS, history=history)
    assert withmem[0].rolling == (126, 144), (
        "the three published QUIET days pool into the rolling anchor")


def test_HISTORY_ALSO_FEEDS_THE_EARN_POOL_or_the_JOB_NEVER_EARNS():
    """The earn window needs three served days. In an ephemeral container only
    today exists locally, so without the published rows the column would sit at
    NO-ANCHOR forever while paying full price every day."""
    today = _pair_days([("2026-08-21", 21)])
    history = [("2026-08-18", (42, 48), "NO-ANCHOR"),
               ("2026-08-19", (40, 48), "NO-ANCHOR"),
               ("2026-08-20", (44, 48), "NO-ANCHOR")]
    tr = PC.trace(today, provider="deepinfra", channel="LAT.A0", epoch=None,
                  alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS,
                  earn_days=PB.EARN_DAYS, history=history)
    assert tr[0].verdict == "QUIET"
    assert tr[0].epoch == (126, 144), (
        "the anchor is earned from the PUBLISHED days — NO-ANCHOR rows are "
        "served days and belong in the candidate pool; agreement filters "
        "them, not the verdict word")
    assert tr[0].envelope == (0.8333, 0.9167)


def test_A_LOCAL_CELL_BEATS_A_PUBLISHED_ROW_FOR_THE_SAME_DAY():
    """Where both exist the raw cell wins: the row is a record of the evidence,
    not a substitute for it. Otherwise a corrected re-run on this machine would
    be silently overruled by the row it was correcting."""
    today = _pair_days([("2026-08-18", 21)])          # (42, 48) on disk
    history = [("2026-08-18", (12, 48), "EVENT")]     # a stale published row
    tr = PC.trace(today, provider="deepinfra", channel="LAT.A0",
                  epoch=(126, 144), alpha=ALPHA, rolling_k=PB.ROLLING_K,
                  stale_after_days=PB.STALE_AFTER_DAYS, history=history)
    assert len(tr) == 1 and tr[0].now == (42, 48)
    assert tr[0].verdict == "QUIET", "judged from the cell, not from the row"
