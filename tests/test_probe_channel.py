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
    has one it judges — and the anchor it earned is never revised."""
    v = PC.judge_day(_day("2026-08-16", 20, provider="fireworks"),
                     provider="fireworks", date="2026-08-16",
                     channel="LAT.A0", epoch=None, rolling=None, alpha=ALPHA)
    assert v.verdict == "NO-ANCHOR"
    assert v.p_epoch is None and v.p_rolling is None
    assert v.now[1] == 192, "it still ADMITS to the record"

    earned = v.now
    later = PC.judge_day(_day("2026-08-20", 20, provider="fireworks"),
                         provider="fireworks", date="2026-08-20",
                         channel="LAT.A0", epoch=earned, rolling=None,
                         alpha=ALPHA)
    assert later.verdict == "QUIET" and later.epoch == earned


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
