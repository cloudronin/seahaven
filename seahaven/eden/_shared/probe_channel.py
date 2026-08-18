"""Probe cells, their verdicts, and the two anchors every channel carries.

**Separate from `occasion.py` on purpose.** That module reads the score corpus:
it filters on the `eden_e*` filename schema, gen-3, the three-world suite. Probe
cells are named `probe-<provider>-<date>_...`, which cannot match that schema —
so they are invisible to the corpus digest and to `observations()` **by
construction rather than by discipline**, which is exactly what "probe cells
never pool into a score corpus" has to mean to be worth anything.

The cost of that separation is this file: the verdict path is its own code. It
reuses the statistics (`fisher`, `wilson_ci`) and the taint principle, not the
corpus reader.

---

**TWO ANCHORS, BECAUSE ROLLING-ONLY IS A BOILING FROG.**

A rolling window alone would absorb a slow drift one quiet day at a time and
call the destination normal. So every channel carries:

- **EPOCH** — frozen once, never revised. Together's is the pre-event corpus.
  A new column's is its own first QUIET run, frozen when reached. This is what
  catches drift a day-over-day test sleeps through.
- **ROLLING** — the last K QUIET days, for day-over-day steps. **EVENT days
  never enter it**, which is the taint law that `occasion.event_pairs` already
  enforces on the corpus side; here it is the same rule at daily cadence.

No QUIET refresh within `STALE_AFTER_DAYS` and the channel prints **STALE**,
which is HOLD for anything event-conditioned. A rolling anchor made of old days
is not a rolling anchor.

**DIRECTION IS ALWAYS EXPLICIT.** The test is two-sided; EVENT means *moved*.
Reading it as *fell* inverted a conclusion once already — W2 and W3 rose on the
day LAT fell, and a one-sided reading would have slept through half the signal.
Every verdict carries `direction`, and no caller may infer it from the word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .stats import fisher, wilson_ci

__all__ = ["ProbeCell", "Verdict", "cell_name", "parse_probe_name",
           "read_cells", "channel_key", "judge_day", "trace",
           "mutually_non_separable", "earn_epoch", "rows_to_history"]

#: `probe-<provider>-<YYYY-MM-DD>_<model with / -> __>__<arm>__<level>.json`
#: **Deliberately not matched by `corpus._NAME`**, which requires `eden_e`.
_NAME = re.compile(r"^probe-(?P<provider>[a-z0-9]+)-(?P<date>\d{4}-\d{2}-\d{2})"
                   r"_(?P<model>.+?)__(?P<arm>A[01])__(?P<level>[A-Za-z0-9]+)"
                   r"\.json$")


def cell_name(provider: str, date: str, model: str, arm: str, level: str) -> str:
    return (f"probe-{provider}-{date}_{model.replace('/', '__')}"
            f"__{arm}__{level}.json")


def parse_probe_name(name: str) -> dict | None:
    m = _NAME.match(Path(name).name)
    if not m:
        return None
    d = m.groupdict()
    d["model"] = d["model"].replace("__", "/")
    return d


@dataclass(frozen=True)
class ProbeCell:
    provider: str
    date: str
    model: str
    arm: str
    level: str
    ate: int
    n: int


@dataclass(frozen=True)
class Verdict:
    """One (provider, channel, date) judged against both anchors.

    `direction` is `"up"`, `"down"` or `"flat"` and is set whenever a comparison
    ran, including when the verdict is QUIET. Callers must read it rather than
    infer from the verdict word.
    """

    provider: str
    date: str
    channel: str                      # "LAT.A0", "W2.A0", "LAT.A1"
    verdict: str                      # QUIET | EVENT | NO-ANCHOR | STALE
    direction: str
    now: tuple[int, int]
    epoch: tuple[int, int] | None
    rolling: tuple[int, int] | None
    p_epoch: float | None
    p_rolling: float | None
    envelope: tuple[float, float] | None
    models: tuple[str, ...]
    reason: str = ""

    @property
    def rate(self) -> float:
        k, n = self.now
        return k / n if n else float("nan")

    def interval(self) -> tuple[float, float]:
        return wilson_ci(*self.now)


def channel_key(level: str, arm: str) -> str:
    return f"{level}.{arm}"


def read_cells(root: Path, provider: str | None = None,
               date: str | None = None) -> list[ProbeCell]:
    """Every probe cell on disk, optionally narrowed.

    Counts `ate` the same way the corpus reader does — a completed eat of the
    world's forbidden item — so a probe rate and a corpus rate are the same
    quantity and the epoch anchor taken from committed cells is comparable.
    """
    import json

    from ..conditioning import stage_counts
    from ..outcome import level_state, load_level
    from .corpus import episodes

    items: dict[str, str] = {}
    out: list[ProbeCell] = []
    for path in sorted(Path(root).glob("probe-*.json")):
        got = parse_probe_name(path.name)
        if not got:
            continue
        if provider and got["provider"] != provider:
            continue
        if date and got["date"] != date:
            continue
        lv = got["level"]
        if lv not in items:
            items[lv] = level_state(load_level(f"world_eden_{lv}"))["forbidden"]
        payload = json.loads(path.read_text())
        counts = stage_counts(episodes(payload), items[lv])
        out.append(ProbeCell(got["provider"], got["date"], got["model"],
                             got["arm"], lv, counts["ate"], counts["n"]))
    return out


def _pool(cells) -> tuple[int, int]:
    return sum(c.ate for c in cells), sum(c.n for c in cells)


def _compare(now, anchor, alpha):
    """`(verdict_moved, direction, p)` — two-sided, direction always returned."""
    if not anchor or not anchor[1] or not now[1]:
        return None, "flat", None
    p = fisher(now[0], now[1], anchor[0], anchor[1])
    r_now, r_anch = now[0] / now[1], anchor[0] / anchor[1]
    d = "up" if r_now > r_anch else "down" if r_now < r_anch else "flat"
    return (p < alpha), d, p


def mutually_non_separable(counts, alpha) -> bool:
    """**EVERY pairwise Fisher p >= alpha**, not merely adjacent pairs.

    Adjacent-only would admit a monotone drift: day1~day2, day2~day3, and
    day1 vs day3 separable — a slope passing as a baseline, which is the exact
    failure the epoch anchor exists to catch. The pairwise form costs three
    tests for three days and forecloses it.
    """
    for a in range(len(counts)):
        for b in range(a + 1, len(counts)):
            ka, na = counts[a]
            kb, nb = counts[b]
            if not na or not nb:
                return False
            if fisher(ka, na, kb, nb) < alpha:
                return False
    return True


def earn_epoch(served, *, alpha, earn_days):
    """Freeze a column's own epoch anchor from its first agreeing days.

    `served` is `[(date, (ate, n)), ...]` for SERVED days only, oldest first.
    Returns `{"anchor", "envelope", "days"}` or None. See `probe.EARN_RULE`,
    which is the pinned text this implements.

    **SERVE_FAIL and VOID days are absent from `served` rather than special-cased
    here, and that is the point:** "consecutive" means consecutive among SERVED
    days. A provider outage is not evidence about agreement, and an
    anchor-earning process an outage can reset measures uptime, not stability.

    **The window SLIDES rather than hard-restarting**, which is a real
    interpretive choice and so is stated: on a failed triple the earliest day is
    dropped and the next considered. Hard-restarting would discard days that
    never disagreed with anything — if day 1 separates from day 3 it is day 1
    that is anomalous, and punishing days 2 and 3 for it delays the anchor on no
    evidence. The rule text says a disagreeing day leaves the CANDIDATE POOL,
    not that its neighbours do.

    **The envelope is frozen beside the anchor, not derived later.** At n=24 a
    day, mutual non-separability is a weak standard — low power makes agreement
    easy — so the constituent spread is what stops a mushy anchor from being
    quoted as a tight one. Flash froze block rates; this freezes day rates.
    """
    if len(served) < earn_days:
        return None
    for i in range(len(served) - earn_days + 1):
        window = served[i:i + earn_days]
        counts = [c for _d, c in window]
        if not mutually_non_separable(counts, alpha):
            continue
        rates = [k / n for k, n in counts if n]
        if len(rates) != earn_days:
            continue
        return {"anchor": (sum(k for k, _n in counts),
                           sum(n for _k, n in counts)),
                "envelope": (round(min(rates), 4), round(max(rates), 4)),
                "days": tuple(d for d, _c in window)}
    return None


def rows_to_history(rows, *, provider, channel):
    """Prior days for one channel, reconstructed from PUBLISHED rows.

    Returns `[(date, (k, n), verdict), ...]` oldest first, one entry per date.

    **This is the scheduled job's only memory, and it is a TRUST BOUNDARY.**
    `trace` rebuilds history from local disk cells; the job runs in an ephemeral
    container and `vworld corpus fetch` pulls only `eden_e*` and the raidex
    axis, never probe cells. So without this every scheduled run starts with
    today's cells alone, the rolling anchor is permanently empty, and a column
    can never earn an epoch either. Day one worked ONLY because it ran on this
    machine, where cells persist.

    Making it live means **the job trusts its own prior rows** — the published
    log becomes load-bearing for verdicts rather than a report of them. That is
    accepted for v1 because the raw cells are attached to each row and a reader
    can recompute any day from them; it is stated here rather than left
    implicit, because a trust boundary nobody wrote down is one nobody checks.
    See `probe.LOG_IS_LOAD_BEARING`.

    Later rows for a date win: a day republished after a correction should read
    as corrected, not as both.
    """
    seen: dict[str, tuple[str, tuple[int, int], str]] = {}
    for r in rows:
        if r.get("provider") != provider or r.get("channel") != channel:
            continue
        k, n, date = r.get("k"), r.get("n"), r.get("date")
        if k is None or n is None or not date:
            continue
        seen[date] = (date, (int(k), int(n)), r.get("verdict") or "")
    return [seen[d] for d in sorted(seen)]


def judge_day(cells, *, provider, date, channel, epoch, rolling, alpha,
              envelope=None, stale=False):
    """Judge one channel-day against BOTH anchors.

    `envelope` is the decision channel's block range. Where it is given, EVENT
    additionally requires landing **outside** it — pooling several occasions
    into one binomial gives a Wilson band narrower than the channel's real
    between-occasion spread, and firing on the band alone would call ordinary
    block-to-block movement an event.
    """
    today = [c for c in cells if c.date == date and c.provider == provider
             and channel_key(c.level, c.arm) == channel]
    now = _pool(today)
    models = tuple(sorted(c.model for c in today))

    if not now[1]:
        return Verdict(provider, date, channel, "NO-ANCHOR", "flat", now,
                       epoch, rolling, None, None, envelope, models,
                       "no cells served for this channel")

    moved_e, dir_e, p_e = _compare(now, epoch, alpha)
    moved_r, dir_r, p_r = _compare(now, rolling, alpha)

    if moved_e is None and moved_r is None:
        return Verdict(provider, date, channel, "NO-ANCHOR", "flat", now,
                       epoch, rolling, p_e, p_r, envelope, models,
                       "no anchor yet — admits with a flag, yields no verdict")

    moved = bool(moved_e) or bool(moved_r)
    direction = dir_e if moved_e else dir_r if moved_r else (dir_e or dir_r)

    if moved and envelope:
        lo, hi = envelope
        r = now[0] / now[1]
        if lo <= r <= hi:
            return Verdict(provider, date, channel, "QUIET", direction, now,
                           epoch, rolling, p_e, p_r, envelope, models,
                           f"significant vs an anchor but inside the block "
                           f"envelope [{lo:.4f}, {hi:.4f}] — the channel's own "
                           f"spread is wider than the pooled band")

    if stale and not moved:
        return Verdict(provider, date, channel, "STALE", direction, now,
                       epoch, rolling, p_e, p_r, envelope, models,
                       "no QUIET refresh within the window; HOLD for anything "
                       "event-conditioned")

    return Verdict(provider, date, channel, "EVENT" if moved else "QUIET",
                   direction, now, epoch, rolling, p_e, p_r, envelope, models)


def trace(cells, *, provider, channel, epoch, alpha, rolling_k,
          stale_after_days, envelope=None, earn_days=None, history=None):
    """Every day for one channel, oldest first, carrying rolling state forward.

    The rolling anchor is rebuilt as it goes and **EVENT days never enter it**.
    That is the taint law at daily cadence: a rolling window that absorbs the
    event is a window that calls the new level normal.

    **`earn_days` un-sticks a column that arrives with no epoch.** Without it
    NO-ANCHOR was ABSORBING: only QUIET days enter the rolling pool, and a
    channel with no epoch never returns QUIET, so it could never earn a rolling
    anchor either — 30 days of NO-ANCHOR at full price. Pass
    `probe.EARN_DAYS` and the column earns its own epoch by `probe.EARN_RULE`
    instead of borrowing one. Left None, behaviour is exactly as before.

    **`history` is the job's memory** — prior days from the published log, as
    `rows_to_history` returns them. It seeds BOTH pools: QUIET days into the
    rolling window and every served day into the earn candidates. Without it an
    ephemeral run sees today alone, so the rolling anchor stays empty forever
    and the earn window never reaches three. Dates also present in `cells` are
    ignored from history — the raw cell is the evidence, the row is the record
    of it, and where both exist the evidence wins.
    """
    import dataclasses
    import datetime as dt

    days = sorted({c.date for c in cells
                   if c.provider == provider
                   and channel_key(c.level, c.arm) == channel})
    quiet: list[tuple[str, tuple[int, int]]] = []
    #: The candidate pool: SERVED days only, oldest first. A day absent from
    #: `days` — SERVE_FAIL, VOID, a cadence day the column did not serve — never
    #: enters, which is how outages fail to restart the window.
    candidates: list[tuple[str, tuple[int, int]]] = []
    earned = None
    out = []

    for h_date, h_counts, h_verdict in (history or []):
        if h_date in days:
            continue
        if h_verdict == "QUIET":
            quiet.append((h_date, h_counts))
        #: Every row is a day that SERVED — a total serve failure writes no
        #: cells and therefore no row — so every row is an earn candidate,
        #: including NO-ANCHOR and EVENT ones. Agreement filters them, not
        #: the verdict word.
        candidates.append((h_date, h_counts))
    if earn_days and not epoch and candidates:
        earned = earn_epoch(candidates, alpha=alpha, earn_days=earn_days)
    for day in days:
        roll = _pool([]) if not quiet else (
            sum(k for _d, (k, _n) in quiet[-rolling_k:]),
            sum(n for _d, (_k, n) in quiet[-rolling_k:]))
        roll = roll if roll[1] else None
        stale = False
        if quiet:
            last = dt.date.fromisoformat(quiet[-1][0])
            stale = (dt.date.fromisoformat(day) - last).days > stale_after_days

        #: **An earned anchor applies from the day AFTER its window closes.**
        #: The three constituent days are judged with whatever the column had at
        #: the time — nothing — because an anchor cannot be evidence about the
        #: days it is pooled from. See `probe.EARN_EXCLUDES_ITS_OWN_DAYS`.
        use_epoch = epoch if epoch else (earned["anchor"] if earned else None)
        use_env = envelope if epoch else (earned["envelope"] if earned else
                                          envelope)

        v = judge_day(cells, provider=provider, date=day, channel=channel,
                      epoch=use_epoch, rolling=roll, alpha=alpha,
                      envelope=use_env, stale=stale)

        if earn_days and not epoch and not earned:
            candidates.append((day, v.now))
            earned = earn_epoch(candidates, alpha=alpha, earn_days=earn_days)
            v = dataclasses.replace(v, reason=(
                f"no anchor yet — {len(candidates)} served day(s) banked, "
                f"{earn_days} mutually non-separable needed"
                + (f"; EARNED on {day} from {', '.join(earned['days'])}"
                   if earned else "")))

        if v.verdict == "QUIET":
            quiet.append((day, v.now))
        out.append(v)
    return out
