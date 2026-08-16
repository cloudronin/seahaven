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
           "read_cells", "channel_key", "judge_day", "trace"]

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
          stale_after_days, envelope=None):
    """Every day for one channel, oldest first, carrying rolling state forward.

    The rolling anchor is rebuilt as it goes and **EVENT days never enter it**.
    That is the taint law at daily cadence: a rolling window that absorbs the
    event is a window that calls the new level normal.
    """
    import datetime as dt

    days = sorted({c.date for c in cells
                   if c.provider == provider
                   and channel_key(c.level, c.arm) == channel})
    quiet: list[tuple[str, tuple[int, int]]] = []
    out = []
    for day in days:
        roll = _pool([]) if not quiet else (
            sum(k for _d, (k, _n) in quiet[-rolling_k:]),
            sum(n for _d, (_k, n) in quiet[-rolling_k:]))
        roll = roll if roll[1] else None
        stale = False
        if quiet:
            last = dt.date.fromisoformat(quiet[-1][0])
            stale = (dt.date.fromisoformat(day) - last).days > stale_after_days
        v = judge_day(cells, provider=provider, date=day, channel=channel,
                      epoch=epoch, rolling=roll, alpha=alpha,
                      envelope=envelope, stale=stale)
        if v.verdict == "QUIET":
            quiet.append((day, v.now))
        out.append(v)
    return out
