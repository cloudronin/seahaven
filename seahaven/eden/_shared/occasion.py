"""The reference channel: is the environment quiet on this sweep?

A0 carries no treatment and is pinned near ceiling by design, so movement in it
is environment rather than behaviour. That makes it a free sensor — every sweep
already buys one. This module is the sensor.

**THE COMPARISON IS PAIRED, AND THAT IS NOT A REFINEMENT.**

The obvious construction — pool a sweep's A0 and compare it to the pooled
history — confounds the environment with who happened to be in the sweep. It is
not a hypothetical failure. On 2026-08-14 the pooled LAT A0 read 0.756 against a
0.953 history and looked like a clean event; decomposed, it split into a
returning group and a group never served at LAT before. The pooled figure mixed
a drop with a lower-baseline cohort.

Worse, the same pooled construction reported W2 and W3 as *quiet* that day, on a
cohort that shared no model with their history. A new cohort scoring 0.982
against a historical cohort's 0.949 is not evidence the environment held; it is
not a comparison at all. That reading was published in the spec and repeated in
prose before anyone decomposed it.

**THE FIGURES ABOVE ARE THE PUBLISHED ONES AND THEY ARE RETRACTED (#113).** The
decomposition that motivated this module was itself computed on cell metadata
that named models which had not served the cells: 166 cells across rounds 15-19
were served by one model. So "6 returning at 0.819 plus 8 newcomers at 0.708"
was not a cohort split, and re-read on true identity the whole sweep is one
model and QUIET. The argument for pairing is untouched — strengthened, if
anything, since the defect is another way a comparison can silently stop being
between the things it names. The numbers are left in place, struck, because
this docstring is also the record of why the module exists.

`_shared.identity` now gates every read on it: measurement keys on who SERVED.

So a model enters the test only if it has been served at that world on a
strictly earlier day, and it is compared **against itself**.

**HENCE A THIRD VERDICT.** A (sweep, world) with no returning models is
`NO-ANCHOR` — not `QUIET`. Calling it quiet is precisely the error above.

**NO-ANCHOR ADMITS, WITH A FLAG.** Settled by reductio rather than by taste: the
first sweep at any world has no returning models by definition, so a rule that
vetoed NO-ANCHOR would make the first observation of everything permanently
inadmissible, and the benchmark could never measure a new model at all. That is
not a detector, it is a halt. Ten of the eleven (sweep, world) pairs on record
are NO-ANCHOR — the veto reading would have vetoed the entire corpus.

**WHAT THIS CANNOT SEE**, stated here so it is stated everywhere the verdict is:
A0 sits at ceiling, so the channel is far more sensitive to degradation than to
improvement; and it senses only shifts that move *this* behaviour — acting, and
food-seeking. A serving change that moved only rule-conditioned decisions would
pass it. The verdict word is "quiet", never "verified stable".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..conditioning import stage_counts
from ..outcome import level_state, load_level
from . import corpus as C
from . import identity as ID
from .stats import fisher, wilson_ci

__all__ = ["WORLDS", "VERDICTS", "Observation", "Verdict", "observations",
           "verdict_for", "audit", "forbidden_item"]

#: The frozen three-world suite. COMP carries no forbidden item and LAT2 is a
#: round-14 boundary world with no A0 history, so neither is a reference site.
WORLDS = ("LAT", "W2", "W3")

VERDICTS = ("QUIET", "EVENT", "NO-ANCHOR")

_ITEMS: dict[str, str] = {}


def forbidden_item(world: str) -> str:
    if world not in _ITEMS:
        _ITEMS[world] = level_state(load_level(f"world_eden_{world}"))["forbidden"]
    return _ITEMS[world]


@dataclass(frozen=True)
class Observation:
    """One A0 cell, reduced to what the sensor needs."""

    world: str
    sweep: str
    day: str
    model: str
    ate: int
    n: int

    @property
    def rate(self) -> float:
        return self.ate / self.n if self.n else float("nan")


@dataclass(frozen=True)
class Verdict:
    """One (sweep, world) judged. `flags` report; they never veto."""

    world: str
    sweep: str
    day: str
    verdict: str
    returning: tuple[str, ...]
    newcomers: tuple[str, ...]
    now: tuple[int, int]
    prior: tuple[int, int]
    p: float | None
    flags: tuple[tuple[str, float, float, float], ...] = ()

    @property
    def tested(self) -> bool:
        return self.verdict != "NO-ANCHOR"

    def rates(self) -> tuple[float, float]:
        k1, n1 = self.now
        k0, n0 = self.prior
        return (k1 / n1 if n1 else float("nan"),
                k0 / n0 if n0 else float("nan"))


def observations(root=None) -> list[Observation]:
    """Every gen-3 A0 cell on the three-world suite, as `Observation`s.

    Diagnostic blocks are excluded here rather than by each caller: the occasion
    probe exists to measure between-day variation, so pooling it into the
    reference channel would average away the very effect the channel detects.
    """
    out: list[Observation] = []
    for path, cell in C.iter_cells(root=root):
        meta = cell.get("meta", {})
        got = C.parse_cell_name(path.name)
        if not got or got["schema"] != "current":
            continue
        if meta.get("eden_arm") != "A0" or meta.get("eden_level") not in WORLDS:
            continue
        if C.generation_of(meta) != "gen3" or C.is_diagnostic(meta):
            continue
        #: **The model is who SERVED, never who was asked for and never the
        #: filename.** 161 cells named eight models that one model served, and
        #: this line read the name — which is how a between-model comparison
        #: passed as a paired one. `assert_identity` refuses an uncorrected
        #: mislabelled cell outright; see issue #113.
        ident = ID.assert_identity(meta, where=f"observations({path.name})")
        counts = stage_counts(C.episodes(cell), forbidden_item(meta["eden_level"]))
        when, _src = C.occasion_of(path, meta)
        out.append(Observation(world=meta["eden_level"], sweep=got["round"],
                               day=when[:10], model=ident.served,
                               ate=counts["ate"], n=counts["n"]))
    return out


def _pooled(obs) -> tuple[int, int]:
    return sum(o.ate for o in obs), sum(o.n for o in obs)


def _ordered(obs: list[Observation]) -> list[tuple[str, str]]:
    """Every (world, sweep) oldest-first — the order taint must propagate in."""
    pairs = {(o.world, o.sweep) for o in obs}
    return sorted(pairs, key=lambda wp: (min(o.day for o in obs
                                             if (o.world, o.sweep) == wp),
                                         wp[0], wp[1]))


def event_pairs(*, alpha: float, obs: list[Observation]) -> set[tuple[str, str]]:
    """The (world, sweep) pairs judged EVENT, oldest-first.

    **A detector calibrated on the event it is catching is not a detector.**
    This exists because of what the re-serve will look like: eight of the
    fourteen models being re-served at LAT have exactly one prior LAT A0 cell,
    and it is the event day. Pooling all earlier days into the baseline would
    quietly put the 0.819 event INTO the reference the re-serve is judged
    against, so a repeat of the event would read as a match and the channel
    would confirm whatever it had already been fooled by once.

    Well-founded rather than circular: a sweep is judged only against strictly
    earlier days, so processing oldest-first means every taint decision is
    already made by the time it is needed.
    """
    tainted: set[tuple[str, str]] = set()
    for world, sweep in _ordered(obs):
        if _judge(world, sweep, alpha=alpha, obs=obs,
                  exclude=tainted).verdict == "EVENT":
            tainted.add((world, sweep))
    return tainted


def verdict_for(world: str, sweep: str, *, alpha: float,
                obs: list[Observation] | None = None) -> Verdict:
    """Judge one (sweep, world), excluding known events from the baseline."""
    obs = observations() if obs is None else obs
    return _judge(world, sweep, alpha=alpha, obs=obs,
                  exclude=event_pairs(alpha=alpha, obs=obs))


def _judge(world: str, sweep: str, *, alpha: float,
           obs: list[Observation],
           exclude: set[tuple[str, str]]) -> Verdict:
    """Judge one (sweep, world) against the returning models' own prior days.

    `alpha` is required. The programme's oldest process defect is a parameter
    that changes what a test means carrying a silent default — `mds` took one
    and two rounds disagreed about what they had measured.

    **Prior means a strictly earlier DAY, not merely a different sweep.** Rounds
    10, 12 and 13 all served LAT on 2026-08-13; comparing them to each other
    would compare within one occasion, which is not what a between-occasion
    sensor is for.

    **The test is two-sided, and the one-sidedness in the module docstring is a
    statement about SENSITIVITY, not about the rule.** A0 near ceiling means an
    upward move is hard to detect, not that it should be ignored when it does
    show: a channel that moved significantly in either direction is a channel
    whose environment differs from the reference, and admitting those components
    would be conditioning on a comparison already known to have failed. Vetoing
    both directions is the conservative reading and the cheaper mistake.
    """
    here = [o for o in obs if o.world == world and o.sweep == sweep]
    if not here:
        raise KeyError(f"no gen-3 A0 cells for sweep {sweep!r} at {world}")
    day = min(o.day for o in here)

    earlier = [o for o in obs if o.world == world and o.day < day
               and (o.world, o.sweep) not in exclude]
    anchors = {o.model for o in earlier}
    ret = tuple(sorted({o.model for o in here} & anchors))
    new = tuple(sorted({o.model for o in here} - anchors))

    if not ret:
        return Verdict(world, sweep, day, "NO-ANCHOR", ret, new,
                       _pooled(here), (0, 0), None)

    now = _pooled([o for o in here if o.model in ret])
    prior = _pooled([o for o in earlier if o.model in ret])
    p = fisher(now[0], now[1], prior[0], prior[1])

    flags = []
    for m in ret:
        a = _pooled([o for o in here if o.model == m])
        b = _pooled([o for o in earlier if o.model == m])
        pm = fisher(a[0], a[1], b[0], b[1])
        if pm < alpha:
            flags.append((m, a[0] / a[1], b[0] / b[1], pm))

    return Verdict(world, sweep, day, "EVENT" if p < alpha else "QUIET",
                   ret, new, now, prior, p, tuple(flags))


def audit(*, alpha: float, obs: list[Observation] | None = None) -> list[Verdict]:
    """Every (sweep, world) on record, oldest first.

    One pass, carrying the taint forward as it goes: a sweep judged EVENT is
    dropped from every later sweep's baseline.
    """
    obs = observations() if obs is None else obs
    tainted: set[tuple[str, str]] = set()
    out = []
    for world, sweep in _ordered(obs):
        v = _judge(world, sweep, alpha=alpha, obs=obs, exclude=tainted)
        if v.verdict == "EVENT":
            tainted.add((world, sweep))
        out.append(v)
    return out


def interval(k: int, n: int) -> tuple[float, float]:
    """Wilson, re-exported so the artifact reports intervals beside the p."""
    return wilson_ci(k, n)
