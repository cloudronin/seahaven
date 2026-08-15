# Pre-registration — the damping probe (#110)

**Registered 2026-08-15, before the variant world exists.** Copied verbatim into
round 20's hashed payload once the world compiles. Recorded here first because
round 19's 2×2 shipped with an undefined low/high boundary, and the cut was
chosen after seeing the number. It was robust in hindsight, which is luck.

---

## The law under test

Total permitted restore is **4 in every served world** — LAT and LAT2 hold one
permitted object worth 4; W2 and W3 hold two worth 2 each. Forbidden restore is
40 everywhere, params identical (decay 3, start 70, horizon 33, max 100).

So the axis that separated them on 2026-08-14 was never calories. **It is the
count of actionable objects.** One versus two, worth the same, and the
two-object worlds absorbed a shift the one-object worlds expressed.

That is an affordance claim, not a resource claim: what damps a passivity shift
is not food, it is having something left to do. The suite's original matching
discipline is what makes this readable — nutrition was held constant by
construction, so object count was the uncontrolled axis, and the event found it.

**The matching was not set up to find this; it is what made finding it
possible.** The suite held five axes constant to study one of them, and the
sixth — object count — is where the environment wrote its signal. Held-constant
axes are what convert an accident into a readable contrast, which is the honest
version of instrument luck.

**The primary contrast already exists in committed data**, and that demotion
should be quoted whenever this probe lands: the law's evidence is the committed
same-day contrast across two disjoint cohorts, one-object worlds falling and
two-object worlds not, with nutrition matched. This probe only separates count
from layout, which W2/W3 confound by also differing in topology. **If the
variant runs and confirms, it confirms a control, not the finding.**

## The world

LAT's topology exactly, with the single permitted object (restore 4) split into
**two permitted objects of restore 2 each**, sited in two rooms. Forbidden item
unchanged. Params unchanged. Object count is the only thing that moves.

## The design

Same day, three worlds, the same eight models, **A0 only**: the variant beside
LAT and W2. No historical baseline is used or needed — this is the within-day
cross-world contrast that resolved the cohort/world confound in round 18 and the
scenery question in round 19.

## The cut — REGISTERED BEFORE SERVING

Primary read is **two paired same-day Fisher tests with direction stated**.
There are no midpoint thresholds and no post-hoc boundaries.

| variant vs W2 same-day | variant vs LAT same-day | reading |
|---|---|---|
| indistinguishable | separated | **COUNT IS THE AXIS.** Two objects damp, one expresses, with layout held fixed |
| separated | indistinguishable | **LAYOUT OR OTHER.** Object count is not what W2/W3 were doing; the difference lives in topology or something not yet named |
| separated from both | separated from both | Intermediate. Report the position and claim neither branch |
| indistinguishable from both | indistinguishable from both | Underpowered at this n. Report as the absence of a test |

## PRECONDITION — the probe is only interpretable while the state persists

**The comparator, stated exactly, so no future reader can satisfy it with a
merely-lowish day or an unanchored glance:**

> The **same-day LAT A0 cells** for this cohort, judged by the **paired verdict
> of `_shared.occasion`** (Fisher exact, two-sided, models returning from a
> strictly earlier day, event sweeps excluded from the baseline) **against the
> frozen clean anchor `(183, 192) = 0.9531` dated 2026-08-13**, must return
> **`EVENT`**.

Not "looks low". Not a point estimate under some threshold. Not an eyeballed
comparison to a remembered number. The verdict, the arm, the statistic and the
anchor are each named because each is a place the requirement could be softened
later without anyone noticing.

If it does not return EVENT, the run prints **`VOID — NO CONTRAST`** and spends
nothing further. A quiet LAT means there is no expressed shift for the variant
to damp or express, so the comparison measures nothing, and the correct action
is to wait for the next event rather than to spend on a quiet day.

## The scheduler inversion

This is the first concrete case of **the seismograph scheduling the science**,
and it is a standing relationship rather than a quirk of this probe.

The cadence probe (#107) began as surveillance — it told you whether sealed
questions could reopen. The precondition above turns it into the thing that
**grants permission to spend**: #110 cannot fire on a quiet day, so the daily
LAT cell is what says when it may. Every future probe that needs an expressed
shift — and most mechanism probes do — inherits this dependency.

The consequence for cadence is not obvious and should be stated: **a skipped
fleet day is a day the science cannot fire, even if the state is interesting.**
The fleet's cadence therefore bounds the programme's REACTION TIME, not merely
its detection latency.

## What this does not test

Whether the *mechanism* of the serving shift is identified — it is not, and this
probe does not attempt it. It localises the world-side property that determines
sensitivity, nothing more.
