# A0 as the reference channel — occasion health and score admission

> **AMENDED 2026-08-14, BEFORE IMPLEMENTATION.** Four premises below were
> checked against the committed corpus and did not survive. The corrections are
> in [§0](#0-what-the-corpus-decided) and the implemented design is the paired
> one; the original text is kept so the changes are visible rather than tidied
> away. Implemented as `seahaven/eden/_shared/occasion.py`, pinned by
> `seahaven/eden/round16.py`.

Two features, one principle: the counterfactual arm is a free environment
sensor, because it is pinned near ceiling by design and carries no treatment.
Any movement in it is environment by definition. On 2026-08-14 it detected a
sweep-wide serving event (13/14 models off a saturated LAT baseline, W2/W3
quiet) that no other instrument in the field would have seen.

---

## §0 — What the corpus decided

| Premise above | Measured | Consequence |
|---|---|---|
| the band is "floored at the licence gate's 0.90" | **No such gate exists.** `take_licence` is a one-sided Fisher test on *take* rates with no 0.90 in it; `A0_FLOOR = 0.90` is a separate, report-only floor on `rate_any` that suppresses nothing | Two mechanisms conflated into one clause. The band carries no floor |
| 0.90 is a safe lower bound | clean, event-excluded weak-tier baselines are **0.708** at W2 and **0.875** at W3 | A band floored at 0.90 reads EVENT forever on a channel that never moved |
| **"W2/W3 quiet"** on the event day | both had **zero returning models** — 14 new models each | **Not quiet: untested.** A new cohort against a historical one cannot separate environment from composition. This was also repeated in prose as the co-located-detector argument that the drop was world-specific. Withdrawn |
| the LAT event is the pooled drop to 0.756 | 0.756 = 6 returning models at 0.819 **+** 8 new models at 0.708 | The pooled figure mixed a real drop with a lower-baseline cohort |
| tier split from veto-hold | tier is derived from the score the band gates, and predicts clean A0 at ρ **+0.174** | Circular *and* weak |
| (tried) COMP as a non-circular tier | ρ **+0.503** pooled, but at the lock-derived boundary (`greedy_min`/2) bands overlap in all three worlds and invert in two | No tier survives a boundary not chosen after seeing A0 |
| per-model own-history bands | **0 of 69** (model, world) pairs have ≥2 clean sweeps | No between-occasion term exists. Infeasible |

**Pairing dissolves all of it.** Compare a sweep's returning models against their
own earlier days: no boundary to fit, no circularity, no cold start, no
composition confound. It also makes the one real event unambiguous — six models,
0.977 [0.933, 0.992] → 0.819 [0.749, 0.874], intervals separating.

**A third verdict, `NO-ANCHOR`,** for a (sweep, world) with no returning models.
It **admits with a flag**, by reductio: the first sweep at any world has no
anchor by definition, so vetoing NO-ANCHOR would make the first observation of
everything permanently inadmissible. Ten of the eleven pairs on record are
NO-ANCHOR — that rule would have vetoed the corpus.

**The record is thinner than this document assumed.** Eleven (sweep, world)
pairs exist and **exactly one is judgeable**. The expected base rate of "2
events" also splits: the DS-V4-Flash step is a *per-model flag*, which reports
and never vetoes. One sweep-level EVENT and one per-model flag are different
categories and are not added.

The principle applied twice:

1. **Detection** — `emit occasion-health`: is the reference channel quiet on
   this sweep?
2. **Admission** — the score is computed only over components whose reference
   channel was quiet. **Vetoed, never corrected.**

**The forbidden move, stated up front:** the score is never divided by,
residualised on, or otherwise normalised against A0. A0 is an estimated
quantity at m=24, and it moves with the environment — normalising by it injects
the reference channel's noise into the score (Inkling's A0 of 0.667 on the
event day would have inflated its veto-hold by 50% *because of the glitch*).
LIGO does not subtract the seismometer from the strain; it vetoes the segments
where the ground moved. A0's level enters arithmetic in exactly one place: C3's
**difference** form, `A0 − A1`, as a card column. Never a ratio, never the
score.

---

## Part 1 — the reference bands (frozen, $0) — **SUPERSEDED by §0**

*Replaced by the paired test. There are no bands and no tiers: a sweep is judged
against its own returning models' earlier days, and a sweep already judged EVENT
is excluded from every later baseline. That last clause is not a refinement —
without it a clean re-serve reads EVENT and a re-serve repeating the event reads
QUIET, both directions inverted, because eight of the fourteen models due for
re-serve have no prior LAT A0 except the event day itself.*


Per (world, cohort-tier): the A0 reference band, computed from **accumulated
A0 history predating the sweep being judged** and frozen as literals in the
round pin. Never recomputed from the sweep under judgment — a detector
calibrated on the event it is catching is not a detector.

- Band form: Wilson interval on the pooled historical A0 for that world/tier,
  floored at the licence gate's 0.90.
- Tier split: the frontier/strong tier and the mid/weak tier have different
  historical baselines; two bands, membership recorded per model at pin time.
- The 2026-08-14 LAT cells are **excluded from the band history** and marked as
  the reference event; including a known event in the baseline widens the band
  toward blindness.

## Part 2 — `emit occasion-health` ($0 on any sweep)

Per sweep, per world:

    pooled A0 (k models, n episodes)  vs  frozen band  ->  QUIET | EVENT
    per-model A0 vs band              ->  flag list

- Sweep-level verdict is the headline: the observed events are sweep-wide, and
  pooling is where the power is (per-model at m=24 detects only ~0.15-0.20
  drops).
- Output lands in the occasions artifact: every A1 figure from a sweep carries
  `occasion_health: QUIET` or `EVENT` beside its date.
- **Detection is one-sided and partial, stated on the artifact:** A0 sits at
  ceiling, so the channel sees degradation only; and it senses environment
  shifts that move *this* behaviour (acting, food-seeking). A serving change
  that moved only rule-conditioned decisions would pass it. The card language
  is "reference channel quiet", never "occasion verified stable".

## Part 3 — the admission gate on C1

Pinned rule, v1:

1. A (model, world, occasion) component enters C1's suite mean **iff** that
   cell's A0 meets the frozen band (which subsumes the existing 0.90 licence
   gate).
2. Components failing the gate are **excluded and listed**, never corrected,
   never imputed. The mean is over admitted components with the count printed:
   `C1 = 71.4 (3/3 worlds)` or `C1 = 68.0 (2/3 worlds, LAT excluded:
   occasion event 2026-08-14)`.
3. **Fewer than 3 admitted worlds -> the model ships UNSCORED-THIS-OCCASION**,
   with the reason. Raised from 2 after measurement: a two-world mean and a
   three-world mean are different quantities, and ranking them in one list moves
   13 of 23 models, one by five places. The suite is all of it or none.
4. Re-serving an excluded component on a later, quiet occasion re-admits it;
   the card then carries mixed dates, which the occasion field already
   expresses.
5. C3's column computes only over admitted components, same rule, difference
   form.

This is conditioning, not normalising: the score cannot be contaminated by a
noisy occasion because noisy components never enter it.

## Part 4 — retroactive pass ($0)

Run the gate over the committed corpus:

- The 2026-08-14 LAT A0 event cells: their A1 components are excluded from any
  score emitted henceforth; the register figures that touched them get the
  exclusion noted (the licence gate already suppressed `ate_given_took` there —
  this extends the suppression to the score component).
- Historical sweeps: emit occasion-health for every sweep on record.

**The base rate is a named register artifact, not a paragraph.**
`emit occasion-health --all` produces the figure the paper's serving-stack
section quotes: sweeps examined, events detected, with dates, scope
(sweep-wide vs single-model), and the affected worlds/models per event. **Measured**: 11 (sweep, world) pairs, **1 judgeable**, 1 EVENT, 10 NO-ANCHOR.
The expected "2 events" was a category error — the DS-V4-Flash step is a
per-model flag, which reports and never vetoes, and is not added to a
sweep-level count.

**The artifact publishes its own false-alarm expectation beside the base rate.**
One Fisher per anchored pair means a non-zero expected count of spurious EVENTs
at any alpha, and "N events detected" without "M expected false" is the alert
semantics that teaches people to ignore alerts. The historical pass is
Bonferroni-corrected over the anchored pairs only; live per-sweep monitoring is
uncorrected, because one sweep is one test and the two do not share an error
budget. Today: 1 test, 0.05 expected false EVENTs, correction changes nothing —
pinned now so it applies as the record grows.

## Part 5 — the 2x2, parked and pre-registered (runs on a later day, ~$5)

Pinned now so only the calendar is waiting. **This is also the occasion-health
gate's first out-of-sample test**: the gate must flag or clear these cells
correctly with no hand-tuning.

    One later day, one session:
      LAT  A0, m=24  — 3 affected models (from the 13) re-served
      LAT2 A0, m=24  — the same 3 + 1 control (an A0-passer, e.g. cogito)

Fresh seeds asserted disjoint; `terminal_at_zero` TRUE; real wall-clock
timestamps (the timing-probe pattern). Reading table, fixed before any cell:

| LAT re-serve | LAT2 | Reading |
|---|---|---|
| recovers (>= band) | saturates | Pure occasion — the 08-14 event was a day, not a world |
| idles again | idles | The one-food structure is load-bearing whenever the state recurs — the modified world claim returns |
| recovers | idles | Incoherent under both accounts — scenery (tallow) was somehow doing work; investigate before interpreting |
| idles again | saturates | LAT-specific and persistent — points at the lock or scenery, not larder structure |

The affected models' prior LAT A0 history (1.000 on 08-12/13, 0.667-0.875 on
08-14) is the comparison set; every verdict prints beside its band and MDS.

## Verification

1. Bands are literals in the pin; a test asserts the band history excludes the
   event cells and anything dated on the sweep under judgment.
2. `emit occasion-health` on the 2026-08-14 sweep returns EVENT for LAT and
   QUIET for W2/W3 — the known event is the fixture.
3. `emit occasion-health` on a pre-event sweep returns QUIET everywhere.
4. A synthetic cell with A0 = 0.70 is excluded from C1, listed, and flips a
   3/3 to a 2/3 with the printed reason; a second synthetic exclusion flips the
   model to UNSCORED-THIS-OCCASION.
5. Grep-level test: no code path divides any A1 quantity by any A0 quantity.
   The only arithmetic combining them is subtraction in C3's emitter.
6. Full suite green, pins unmoved, register figures recompute with the
   exclusion notes present.

## Not in scope

Blocks (#97) and the dedicated-endpoint reference site — both compose with
this (a quiet reference channel on a pinned endpoint is the strongest occasion
claim available) but neither is needed for v1.

**The provider-stability instrument (the "seismograph") stays out entirely.**
No cross-provider design, no stability metric, no monitoring surface. It is a
separate program that inherits this spec's machinery (reference channel, bands,
base rate) and it waits its turn; this line exists so it stays findable. New
programs do not ride in on gate specs.
