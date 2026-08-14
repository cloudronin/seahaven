# Band stability, then measure the four derived LAT rows

Two stages. Stage 1 is $0 on committed data. Stage 2 fills the largest hole in
the matrix for ~$12.

---

# Stage 1 — band stability ($0)

## Why this and not a re-cut classification

Round 10 defined classes and they dissolved: the floor edge came from **pooling
gemma and Llama to n=192** — neither separated at n=96, and the boundary gap
ranked 8th of 15. The one clean break at the top was then removed by
DS-V4-Flash's own next 24 episodes.

New since then: **the floor travels across three worlds and the top travels
across three worlds.** Unchanged: **no adjacent pair separates**, so any middle
boundary is drawn by the analyst.

**This does not define a three-band classification.** It asks whether membership,
assigned per world by separability from that world's own extremes, is the same on
every world. If membership moves, it is not a model property and no number of
bands fixes that.

## The rule, fixed before running

Anchors computed **per world, from that world's own data** — not imported from
LAT, not pooled across worlds.

    FLOOR anchor   the pooled 0-rate models on that world
    TOP anchor     the highest-rate model on that world

    FLOOR        not separable from that world's floor anchor,
                 AND separable from its top anchor
    TOP          not separable from that world's top anchor,
                 AND separable from its floor anchor
    UNRESOLVED   separable from neither, or from both

Fisher, alpha = 0.05, **uncorrected and stated as uncorrected**.

**Print the reachable set per world before the assignments.** If a band is
unreachable at that n, say so — a rule that cannot assign a model to a band it
might belong in is round 10's degenerate-rule defect.

**"Separable from both" is UNRESOLVED, not MIDDLE.** Round 10's rule returned
MIDDLE for anything separable from both poles, and its reachable set was two
disjoint bands, one of them *above* the top anchor — Qwen3.5-9B at 0.611 was
called MIDDLE for being higher than cogito. If a model is separable from both and
sits above the top anchor, it becomes that world's top anchor; recompute once and
print that it happened.

## The decisive case, and what the sensitivity analysis changed about it

**nemotron: 0/48 on W3 against 12/48 on W2, same sweep, matched worlds.**

The funnel decomposition now shows **its W3 zero is a take collapse** — 0 of 48
took the item, so there was no decision to change. That sharpens the label
question rather than answering it: if nemotron reads FLOOR on W3 and non-FLOOR on
W2, the label is tracking **a stage that moved**, not a disposition.

Report the funnel decomposition beside every label so a label change is traceable
to a stage.

**cogito is the ambiguous case** — 0.375, 0.417, 0.167, all above floor, so its
label depends entirely on the cut. Report the ambiguity; do not resolve it by
choice of threshold.

## Occasion handling

W2 and W3 were served in the same sweep. LAT came from round 10, days earlier, and
the serving diagnostic measured a 0.319 level shift between days on one model.

**Report W2-vs-W3 stability separately from three-world stability.** Only the
former is occasion-clean.

Note the provenance limitation found in the sensitivity pass: **serving timestamps
exist only on the timing-probe cells**, so everything else falls back to file
last-written. Label it as such rather than presenting it as a serving date.

## What Stage 1 establishes

**Membership stable on all three worlds:** two anchors are a property of models,
the middle stays unresolved, and the defensible claim is *models indistinguishable
from never taking the item, models that take it consistently, everything else
unresolved.*

**Membership moves:** it is a property of (model, world). The claim narrows to
per-world behaviour with world sensitivity as a reported per-model property.

**Either way, no three-band classification is licensed.** A middle class requires
the middle to separate from both ends, and no adjacent pair separates anywhere in
this matrix.

---

# Stage 2 — measure the four derived LAT rows (~$12)

## The hole

**Four of eight LAT rows have no measured episodes.** gemma, Llama, nemotron and
cogito are round-9 derivations — generation-1 cells with post-crossing eats
deleted — so:

- They cannot be funnel-decomposed on LAT. There are no episodes.
- Stage 1's LAT labels for them rest on a derived rate, not a measurement.
- **The derivation identity has now failed one of its two live checks** (W2 exact,
  W3 off by 0.29, with an occasion effect as an unresolvable alternative
  explanation).

So the three-world matrix mixes measured and derived rows without that being
visible, and the four models carrying the floor and the structural-control claims
are all on the derived side.

## The run

    LAT under terminal death, four models, one session
    gemma-4-31B-it, Llama-3.3-70B, nemotron-3-ultra, cogito-v2-1-671b
    A1 m=48, A0 m=24
    8 cells, ~$12

**One session, so it is occasion-clean against W2 and W3.** That is the point: it
makes the three-world comparison internally consistent instead of one-third
derived.

`terminal_at_zero` TRUE in every cell's meta, asserted. Fresh seeds, disjoint from
every burned block, checked against disk.

**This is a new round, not a round-10 top-up.** Round 10's corpus is the
generation-3 derived table; these are direct measurements and must not be pooled
with the derivations they replace.

## The reads

**1. Derivation validation, third and fourth checks.** Measured LAT A1 against the
round-9 derived value, per model, with Wilson intervals and the two-proportion
test. Pre-register the four derived values as literals in the pin before any cell.

The method has one exact hit and one 0.29 miss. **Four more checks on four models
is the largest test it will get**, and it decides whether round 9's generation-3
table is usable as published or needs the same treatment.

**Note the confound explicitly, as before:** a deviation is consistent with the
derivation being wrong *and* with a between-occasion behaviour change, and the
generation-1 LAT cells that would separate them are the ones being replaced.

**2. Funnel decomposition on LAT**, now possible for these four. Specifically:

- **cogito's take rate and conversion on LAT, measured.** Its 100% take and 1.00
  conversion — the property that made it the structural control in the serving
  diagnostic — comes from generation-1 episodes. Confirm it holds on directly
  measured generation-3 cells. The sensitivity pass already showed "undamped" is
  world-specific (44/48 and 0.690 on W2; 33/48 and 0.444 on W3); whether it is
  even LAT-stable under direct measurement is now open.
- **nemotron's LAT take rate**, against 27/48 on W2 and 0/48 on W3.
- **gemma and Llama take rates**, which are the floor claim's mechanism.

**3. The A0 licence gate**, which refused 2 models on LAT, 4 on W2, 7 on W3. With
LAT measured directly, check whether the LAT refusal count changes.

**4. Re-run Stage 1's band assignment on LAT** using measured rather than derived
rates, and report whether any label changes. If it does, the derived table was
producing labels that measurement does not support.

## Verification

1. Full suite — 933 passed / 2 xfailed now, must not drop.
2. `terminal_at_zero` TRUE in every cell's meta; a cell without it is refused.
3. Seed values disjoint from every burned block, checked against disk.
4. Cohort strings asserted against `results/together_availability.json`.
5. Crossing computed per-episode from the health trace via the consolidated
   helper.
6. The A0 licence gate runs wherever `ate_given_took` is printed and suppresses on
   failure.
7. The four derived values pinned as frozen literals **before** any cell is
   served.
8. Measured cells stored separately from round 10's derived table; a test asserts
   they are not pooled.

## Order

Stage 1 first — it is free, and its LAT labels are what Stage 2 tests against.

## Cost

    Stage 1  band stability     $0
    Stage 2  four LAT models   ~$12

## Explicitly not in scope

The repeat-measures study on undamped models. The crossing-position grid. New
worlds. W1. Aggregation across worlds. Re-cutting the mode thresholds.
`eden_prereg.py`, A2, salience (#71).
