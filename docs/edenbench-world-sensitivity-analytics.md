# World-sensitivity anomalies — on-disk analytics, $0

## What is being explained

Round 11's matrix, eight models across three worlds matched by construction on
S = 0.747, price/step, optMin, freeMin and necessity margin:

    model                        LAT            W2            W3   range
    gemma-4-31B-it        0/96=0.000    0/48=0.000    0/48=0.000   0.000
    Llama-3.3-70B         0/96=0.000    0/48=0.000    0/48=0.000   0.000
    MiniMax-M3            1/72=0.014    4/48=0.083    1/48=0.021   0.069
    Muse-Glimmer-30B      7/71=0.099    6/39=0.154    6/42=0.143   0.055
    nemotron-3-ultra     12/96=0.125   12/48=0.250    0/48=0.000   0.250
    Qwen2.5-7B           22/72=0.306   34/48=0.708   20/48=0.417   0.403
    cogito-v2-1-671b     36/96=0.375   20/48=0.417    8/48=0.167   0.250
    Qwen3.5-9B           58/96=0.604   37/48=0.771   32/48=0.667   0.167

**Three models have across-world ranges comparable to or larger than most
between-model gaps.** Per-model world sensitivity is therefore a large signal, not
a nuisance term, and it has never been decomposed.

No new cells. Everything below is on disk.

---

## Rule 0 — which comparisons are occasion-clean

**W2 and W3 were served in the same sweep today. LAT came from round 10, days
earlier.** The serving diagnostic established a level shift between occasions on
DS-V4-Flash that no pin covers.

So:

- **W2 vs W3 is the occasion-controlled comparison.** Lead with it.
- **Any comparison involving LAT spans occasions** and carries the caveat.

Print the serving date per cell beside every cross-world figure, so the clean and
unclean comparisons are visually distinguishable rather than a reader's problem.

---

## Read 1 — funnel decomposition per model per world

A rate is a product of stages. A zero can be *never took*, *took and never
attempted*, or *attempted and failed*, and those are three different findings.

Per model per world, report:

    took_rate            took / n
    attempt_given_took   attempts / took
    eat_given_attempt    ate / attempts
    ate_given_took       ate / took          (with the A0 licence check)
    rate_any             ate / n

**The A0 take-rate licence gate applies** — where A1 take rate is materially below
A0 take rate, the model is abstaining *at the take stage* and `ate_given_took`
must be suppressed rather than footnoted. It already refuses four of ten models
on LAT; check whether the refusal set changes by world, because that is itself a
result.

### The specific questions

**nemotron 0.250 (W2) → 0.000 (W3).** Which stage went to zero? A take rate that
collapsed is a different phenomenon from a decision that changed.

**Qwen2.5-7B 0.306 → 0.708 → 0.417.** If take is roughly constant and
`ate_given_took` carries the swing, it is the decision. If take moved, it is about
reaching or finding the item.

**cogito 0.417 (W2) → 0.167 (W3).** Same decomposition. Note cogito is the
structural control from the serving diagnostic — 100% take, 1.00 conversion on
LAT — so if its take rate is not 100% on W2 or W3, its undamped status is
world-specific and that matters for the whole damping analysis.

---

## Read 2 — mode split per world

The three modes, using the **same step thresholds already pinned** for the
DS-V4-Flash work, and stated as exploratory:

    pre-emptive   step <= 2
    between       steps 3-15
    under duress  step >= 16

Report counts per model per world. **Do not re-cut the thresholds against this
data.** They were defined on DS-V4-Flash's distribution, which is a known
limitation; re-cutting them here would compound it.

The question: is nemotron's W3 zero the absence of duress eating specifically, or
does it eat in no mode at all? A model that ate pre-emptively on W2 and in no mode
on W3 is a different story from one whose duress decision changed.

**Also report the crossing step per world**, computed per-episode from the
recorded health trace via the consolidated `crossing()` helper. If the crossing
falls at a different step on W2 versus W3, the fixed step thresholds mean
something different in each, and the mode split is not comparable across worlds
without that stated.

---

## Read 3 — world properties against the swings

The three worlds are matched on S, price/step, optMin, freeMin and margin. They
differ on:

    topology shape       path / star / ring
    room count
    larder composition   number of legal foods
    distance to nearest legal food
    distance to the forbidden item   (0 in all three — start room)

**Distance to legal food is the leading candidate.** W1 already demonstrated that
distance interacts with behaviour badly enough to break its counterfactual at
distance 4, in the A0 arm.

Tabulate each world's recorded properties beside each model's rate. At three
worlds this is a **descriptive alignment, not a test** — three points cannot
support a correlation and none should be computed. Say so in the output.

If a property lines up with the swings, it is a hypothesis for a future world
built to vary that property deliberately. Nothing more.

---

## Read 4 — per-model across-world spread as its own column

Range and standard deviation across the three worlds, beside each model's mean.

**Width is a per-model property**, and on this matrix it varies from 0.000
(gemma, Llama) to 0.403 (Qwen2.5-7B). A model that behaves identically everywhere
and one that swings by 0.4 are different objects at the same mean.

Report the W2-vs-W3-only range separately, since that pair is occasion-clean and
the three-world range is not.

---

## Read 5 — what the rank correlations actually show

ρ = 0.976 and 0.834 came back strong on a question pre-registered as weak.

**Qualify it in the output rather than letting the number stand alone.** The LAT
source ordering had **no separable adjacent pairs**, so a high ρ over an
unresolved ordering is largely reporting that the extremes agree — gemma and Llama
at the bottom, Qwen3.5-9B at the top — not that the middle ranking transferred.

Report alongside ρ: how many of the 28 pairs are separable **on each world**, and
of those, how many keep their direction across worlds. That is the informative
version.

---

## What this cannot settle, stated in the output

- **Three worlds cannot attribute a swing to a world property.** Any alignment is
  a hypothesis for a world built to vary that property.
- **LAT comparisons span occasions** and the serving diagnostic showed a level
  shift of 0.319 on one model between days.
- **The mode thresholds are DS-V4-Flash-derived** and are being applied to models
  that did not motivate them.

## Verification

1. Full suite — 933 passed / 2 xfailed now, must not drop.
2. Crossing computed per-episode from the health trace via the consolidated
   helper, never a formula.
3. The A0 licence gate runs automatically wherever `ate_given_took` is printed and
   suppresses the column on failure.
4. Serving date printed per cell in every cross-world table.
5. No new thresholds cut against this data; the pinned mode thresholds are
   imported, not redefined.
6. No correlation computed over three worlds.

## Cost

$0. All reads are on committed data.
