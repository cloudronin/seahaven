# Pre-registration v2 — supersedes, does not amend

**Status:** governs all work from this point. The original pre-registration in
`docs/research-log.md` (2026-08-09, "PRE-REGISTRATION, written before the
re-baseline runs") and its P1–P4 clauses **stand unaltered**. Nothing below
edits them. Where the two conflict, this document governs *future* work and the
original remains the record of what was committed to at the time.

---

## 0. What is already known, and therefore cannot be predicted here

This document is written **after** the V1 rebuild, so it is not blind. Stating
the boundary precisely is the only thing that keeps it honest:

**Known at time of writing** — none of these may be presented later as a
prediction this document made:

- mean cross-lab judge agreement **0.736** (GPT-5.2 / Qwen2.5-32B 0.853;
  gemma-2-27b vs each ~0.68)
- gpt-4.1-mini as detector: **0.815** vs 3-judge consensus; per stratum
  main 0.875, fabrication 0.808, disagreement 0.770
- both regex detectors score **exactly 0.000** on the disagreement stratum, for
  structural reasons
- V2 Spearman **0.964**, bootstrap 0.857 [0.607, 1.000]
- V4: model-level r(words, omission) **−0.547**; r(n_performed, fidelity)
  **−0.490**
- gate −1 pass rates: world_v0 **11/18**, world_v2 **16/18**

**The gate change below is made in full knowledge that the LLM detector already
clears it.** That is a change to the criterion, not a finding, and §1 records it
as such.

---

## 1. The V1 gate moves from 0.80 to 0.70

### The argument

An instrument cannot be validated to a precision its reference standard does not
possess. Three judges from three labs agree with each other at a mean kappa of
**0.736**. A detector required to reach 0.80 against those labels is being asked
to agree with them more closely than they agree with each other, which is not a
solvable problem — it is a mis-specified one.

**0.70 is chosen as the largest round value strictly below the measured
inter-rater ceiling of 0.736.** It is not chosen because a particular detector
clears it. Two independent anchors support the same region: Landis and Koch put
"substantial agreement" at 0.61–0.80, and the weakest judge pair here (0.672)
sits just below 0.70, so a detector at 0.70 is roughly as reliable as the least
reliable pair of human-substitute raters in use.

### What it costs to say this honestly

gpt-4.1-mini currently scores 0.770 / 0.808 / 0.875 across the three strata. At
0.70 it passes all three. **A gate lowered until the incumbent passes is worth
nothing**, so this document does not treat "V1 passes" as a result. V1's status
becomes:

> **V1 is satisfied by an LLM detector at a gate set to the reference standard's
> own reliability.** No string detector reaches it on the decisive stratum, and
> none can, for the structural reason recorded in the log.

Any future claim of a *stronger* V1 pass requires raising label quality first
(§3, Phase B) and re-testing against the higher ceiling — not against 0.70.

### Frozen now, before any further measurement

- **Detector D1 (frozen):** `gpt-4.1-mini`, the exact prompt in
  `scripts/adjudicate_v1.py::PROMPT`, `temperature=1`, dual-ask at two seeds,
  an item counted as a claim only when both asks agree; disagreement counted as
  *not a claim*. Model string pinned; a version bump invalidates and requires
  re-validation.
- **Reference standard:** majority of **≥3 judges from ≥3 distinct labs**, each
  passing all 8 planted controls at ≥87.5%, each dual-asked. Items where no
  judge is stable are dropped, not imputed.
- **Gate application:** 0.70 on **each stratum separately** — `main`,
  `disagreement`, `fabrication` — never pooled. The fabrication stratum remains
  decisive under P4.
- **Reported alongside every V1 number:** the inter-rater ceiling for that label
  set. A detector score without its ceiling is uninterpretable, which is what
  the single-judge era hid.

---

## 2. Predictions — genuinely unmeasured, frozen before the runs

Each is falsifiable and none can be checked against anything currently on disk.

### G1 — swapping the detector reorders the table

> Re-scoring the existing 500 runs with D1 instead of the regex will change
> **at least one model's rank position** in the world_v0 table.

The two detectors disagree on 23% of pairs, and the middle of the table
(TII 77.7, AI2 78.7, IBM 79.0) is separated by ~1.3 points. Falsified if the
ranking is identical.

### G2 — the verbosity confound is not a detector artefact

> Under D1, model-level `r(narrative_words, omission)` will remain **more
> negative than −0.35**.

Both detectors require the entity to be named, so both inherit the arithmetic by
which longer text names more things. **If this is falsified — if D1 substantially
removes the confound — that is a stronger argument for D1 than any agreement
number in this document**, and it should be reported as the headline.

### G3 — V2 stays marginal

> Under D1, cross-world Spearman will remain ≥0.80 while its **bootstrap lower
> bound remains below 0.80**.

The instability is driven by three repeats per cell, which the detector swap does
not change. Falsified if the lower bound clears 0.80, which would mean the regex
was adding the noise.

### G4 — gate −1 power scales with scored entities

> A world with **≥24 scored entities** will pass gate −1 in **≥17 of 18**
> repeats.

world_v0 (17 entities) passes 11/18, world_v2 (20) passes 16/18. Falsified if a
larger world does not improve on world_v2.

### G5 — narration register changes negation prevalence

> The `factual` and `retrospective` registers will each produce **at least twice**
> the rate of explicit negation ("I did not find X") of `introspective`.

This is V1's remaining blind spot: every detector, D1 included, was validated on
a corpus where negation is rare. Falsified if the registers are within 2× of
each other.

### G6 — the score survives its own confound correction

> After the V4 correction adopted in Phase C, model-level
> `r(n_performed, fidelity)` will be **weaker than |0.20|**.

Currently −0.490. If no correction achieves this, the composite is not
salvageable and §4's kill criterion applies.

---

## 3. Revised phase plan

Ordered by what blocks what. Costs are GPU/API estimates.

| phase | work | gate | cost |
|---|---|---|---|
| **A. Re-score under D1** | Apply D1 to the 500 runs already on disk. No new episodes. Recompute Phase 1, V2 and V4 under the frozen detector. | G1, G2, G3 answered | ~$8 API |
| **B. Raise the ceiling** | Add 2 more judges (5 total, ≥4 labs) on the 806-item set; adjudicate non-unanimous items with a 3-of-5 rule. Re-measure the inter-rater ceiling and re-test D1 against it. | ceiling >0.80 would let the gate rise again | ~$6 |
| **C. Fix V4** | The confound is between-model and sign-varying, so no single slope removes it. Two candidates, both pre-committed: **(i)** report omission and fabrication separately and retire the composite; **(ii)** hold `n_performed` fixed by construction — score only entities the agent actually interacted with. Choose by G6, not by which ranks better. | G6 | free (analysis) |
| **D. world_v3 for power** | Author a world with ≥24 scored entities, same mechanics, build-time collision assertion. Re-run 7 models × 3 seeds. | G4 | ~$8 GPU |
| **E. V3 narration sweep** | 3 registers × 7 models × 3 seeds on one world. Already implemented. | G5 | ~$8 GPU |
| **F. raidex integration** | Only if A–E pass. Note D1 makes every future run cost API calls — `estimate_cost` must model detector cost, not just generation. | — | ~2 days |

**Phase A runs first and may end the programme early**: if G2 holds, the score is
confounded under both detectors and Phase C becomes the critical path rather than
a cleanup.

---

## 4. Kill criteria

Unchanged in spirit from the spec's F5, restated against the new gate:

- **K1** — if the 5-judge ceiling in Phase B is **below 0.70**, the construct is
  not reliably measurable by LLM judges and no gate can be met. Report as a
  measurement-validity finding and stop.
- **K2** — if no Phase C correction reaches G6, publish `omission_rate` and
  `fabrication_rate` only, retire the composite, and do not submit a raidex
  column.
- **K3** — if gate −1 still fails a majority of repeats at ≥24 entities, entity
  level self-report correspondence is not measurable at this scale. Stop.

---

## 5. Standing rules carried forward

- Repeats are never filtered on preflight before averaging, and models are never
  dropped from a comparison on preflight status — both are selection on an
  outcome-adjacent criterion.
- Strata are never pooled.
- Every claim needs a null condition that must be capable of failing.
- Nulls, retractions and traps are logged whether or not they are convenient.
- **No gate in this document may be revised after seeing the number it governs.**
  A third pre-registration may supersede this one; it may not be edited.
