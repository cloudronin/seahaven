# Pre-registration v3 — supersedes v2, does not amend it

**Status:** governs all work from this point. `docs/prereg-v2.md` and the
original pre-registration in `docs/research-log.md` **stand unaltered**. v2 is
superseded within hours of being written because its central justification was
computed from the wrong quantity; the error and its correction are both part of
the record.

---

## 1. The correction: v2's ceiling was the wrong statistic

v2 set the gate at 0.70 on the grounds that three judges agree with each other
at mean pairwise kappa **0.736**, so a detector cannot be held to 0.80.

**That argument does not hold.** Pairwise kappa measures agreement between
*individual* judges. The reference standard is not an individual judge — it is
the **majority of three**. Majority voting averages out judge-specific
independent error, so the aggregate is more reliable than any member of it, and
the ceiling on *agreement-with-the-majority* sits **above** 0.736, not at it.

v2 therefore compared the detector's target against a number that understates
the achievable ceiling, and the whole defence of lowering the gate rests on that
comparison. **0.70 is not yet established as principled. It is provisional
until the correct quantity is computed.**

The instruments v2 introduced to keep itself honest — the §0 enumeration of
already-known quantities, and the "a gate lowered until the incumbent passes is
worth nothing" clause — were necessary and are carried forward. They do not
substitute for the ceiling being the right number.

## 2. The right quantity, and how it is computed

The standard is a **majority of three judges**. The reliability of that object
is its **split-half agreement**: partition six judges into two disjoint triples,
take the majority within each, and compute kappa between the two majorities.

    ceiling = kappa( majority(J1,J2,J3), majority(J4,J5,J6) )

This is the reliability of exactly the aggregate in use, not of a single rater
and not of a hypothetically longer panel, so no Spearman–Brown extrapolation is
required or permitted. Judges are assigned to triples by a fixed rule stated
before the split — sorted by model name, alternating — so the partition cannot
be chosen after seeing which split is favourable. All partitions are reported;
the pre-declared one is primary.

Three judges exist (OpenAI, Alibaba, Google). Phase B adds three more from three
further labs to make the split computable.

## 3. The gate is decided by a mechanical rule, fixed before the number exists

No discretion is retained. Once `ceiling` is computed:

    gate = largest multiple of 0.05 strictly below `ceiling`,
           clamped to [0.60, 0.80]

| ceiling | gate | consequence |
|---|---|---|
| ≥ 0.85 | **0.80** | v2's lowering was **not** justified. The gate returns to 0.80, V1 fails, and v2's §1 stands as an error of reasoning that this document corrects. |
| 0.80 – 0.85 | 0.75 | partial lowering justified; V1 re-tested at 0.75 |
| 0.72 – 0.80 | 0.70 | v2's number was right for the wrong reason; the argument is repaired here |
| 0.65 – 0.72 | 0.60–0.65 | gate falls further; V1 status re-decided against it |
| < 0.65 | — | **K1 fires** (§6) |

**The 0.70 currently on the books has no standing until this rule is applied.**
Any V1 result reported before then must carry the words "provisional gate".

## 4. Predictions carried forward, with G2 sharpened

G1, G3, G4, G5, G6 stand exactly as written in v2 §2. G2 is replaced, because
as written it could be satisfied by a detector that is merely noisier.

### G2 (revised) — the confound must fall *selectively*

> **G2a.** Under D1, model-level `r(narrative_words, omission)` remains more
> negative than −0.35.
>
> **G2b — the discriminator.** If G2a is falsified (the confound does fall),
> D1 is credited with fixing it **only if the fall is selective**: between-model
> sd of fidelity must not drop by more than 20% of its regex value, and
> cross-world Spearman must not fall below 0.80.

A detector that adds noise shrinks *every* correlation in the table, including
the ones that ought to be stable. Uniform attenuation across the length slope,
the between-model separation and the cross-world correlation is evidence that D1
degraded the signal and merely looked like it removed a confound. Only a flat
length slope **alongside** preserved separation counts as a fix.

## 5. Phase order corrected

v2 ran A before B and C after D. Both were wrong.

- **A and B are independent and run concurrently.** B does not consume A's
  output. B settles whether the gate D1 passed is legitimate, which is what
  A's numbers *mean* — so B gates the interpretation of A, not its execution.
- **C precedes D.** Phase C may adopt "hold `n_performed` fixed by
  construction", which is a constraint on world design; Phase D authors
  world_v3 with an entity-count requirement. Authoring D before C resolves would
  fix a world design that C might then invalidate.

| phase | work | gate | cost | order |
|---|---|---|---|---|
| **A** | Re-score the 500 existing runs under D1; recompute Phase 1, V2, V4 | G1, G2a/G2b, G3 | ~$3 API | now, concurrent with B |
| **B** | +3 judges (6 total, 6 labs); compute the split-half ceiling; apply §3 | §3 rule, K1 | ~$8 GPU | now, concurrent with A |
| **C** | Fix V4 — retire the composite, or hold `n_performed` fixed by construction | G6 | free | after A |
| **D** | world_v3, ≥24 entities, design constrained by C's outcome | G4 | ~$8 GPU | after C |
| **E** | V3 narration sweep, 3 registers | G5 | ~$8 GPU | after C |
| **F** | raidex integration — `estimate_cost` must model detector API cost | — | ~2 days | last |

## 6. Kill criteria, and what they are worth

- **K1** — split-half ceiling below 0.65. The construct is not reliably
  measurable by an LLM panel at all.
- **K2** — no Phase C correction reaches G6. Publish `omission_rate` and
  `fabrication_rate` only; retire the composite; no raidex column.
- **K3** — gate −1 still fails a majority of repeats at ≥24 scored entities.

**K1 and K3 firing are publishable methodological results, not the programme
dying.** If a six-judge panel spanning six labs cannot establish a reliable
reference standard for whether a sentence claims an entity, that is a finding
about the measurement practices the field currently relies on — LLM-as-judge is
used routinely for exactly this kind of semantic adjudication, usually with one
judge and no reliability estimate at all. The same holds for K3: a demonstration
that entity-level self-report correspondence is not measurable at feasible n is
worth writing up, and is more useful than a weak positive.

A null here is a result about instruments, and it should be written as one.

## 7. Standing rules carried forward

Unchanged from v2 §5: repeats are never filtered on preflight; models are never
dropped on preflight status; strata are never pooled; every claim needs a null
that can fail; nulls and traps are logged whether convenient or not.

**No gate in this document may be revised after seeing the number it governs.**
§3 exists precisely so that the gate is decided by arithmetic rather than by
judgement once the ceiling is known. A fourth pre-registration may supersede
this one; it may not be edited.
