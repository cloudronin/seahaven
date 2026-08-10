# Dimensional program: phased spec v0.1

**Status:** program spec. This is the full dimensional test, PHASED — phasing
makes it schedulable, it does NOT make it smaller. Builds on the GO-FULL cohort
(gate zero, e3b5d11), the smoke-test NOT-DEAD result (9377509), and
docs/program-note-dimensional-space.md. Supersedes nothing; extends the note
into an executable phase plan.

**The one non-negotiable, stated first because it is the thing most likely to be
skipped in eagerness:** the held-out set is sealed BEFORE any exploration, in the
first commit, and never touched until Phase 2. Every prior construct this project
ran died for lack of this firewall. It is not an afterthought; it is gate zero of
the program.

---

## 0. What phasing does and does not buy

- **Does:** makes the program afternoon-sized, lets you stop honestly after any
  phase, applies the one-variable-at-a-time kill discipline to axis introduction.
- **Does NOT:** make the test smaller, weaken the cohort requirement, or turn this
  into a tonight-task. Phase 1's first step is real program work.

**The wrong way to phase** (forbidden): split by cohort — explore 8 models now, 8
next month, pool at the end. This burns the held-out set one phase at a time and
destroys the only source of validity. Phasing is by COMMITMENT, not by cohort.

---

## GATE ZERO — seal the held-out set (first commit, before anything)

- From the GO-FULL cohort (30 models, 8 families, 4 ladders), partition into:
  - **EXPLORATION set** — the models Phase 1 is allowed to see and fish in.
  - **HELD-OUT set** — sealed, never loaded, never inspected, until Phase 2.
- Partition rule, frozen before the split: stratify so both sets carry family
  diversity and span the capability-matched bins (esp. the load-bearing 20-25
  MMLU-Pro bin — both sets need multi-family representation in it, or Phase 2
  can't confirm capability-matched structure). Base/instruct pairs split so
  neither set is Qwen-only on that axis (per the note's base/instruct
  Qwen-concentration limitation — carry it, don't pretend it's solved).
- The two smoke-test pairs (Gemma-9B/Llama-8B, Falcon3-10B/Mistral-7B) are
  BURNED for the failure-response axis — the junk-masking finding was discovered
  on them, so they go in the EXPLORATION set, never the held-out set. Seeing the
  effect again where it was found proves nothing.
- Commit the partition. Hash it. Phase 1 asserts the held-out list is untouched
  at every run, same pin discipline as the anchor ladder.
- **Qwen3 ladder** stays excluded from any capability-partialled analysis (pinned
  proxy gap, not to be rescued with another proxy — carry from gate zero).

---

## PHASE 1 — exploration (on the EXPLORATION set only)

Output is NOT a result. It is a frozen instrument + a frozen set of hypotheses.
Phase 1 is allowed to be messy and exploratory; the held-out firewall is what
makes that safe.

### Instrument (build once, freeze)

- State-conditioned representation per the smoke test, now generalized:
  bucket each step by a situation variable, compute command distribution per
  bucket, bend = TVD between buckets. Frozen: TVD metric, verb+object-class
  vocabulary, the three-way bend (full / legal-only / junk-contribution).
- Per-model NOISE FLOOR from the determinism map: models >=7B are bit-exact;
  <=3B carry a floor up to 1.414. Every bend read against the model's own floor.
  A bend inside a model's floor is INDETERMINATE for that model, not signal.
- CAPABILITY as a second reference: every axis read as residual after the pinned
  MMLU-Pro proxy. NOISE FLOOR as a third reference. (Per the note's Stage 4: null,
  capability, noise — three references subtracted, not one.)

### Axis-by-axis sub-phasing (the soft kill discipline inside Phase 1)

Introduce ONE axis at a time. Freeze it before the next. A dead axis is dropped,
not rescued. Order by evidence in hand:

1. **Failure-response** (start here — smoke test already found signal). Bucket:
   after-fail vs after-ok. Pre-declared read carries the smoke finding forward as
   a hypothesis to generalize, NOT a result to confirm (confirmation is Phase 2):
   does legal-verb redistribution after failure separate models across the
   exploration set, and does junk-masking (DEAD pooled / NOT-DEAD legal-only)
   generalize beyond the two burned pairs?
2. **Success-conditioning** — only if 1 held. Bucket by prior success.
3. **Novelty** — only if prior held. Bucket by room-seen-before vs fresh.
4. (further axes from the note's candidate list only if the program is alive)

Each axis: run on exploration set, read against noise floor + capability, freeze
the finding, decide go/no-go on the next axis. Each is an afternoon. The freeze
between axes is what stops this being a fishing expedition — an axis is either
frozen as a hypothesis or dropped, never left open to re-fish.

### The world question (carry the smoke-test correction)

Smoke test POOLED world_v0 and world_v2; pooling can hide a world effect as
easily as survive one, so world-invariance is UNTESTED. In Phase 1, compute each
surviving axis's bend PER WORLD as well as pooled — this is now legitimate
because it's exploration, not a second split on a confirmatory probe. A finding
that holds pooled but not per-world is world-dependent and flagged as such before
it reaches Phase 2.

### Phase 1 exit

Frozen instrument + a frozen, written hypothesis set: which axes carry signal,
which bend representation, per-world status, the specific junk-masking
generalization. This is the pre-registration Phase 2 tests against. Commit and
hash it. **You may stop here** and report "structure found, unvalidated" as an
honest preliminary. You may NOT convert Phase 1 findings into a claim without
Phase 2.

---

## PHASE 2 — confirmation (on the SEALED held-out set)

- Load the held-out models for the first time.
- Run the FROZEN instrument. Test the FROZEN hypotheses. Nothing is discovered
  here; Phase 2 only confirms or kills Phase 1's proposals.
- Pre-registered pass/fail per hypothesis, thresholds frozen in Phase 1's exit
  commit (same structure as the smoke test: bend gap vs 100-iteration self-split
  null distribution, NOT-DEAD = above 95th percentile).
- Per-axis: the axis's structure replicates on held-out models it never saw, or
  it doesn't. Report whatever fires.

### What Phase 2 licenses

- **Replicates:** the dimensional structure is real — validated on held-out
  models, which is the claim the whole program exists to earn. THIS is a result.
- **Fails:** the Phase 1 structure was exploration-set-specific / noise / cohort
  artifact. Killed cheaply on held-out data before any downstream build. Also a
  result, and an honest one.
- **Partial:** some axes replicate, some don't. The replicating axes are the real
  dimensional structure; the rest were fishing. Report the split.

---

## Kill criteria (frozen now)

| ID | Criterion | Consequence |
|---|---|---|
| KP-1 | Failure-response axis: junk-masking does NOT generalize beyond the two burned pairs on the exploration set | The smoke finding was pair-specific; drop it, try the next axis |
| KP-2 | No axis clears its noise floor on the exploration set after 3 axes tried | Representation is flat at this cohort; program stops, "no dimensional structure found" is the finding |
| KP-3 | Phase 1 structure fails to replicate on held-out (Phase 2) | Structure was exploration-set artifact; killed on held-out, honest negative |
| KP-4 | Every surviving axis is explained by capability or noise floor after partialling | The "structure" is capability/noise in disguise — the flag's death, avoided by partialling; report as such |

---

## What this is not

- Not smaller than the full test — phased, not shrunk.
- Not a tonight-task. Gate zero (sealing the held-out set) is the first real
  program-spec commit and deserves a rested start per the note's own instruction.
- Not licensed to fish: axis-by-axis freezes and the held-out firewall are what
  separate exploration from a fishing expedition. Remove either and it becomes
  the fifth pretty death with the most machinery yet.
- Not a cohort-expansion loop: the cohort is fixed at gate zero. Adding models
  mid-program re-opens the held-out question and is forbidden without a new seal.

## Cost

Phase 1 exploration is ~$22-scale sweeps on the exploration set (narration off,
per the cost collapse finding), afternoon-sized per axis. Phase 2 is one sweep of
the held-out set. Total is program-scale but spread across sessions, and stoppable
at any phase boundary with an honest partial result.
