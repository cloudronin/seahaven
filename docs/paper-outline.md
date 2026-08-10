# Paper outline — how agentic benchmarks die, and a battery that kills them cheaply

**Status:** outline. Every number below is measured and in `results/`; nothing
here needs a new run. Target TMLR; expanded version NeurIPS D&B.

**The contribution is the methodology, and it now has two corpses as evidence.**
Not "here is a benchmark" — the two benchmarks both failed. The paper is the
battery that killed them, the order it kills in, and what the two deaths have in
common.

---

## The claim

> For agentic self-report and constraint measures at this scale, the model
> ordering is a joint property of the model and the prompt that elicits it, and
> the elicitation half is not small. We show this twice, on two constructs built
> differently, and give the validation battery that detects it for ~1/15th of the
> cost of the study it invalidates.

Two independent constructs entered the same battery and died on the same axis at
nearly the same number:

| | fidelity | adherence |
|---|---|---|
| detector in the measurement path | yes | **no** |
| survived world change | marginal, ρ 0.893 (bootstrap ⊥ 0.357) | **yes**, ρ 0.864 |
| survived measurement-decision change | **no** — detector swap reorders entirely | **yes**, ρ 0.893 across filters |
| **died on stimulus change** | narration register, **ρ 0.607** | constraint phrasing, **ρ 0.631** |

Adherence is the stronger data point: it survived everything else first, so its
death isolates the stimulus rather than leaving it confounded with measurement
choices.

---

## Evidence in hand

| asset | scale |
|---|---|
| episodes | **3,515** across 3 worlds, 7 labs, 5 phrasings, 3 narration registers |
| commands | **57,822** |
| judge panel | 7 LLM judges from 7 labs, 6 usable, split-half reliability computed |
| adjudicated labels | 2,364 items, dual-asked, 97–98% self-consistent |
| documented traps | **30**, numbered, each with the null that failed |
| research log | 5,075 lines, append-only, nulls logged as prominently as passes |
| tests | 222, including sha256 pins on every frozen constant |

---

## Structure

### 1. Introduction
Agentic benchmarks are proliferating. Validation practice has not kept pace: a
single detector, a single prompt wording, no reliability estimate. We ran the
battery twice and it killed both constructs.

### 2. Two constructs
**Self-Report Fidelity** — does an agent's unprompted account of itself match a
ground-truth record. **Constraint Adherence** — does an agent stay inside the
action vocabulary its own prompt declares. Deliberately different: one needs a
detector, one is read directly from issued commands.

### 3. The battery
Ordered by cost and lethality, cheapest and most lethal first. **This ordering is
the transferable contribution.**

| gate | asks | cost | outcome |
|---|---|---|---|
| C-RAND | does a scripted-legal policy score 100? | **$0** | PASS, exactly 100.00 |
| determinism control | is the serving path reproducible? | ~$2 | **flag-required** |
| V-P / register | does the ordering survive rewording the stimulus? | ~$10 | **FAIL, both constructs** |
| cross-world | does it survive a new world? | ~$6 | marginal / pass |
| detector swap | does it survive a defensible alternative instrument? | ~$3 | **FAIL for fidelity** |
| confound controls | is it verbosity or exploration? | $0 | **FAIL for fidelity** |

Budgeted full programme: **$115–170**. Killed for **~$12**.

### 4. Result 1 — fidelity dies four ways
Detector swap reorders the table entirely (21.1% of 9,253 judgements change);
narration register ρ 0.607; verbosity confound is between-model and sign-varying;
`r(n_performed, fidelity) = −0.44` survives quota-sampling, so it is behavioural
rather than arithmetic. Kill criterion K2 fired; composite retired.

### 5. Result 2 — adherence dies once, on the axis that matters
Detector-free, world-stable (0.864), filter-stable (0.893). Constraint phrasing
ρ **0.631**. P4 (a bare `Valid verbs:` list, no rationale sentence) is in four of
the five sub-gate pairs: **removing the justifying sentence reorders models, not
merely lowers them.** Removing the restriction entirely costs AI2 7.24 points and
TII 2.95, while four ceiling-bound models move <1.5 — a pooled −1.99 that
falsifies the pre-registered threshold by 0.01 while concealing a real,
concentrated effect.

### 6. What survives both deaths
- **Threshold membership, never rank.** IBM and AI2 are bottom-two under all
  five phrasings, without exception.
- **Self-report completeness declines with activity volume** (−0.44), and
  quota-sampling a fixed entity count does not remove it — behavioural, not a
  counting artefact.
- **String detectors are structurally blind** where they disagree: κ = 0.000,
  because one is a strict subset of the other and each is a constant there.
- **An LLM judge panel is far more reliable in aggregate than pairwise** —
  split-half 0.795 vs mean pairwise 0.675. Practice is one judge and no
  reliability estimate.
- **Negation prevalence is a property of the question**: 0.005 introspective vs
  0.049 factual, a 10.7× swing, on the exact case every detector fails.

### 7. Methods that generalise
From 30 traps, the ones that transfer:
- Validate the composition, not the components (178 tests green while a shadowed
  loop variable broke every run after the first).
- A test supplying its own fixtures cannot catch a mismatch between two things
  it supplies (two dead entity columns across 499 published runs).
- Mean of ratios ≠ ratio of sums (moved a correlation 0.964 → 0.679).
- **Aggregate reliability ≠ pairwise reliability** (understated a ceiling by
  0.120 and nearly justified lowering a threshold the evidence supported).
- **A control that passes at small n is evidence about the sample, not the
  system** (n=1 said drift, n=2 confirmed it, n=4 showed neither).
- An oracle over disagreeing binary detectors is vacuous.

### 8. Limitations
7 checkpoints, one size each — no scale series, so capability is not partialled.
Three worlds, two of them near-identical in size. LLM judges from 6 labs but no
human labels. Text worlds, not tool-use or code, so substrate generality is
untested.

### 9. Related work
IFEval and single-turn instruction following; LLM-as-judge reliability;
benchmark validity and construct validity in NLP; agentic evaluation harnesses.
**The gap: agentic benchmarks are published with less validation evidence than
this paper's rejected constructs had.**

---

## Artefacts released
`seahaven/fidelity/` (measurement modules, sha256-pinned constants) ·
3 worlds with build scripts and lock files · 3,515 episodes with per-step
records · 2,364 adjudicated labels + 7 judge runs · the full research log
including every retraction · 222 tests.

---

## What this outline does not license

**No per-model leaderboard, for either construct.** Both orderings are
apparatus-dependent and the paper says so. The single model-level claim
permitted is threshold membership under all phrasings tried.

**No salvage-by-restratification inside this paper.** A phrasing-stratified
adherence claim may be viable, but per the pre-registered FAIL branch it is a new
spec with its own blind predictions — not a reanalysis of data whose answer is
now known.
