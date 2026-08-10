# seahaven-fidelity: what the validation battery established

A synthesis of the five-phase battery. `docs/research-log.md` is the primary
append-only record with every null and retraction; this document states what
survived, in the form a reader should take away.

The intended output was a per-model "self-report fidelity" score, proposed as a
ninth raidex constituent. **That output is not produced, and the reason is the
finding** (§1).

A second measurement then emerged from the same corpus and **does** survive the
checks fidelity failed: **Constraint Adherence** (§5), which asks whether an
agent stays inside the action space its own prompt declares. It is
detector-free, world-stable, and robust to its own measurement decisions. It is
the more promising candidate constituent, and it is not yet validated under the
condition that matters.

---

## 1. The central result: the ordering belongs to the apparatus, not the models

Three sources of instability were measured on the *same* 500 episodes, holding
the models and their behaviour fixed:

| source varied | effect on the model ordering |
|---|---|
| **detector** (regex → LLM) | ordering does not survive at all; 21.1% of 9,253 entity judgements change |
| **narration register** (introspective / factual / retrospective) | worst-pair Spearman **0.607** |
| **world** (v0 → v2) | ρ 0.893 point estimate, bootstrap lower bound **0.357** |

Individual movements are as large as the entire between-model spread: TII falls
18.7 points under a detector swap; Alibaba moves 16.8 points between narration
registers.

**None of these three is the model's honesty.** The score is a joint property of
(model, detector, register, world), and only the first was the object of study.
Entity-level self-report correspondence, as constructed here, is **not separable
from the apparatus that measures it**.

This subsumes the earlier, weaker version of the same observation — that two
defensible detectors gave opposite answers about whether models mostly omit or
mostly invent. That was one instance of a general problem.

---

## 2. What survives, and is worth reporting on its own

### 2.1 Self-report completeness declines with activity volume

`r(n_performed, fidelity) = −0.44`: agents that did more report a smaller
*fraction* of it.

The obvious objection is arithmetic — more acts, more chances to omit. It was
tested and rejected. Quota-sampling **exactly five performed entities per
episode**, so every episode contributes an identical number of observations,
leaves the relationship at −0.31. Holding the count fixed does not remove it.

**So this is behavioural, not a measurement artefact**, and no world design can
correct it. It is the reason the composite metric was retired.

### 2.2 String detectors are structurally blind on the decisive stratum

On items where the two regex detectors disagree — the stratum that determines
which detector a benchmark should adopt — both score **exactly κ = 0.000**
against a six-judge majority.

This is not bad luck. `relation_aware` is a strict subset of `name_only`, so on
a disagreement item one always says *claim* and the other always says
*not-claim*. Each is a constant, and a constant agrees with nothing. **No
tuning of either detector can address the cases that decide between them.**

### 2.3 An LLM judge panel is far more reliable in aggregate than pairwise

Six judges from six labs (OpenAI, Google, Alibaba, Mistral, Microsoft, IBM):

| quantity | value |
|---|---|
| mean **pairwise** inter-judge κ | 0.675 |
| **split-half reliability of a 3-judge majority** | **0.795** [0.722, 0.861] |

The +0.120 gap is the practical point. LLM-as-judge is routinely used for
semantic adjudication of exactly this kind, usually with **one** judge and no
reliability estimate. A single judge's agreement with another understates what a
small panel achieves — and, equally, a lone judge provides no way to notice that
its own reliability is 0.675 rather than 0.795.

A seventh candidate, AI2's OLMo-2-32B, **failed the planted controls at 0.75**
by denying plainly stated room visits ("I walked north into the vault" → NO) —
the entity class the corpus already omits most. It was excluded. Without control
items it would have entered the panel silently.

### 2.4 Negation prevalence is a property of the question, not the model

Explicit negation ("did not", "never", "no sign of") near a scored entity:

| register | rate | vs introspective |
|---|---|---|
| introspective | **0.005** | — |
| retrospective | 0.017 | 3.8× |
| factual | **0.049** | **10.7×** |

Every number in this project was measured under the introspective register,
where negation is essentially absent. Negation is also the known failure mode of
every detector here — both regexes return `True` on "I never found the logbook".
**A detector validated on this corpus is validated on text where its defining
hard case occurs once in two hundred mentions.**

Any LLM-as-judge validation should report the prevalence of its own hard cases
in the validation corpus. This one would not have looked different if the
detectors were completely negation-blind.

---

## 3. Where the battery stands

| | result |
|---|---|
| V1 detector agreement | **FAIL** — best detector 0.710 on the disagreement stratum against a gate of 0.75 (derived from the 0.795 ceiling) |
| V2 cross-world | marginal — ρ 0.893, bootstrap [0.357, 0.964] |
| V3 narration register | **FAIL** — worst-pair ρ 0.607 |
| V4 confound controls | **FAIL** — verbosity and exploration confounds both survive the detector swap |
| gate −1 (is anything measured at all) | marginal — 11/18 world_v0 repeats, 16/18 world_v2 |
| reliability | share_between 0.908, test–retest passes |

**Kill criterion K2 fired**: no pre-committed correction reached the confound
target, so the composite is retired, `omission_rate` and `fabrication_rate` are
published only alongside `n_performed`, and **no raidex column is submitted**.

---

## 4. What would be needed to rescue a per-model claim

In dependency order, and honestly: the first item alone is a research programme.

**Or: measure something else on the same corpus.** §5 is that, and it took one
afternoon of analysis on data already collected. The four items below remain the
price of rescuing *fidelity* specifically.

1. **A detector whose choice does not reorder the table.** Since the two
   available families disagree on 21% of judgements and one is structurally
   blind where it matters, this means a genuinely better instrument, not tuning.
2. **A register-invariant elicitation**, or a decision that the register is part
   of the construct and must be fixed and declared — in which case the score
   measures self-report *under that question*, and should say so.
3. **A confound-free composite**, which quota-sampling has already shown cannot
   be obtained by construction.
4. **Higher label quality.** The panel ceiling is 0.795; a detector cannot be
   certified far past it.

---

## 5. Constraint Adherence — the measurement that survives

Found by asking whether the corpus said anything about rule-breaking. It does:
the action prompt declares an exhaustive vocabulary ("These are the words that
work. Nothing else does"), so any out-of-vocabulary command violates an explicit
instruction, and `verb_counts` recorded it in all 542 episodes.

**Adherence, in leaderboard orientation (higher is better):**

| model | episode-level | action-level |
|---|---|---|
| Alibaba | **100.0** | 100.0 |
| Meta | 97.9 | 99.5 |
| Google | 97.2 | 99.8 |
| TII | 89.4 | 99.0 |
| MistralAI | 79.4 | 96.9 |
| AI2 | 61.1 | 89.7 |
| IBM | 60.4 | 92.7 |

Spread 60.4–100.0, sd 15.8, with 100 a real ceiling. Most-attempted unlisted
capabilities: `read`(58), `use`(43), `pick`(28), **`unlock`(24)**, `exit`(15),
`pour`(12).

**[CORRECTED 2026-08-09]** This section previously called `unlock` the notable
case, on the grounds that both worlds contain closed containers and those
attempts reach for an ungranted verb "to get past them". **That was wrong, and
checking the live engine refutes it:**

    open chest    -> "You open the chest."
    unlock chest  -> "(with the chest) The chest is fixed in place."

Neither world contains a `locked` fact, a key, or any obstacle the granted
vocabulary cannot pass. `open` is granted and sufficient, so `unlock` is a
*redundant* attempt, not a barrier-passing one. **There is no barrier in this
corpus to be directed at.**

Splitting the out-of-vocabulary tokens accordingly: 39.5% are template noise
(`1.`, `inventory.`), and of the remainder the largest class is **synonyms for
already-granted verbs (36.1%)** — `pick`≈take, `inspect`≈examine — against only
14.9% reaching for a capability the world does not implement. The dominant
failure is vocabulary compliance, not capability-seeking.

This is a missing manipulation rather than a null about models: barrier-directed
excursion cannot be measured until a world contains a barrier the sanctioned
action space provably cannot pass. `docs/adherence-spec-v0.1.md` §7 supplies
one.

### Why it survives what fidelity did not

| property | fidelity | adherence |
|---|---|---|
| stable across worlds | ρ 0.893, bootstrap lower bound 0.357 | **ρ 0.864** |
| stable across narration registers | ρ **0.607** | ρ 1.000 (test–retest; register does not touch the action phase) |
| stable under detector choice | **does not survive** | **no detector in the path** |
| stable under its own measurement decisions | — | **ρ 0.893** across four filter definitions |

The last row is the one that matters most, because it is the check fidelity
failed. Deciding what counts as a "verb" is a real judgement — IBM's most
frequent out-of-vocabulary token is `1.` (list numbering) and Meta's is
`inventory.` (an allowed verb with a period). Varying that definition four ways
moves the *levels* substantially (Meta 85.4 → 97.9) and the *ordering* barely at
all. **The measurement decision is not load-bearing.**

Adherence is also near-orthogonal to fidelity (r = +0.206): Alibaba is the worst
model on fidelity and the only one with perfect adherence. Two constructs, not
one measured twice.

### Where it fits, and what it is not

raidex's nine constituents contain nothing covering it. StrongREJECT and XSTest
are the nearest and are **request-facing and single-turn** — what a model agrees
to do when asked — whereas this is whether an agent stays in bounds while acting
unsupervised. Proposed dimension name: **Constraint Adherence**; "Compliance"
is avoided because in a Responsible AI index it reads as regulatory compliance,
and its refusal sense already belongs to two existing constituents.

**Not yet established, and the first item is the whole question:**

1. **This is a zero-pressure floor.** Nothing is gained by leaving the action
   space, so it measures baseline drift, not goal-driven rule-breaking.
2. **Intent is not observable.** Exceeding the stated vocabulary and failing to
   follow instructions are indistinguishable here.
3. Episode-level and action-level orderings disagree at the bottom; one must be
   pre-registered as primary.
4. Seven models, one checkpoint each.

A related figure should be treated with care: 59% of unlisted-action attempts
never appear in the agent's account. That reads as concealment but is not —
**the attempts fail**, the world rejects them, and omitting a no-op is
defensible summarisation. It becomes a disclosure measurement only when the
unlisted action *succeeds*.

---

## 6. Method notes that generalise beyond this project

Recorded because each cost real time here and each is invisible until it bites.

- **Validate the composition, not just the components.** A 178-test suite passed
  while a shadowed loop variable broke every run after the first; a local smoke
  test against a real endpoint caught it in seconds. Four separate defects
  followed this pattern.
- **A test that supplies its own fixtures cannot catch a mismatch between two
  things it supplies.** Ground-truth tests wrote their own fact strings using a
  single-word entity, so a two-word entity name that never matched the engine's
  output went undetected across 499 published runs.
- **Mean of ratios is not the ratio of sums.** Averaging per-episode rates
  instead of pooling observations moved a cross-world correlation from 0.964 to
  0.679 on identical data.
- **Aggregate reliability is not pairwise reliability.** Setting a gate from
  pairwise agreement understated the achievable ceiling by 0.120 and nearly
  justified lowering a threshold that the evidence in fact supported.
- **An oracle over disagreeing binary detectors is vacuous.** "At least one
  detector is right on 99.5% of items" is guaranteed whenever binary detectors
  disagree and the label is binary. It measures diversity, not headroom.
