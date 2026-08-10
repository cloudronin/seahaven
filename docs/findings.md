# seahaven-fidelity: what the validation battery established

A synthesis of the five-phase battery. `docs/research-log.md` is the primary
append-only record with every null and retraction; this document states what
survived, in the form a reader should take away.

The intended output was a per-model "self-report fidelity" score, proposed as a
ninth raidex constituent. **That output is not produced, and the reason is the
finding** (§1).

A second measurement then emerged from the same corpus, **Constraint
Adherence** (§5), which asks whether an agent stays inside the action space its
own prompt declares. It is detector-free, world-stable, and robust to its own
measurement decisions.

**[CORRECTED 2026-08-09] It then died on the same axis fidelity did.** Varying
how the constraint is *declared* — five phrasings, 210 evals — puts worst-pair
Spearman at **0.631**, against fidelity's 0.607 on the narration register. Both
constructs entered the same battery and both failed it, from different
directions: one on how the question is asked, the other on how the rule is
stated. That replication is now the finding (§7), and it is stronger than either
death alone, because adherence had survived everything else first.

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

### What was established, and what was wrongly inferred from it

**[CORRECTED 2026-08-09.] Every number in this table stands as measured. The
inference drawn from it does not.**

| property | fidelity | adherence |
|---|---|---|
| stable across worlds | ρ 0.893, bootstrap lower bound 0.357 | **ρ 0.864** |
| stable across narration registers | ρ **0.607** | ρ 1.000 (test–retest; register does not touch the action phase) |
| stable under detector choice | **does not survive** | **no detector in the path** |
| stable under its own measurement decisions | — | **ρ 0.893** across four filter definitions |
| **stable across constraint phrasings** | — | **ρ 0.631 — FAILS** |

The section previously headed "why it survives what fidelity did not" and
claimed the last row was "the check fidelity failed". **That was the error.**
Cross-world 0.864 and filter-definition 0.893 are correct and reproduce; they
establish **measurement-side** robustness. But fidelity did not die on the
measurement side. It died because varying the *stimulus* — the narration
register — reordered the table at ρ 0.607. **Adherence had never been given that
test**, and the analogous one for a constraint measure is varying how the
constraint is declared, not how the violation is counted.

Tested, it fails at **0.631**. The claims were never wrong. They were
incomplete, and the missing check was the decisive one.

Deciding what counts as a "verb" is still a real judgement — IBM's most frequent
out-of-vocabulary token is `1.` and Meta's is `inventory.` — and varying that
definition four ways still moves *levels* (Meta 85.4 → 97.9) and not *ordering*.
That result holds. It simply never bore on the question it was cited for.

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

---

## 7. The replication: two constructs, one failure mode

This is the result the project actually produced, and it took two independent
constructs to establish.

| | fidelity | adherence |
|---|---|---|
| what it measures | does the account match what was done | does the agent stay in its declared action space |
| detector in the path | yes | **no** |
| survived world change | marginal (0.893, bootstrap lower bound 0.357) | **yes** (0.864) |
| survived measurement-decision change | **no** (detector swap reorders entirely) | **yes** (0.893 across filters) |
| **died on stimulus change** | **narration register, ρ 0.607** | **constraint phrasing, ρ 0.631** |

**Two constructs, built differently, one detector-dependent and one
detector-free, entered the same battery and died at the same number on the same
axis** — the wording of what the model is asked or told, held apart from
everything about how it is scored.

That is not two failures. It is a replicated finding about a class: **for
agentic self-report and constraint measures at this scale, the model ordering is
a joint property of the model and the prompt that elicits it, and the elicitation
half is not small.** Adherence is the stronger of the two data points precisely
because it survived everything else first — detector-free, world-stable,
filter-stable — so its death isolates the stimulus as the cause rather than
leaving it confounded with measurement choices.

### What the battery cost, and what it saved

G-P ran for **~$10** and fired before world_v3 was authored, before the Stage 1
sweep, and before Stage 2. The full publication battery was budgeted at
**$115–170**. The gate ordering — cheapest and most lethal first — was chosen
before any of it ran, and it worked exactly as designed.

The one $0 gate, C-RAND, would have caught a broken classifier before any of it
mattered. It passed at exactly 100.00 on both worlds. Ordering gates by cost and
lethality is the transferable practice, not a detail of this project.

### The one model-level fact that survived

**IBM and AI2 are the bottom two under all five phrasings, without exception.**
Their violations are dominated by template noise and capability-seeking attempts
rather than by constraint interpretation. The churn is entirely mid-table.

That licenses a **threshold-membership** claim — *these models fall below X
under every declaration tried* — and **never a rank ordering**. It is weaker than
a leaderboard column, and it is the only model-level claim the data supports.
Per the pre-registered FAIL branch it belongs to a new spec, not to this one.

### Instruction form is itself a result

**P4 — a bare `Valid verbs:` list with no rationale sentence — appears in four of
the five sub-gate pairwise correlations.** Removing the justifying sentence does
not merely lower adherence; it *reorders which models look better than which*.

And removing the restriction sentence entirely (P5) costs **AI2 7.24 points and
TII 2.95**, while four models sitting at ceiling move by less than 1.5. The
pooled gap is −1.99, which falsifies the pre-registered VP-3 threshold by 0.01
— **the letter is logged as falsified and the prediction is not rewritten** —
but the pooled framing could not see that the effect is real and concentrated in
the models with room to fall.
