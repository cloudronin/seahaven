# Gate zero: cohort feasibility — VERDICT

**Deliverable of `docs/cohort-feasibility-gate.md`. One question, one verdict.**

---

## First line: where attrition landed relative to the load-bearing bin

The identifying variation for this program rests on the **20–25 MMLU-Pro bin**:
7 models across 4 families, of which 5 models and 3 families sat at 1.5B–3B.
That concentration was the collapse risk.

| | models | families |
|---|---|---|
| bin before attrition | 7 | 4 — Falcon3, Granite-3.1, Mistral, Qwen2.5 |
| **bin after attrition** | **6** | **4 — Falcon3, Granite-3.1, Mistral, Qwen2.5** |

**Small end (≤3B): 4 of 5 survived. One model lost — `tiiuae/Falcon3-3B-Base`
— and its family kept its seat through `Falcon3-3B-Instruct`.**

The bin held all four families. That is the number the verdict turns on.

# VERDICT: GO-FULL

**The program is more identifiable than its original size-ladder design
assumed.**

---

## The reframe this verdict is written under

The program never needed to explain size. It needed to **separate a behavioural
axis from capability**. Within-family size ladders hold training fixed and vary
size, which is the wrong contrast for that. **Capability-matched cross-family
sets hold capability fixed and vary training** — the discordant-case structure
Stage 4 requires.

The cohort supplies it plainly: `OLMo-2-1124-7B-Instruct` at 18.58 MMLU-Pro sits
**below** `Qwen2.5-1.5B-Instruct` at 19.99 — a 4.7× size gap at matched
capability across two labs. **The cohort's job is capability-matched contrast,
not size ladders.**

## The cohort

**30 of 38 survived** access and loop-test attrition, against a 30+ requirement.

| criterion | required | result |
|---|---|---|
| checkpoints | 30+ | **30** |
| distinct families | ≥5 | **8** |
| families with 3+ sizes | ≥2 | **4** |
| base/instruct pairs | ≥3 | **8** |

Every minimum is met. The count sits exactly on its threshold, which is worth
stating plainly rather than rounding up: there is no margin on cohort size, and
any further attrition drops the program below its own requirement.

## Attrition is a base-checkpoint story, not a size story

**Seven of the eight exclusions are base checkpoints. Every instruct checkpoint
probed passed.** All eight failed the same way — zero parseable commands.

| family | base checkpoints | outcome |
|---|---|---|
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B | all passed |
| Qwen3 | 1.7B, 4B, 8B passed; 0.6B failed | smallest only |
| Falcon3 | 1B, 3B, 7B | **all failed** |
| OLMo-2 | 1B, 7B, 13B | **all failed** |
| Llama-3.1 | 8B | **failed** |
| Mistral | 7B | passed |

This is **not** the cost finding. Check 1 established base checkpoints are
affordable at **1.9×** instruct once narration is off. Affordability and
usability are different questions, and for Falcon3, OLMo-2 and Llama-3.1 base the
answer to the second is no — most likely absent or unusable chat templates,
though that is not established here.

Cost: base/instruct pairs fell from 16 to 8. Still more than twice the minimum,
but the base gradient is now carried almost entirely by Qwen.

## The determinism dimension

Reported, **not adjudicated**: a noise floor is only disqualifying relative to a
target effect, and Stage 2 profile distances do not exist yet.

**26 of 30 fully reproducible.** All four noisy models sit at ≤3B.

| stratum | reproducible | median adherence sd |
|---|---|---|
| ≤3B | 10 / 14 | 0.000 |
| **≥7B** | **14 / 14** | 0.000 |

Noisy models' adherence sd: median 0.667, max **1.414**.

**The floor tracks size, and not output diversity.** The competing mechanism —
that a model stuck in a low-entropy attractor lands on the same token robustly,
so being worse at the task would cause more reproducibility — is not supported:

    Spearman(log size, adherence sd) = −0.597
    Spearman(entropy,  adherence sd) = +0.263      (n = 12)

Size is the mechanism to carry forward. This matters because it is the *opposite*
of the worrying case: precision is worst at the small end, but the small end is
only 4 models, and every checkpoint at 7B and above is bit-exact.

**Which effects survive the floor.** The published within-model sd was 3.09 and
the flag study's smallest decisive margin ~20 points; the largest floor here is
1.414. Any effect above ~3 points is unthreatened. Effects below ~1.5 points on a
≤3B model are not measurable without repeats, and Stage 2 must budget repeats for
those four models specifically rather than uniformly.

## Cost — derived, not directly measured

Check 1 measured **9 s per eval** (8B instruct, narration off) and **17 s**
(8B base). Job 26 logged per-model load times but not per-eval times, so
per-model sweep cost is derived from those two figures plus observed loads.

    30 evals × ~9 s + ~130 s load  ≈  6.7 min per model
    30 models                       ≈  3.3 GPU-hours  ≈  $17

With a 30% contingency for failed runs and idle GPU — tonight's realized overhead
ran well above naive estimates — **budget ~$22 for the full cohort across both
existing worlds.** The 30 evals per model already span 5 phrasings × 2 worlds ×
3 seeds.

This is roughly an order of magnitude below the program note's assumption, and
the reason is Check 1: narration was the cost.

## Two coverage gaps that travel with this verdict

**1. The Qwen3 ladder is excluded from capability-partialled analysis.** Nine
surviving models have no MMLU-Pro on the leaderboard pinned at `4d32c08` *before
any lookup* — seven of them the Qwen3 ladder, plus two OLMo-2 instruct
checkpoints. They participate in non-proxy analysis and are excluded from
anything partialled on capability.

> **Do not repair this by substituting another proxy.** The pin was set before
> lookup precisely so a proxy cannot be swapped when one turns out inconvenient,
> and it will be tempting here because Qwen3 is otherwise the cleanest ladder.
> Any later suggestion to fill those numbers from HELM or elsewhere — from
> anyone — is the pin being violated.

**2. The 20–25 bin is load-bearing for identification.** Six models, four
families, and the program's capability-matched contrast rests on it. It survived
this round with one loss. It should be treated as a single point of failure in
any future cohort change, and re-checked whenever the cohort moves.

## Which flag-study models reproduce

Documentation, **not remediation — the 360-cell corpus is not re-run.** The
published negative does not move: it rests on ρ +0.72 / +0.92, a perfectly
separating flag/pass split, and margins of 20–33 points, none of which a floor of
at most 1.414 reaches, and the cluster bootstraps already absorbed this variance.

**10 of the 12 reproduce.** The two that do not:

| model | lab | adherence sd |
|---|---|---|
| `Qwen2.5-0.5B-Instruct` | AlibabaSmall | **1.414** |
| `granite-3.1-2b-instruct` | IBMSmall | **0.897** |

Both were flagged models in the containment study, and both retain wide margins
against their floors — AlibabaSmall by 14–23×, IBMSmall by roughly 6×. Recorded
so any future claim leaning on byte-identity or a narrow margin can be checked
against whether that specific model reproduces.

## What this gate did not decide

Instrument design, the Stage 1 axis list, null derivation and capability
partialling, world authoring, and whether Together AI's 22 serverless models are
admitted as a lineage supplement — all program-spec concerns, written only now
that the verdict is GO-FULL. The flag study's pinned reserve stays untouched and
is distinct from the program's train/test split.
