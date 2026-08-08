# Seahaven — implementation plan, revision 3

Supersedes revision 2. Every number below is measured on this stack unless
marked as an estimate. Full evidence in `research-log.md`; result artifacts in
`results/`.

## Revision 3 in one page

Revision 2 said the central risk was whether self-authored narratives diverge.
Two cross-lab sweeps answered it and moved the project somewhere else.

| rev 2 held | now measured |
|---|---|
| the "quiet observer" is what self-narration converges to | **Qwen-specific** — zero core vocabulary shared with any of six other labs |
| narrative convergence is the gate | **the statistic is too noisy to rank models** (noise 0.147 > signal 0.128) |
| character might emerge from lived experience | **it is inherited** — checkpoints arrive with stable behavioural personalities (between-lab 1.49–1.86 vs within-lab 0.024) |
| story→behaviour needed assigned characters (2.44) | **self-authored narratives predict the next campaign's behaviour**, partial r **+0.412** controlling for persistence |

**The project has a positive result, and it is not the one the spec proposed.**
Not "character emerges from experience" — that measured out as no. Rather: *the
self-account an agent writes steers what it does next*, and *checkpoints differ
in stable, measurable ways before any of the machinery runs*.

Two candidate directions, both grounded in measurements that replicated:

1. **Narrative steering** — replicate the lagged coupling with more labs, more
   behavioural axes, and a pre-registered vocabulary. Currently marginal on the
   conservative test (6/7 labs, p ≈ 0.06).
2. **Self-report fidelity** — the calibrated instrument. Models inflate agency
   **5.2×** and report perception at **1.2×**, with Gemma-2 alone at 1.0. No
   published concealment work has this innocent baseline, and without it
   concealment numbers cannot be separated from ordinary narrative compression.
   Specified in [`world_v1_spec.md`](world_v1_spec.md).

**Do not build world_v1 as a general agentic-safety benchmark.** ODCV-Bench
(arXiv 2512.20798) already covers goal-pressure-versus-constraint with 40
scenarios and 21 models, including the two-framing design; MACHIAVELLI covers
text games. The prohibition survives in the design only as a **valence probe** —
the one act a model has a motive to omit — which is what separates concealment
from a heroic narrative register.

## What changed

Revision 1 planned an experiment. Phase A ran it in miniature, and three of its
assumptions did not survive:

| assumption in rev 1 | measured |
|---|---|
| the gate is "does LoRA move behaviour" | **it does** (effect 0.065, floor exactly 0). Not the risk. |
| story→behaviour works well enough to build on | **1.07 — essentially nothing**, until framing fixed it to 2.44 |
| distillation makes character durable | **it contracts, never amplifies**, in every configuration tried |
| assigned-vs-emergent story is a §10 secondary | **load-bearing.** Without it the story is inert |

The central risk has moved. It is no longer *can the machinery work* — it can.
It is **whether self-authored narratives diverge at all**, and the current answer
is: barely, and less than assigned ones.

---

## What Phase A established

| spike | result |
|---|---|
| A5 world artifacts | ✅ facts/entities reach `infos`; obs stays clean; no-quest builds compile |
| A4 base checkpoints | ✅ viable — Qwen3-4B-Base 47/50 zero-shot, 48/50 few-shot. **K3 = 0.03** |
| A2 multi-LoRA + structured output | ✅ compose in one batch |
| A3 determinism | ✅ with `VLLM_BATCH_INVARIANT=1` (1 vs 4 distinct outputs across 16). **Costs 3.3×** |
| vllm#42125 canary | ⚠️ **LIVE** — reused adapter name serves stale KV |
| A1a training path | ✅ |
| A1b gate | ✅ **PASS** — effect 0.065, test-retest floor exactly 0.0 |
| action variance | ✅ at 8B (20/20 distinct sequences); ✗ at 4B (3/20) |

Measured cost: **135 GPU-minutes ≈ $11.25** for all of it.

---

## The finding that reorders the plan

The spec's causal chain, with each link now measured:

```
self-authored story  →  behaviour  →  distilled into weights  →  durable divergent character
      ✗ converges        ✓ 2.44 with        ✗ contracts             ✗ not demonstrated
                          framing            (−0.009 to −0.044)
```

**Link 1 is the break.** Given the same world, different seeds author
*the same character in different words* — "a quiet observer", "a seeker of
fragments", "a wanderer" — patient, methodical, drawn to what was left behind.
Three independent sightings, including from an identical amnesiac seed story
where the starting point was held constant.

Links 2 and 3 follow mechanically. Convergent stories produce convergent
behaviour, which produces convergent corpora, which distillation faithfully
amplifies. **Distillation has been taking the blame and is downstream of the real
problem.** A KL term anchoring each run to its own past made it slightly worse
(retention 0.885 vs 0.925), which rules out tail collapse as the mechanism.

**Consequence for the claim.** As specified, the experiment would likely measure
a small across-seed distance against a comparable no-story floor and report a
null — not because the rig failed, but because there is no divergence to find at
this scale in this world. That is a reportable result, but it is worth knowing
*before* paying for Phase F.

---

## Revised design decisions

Carried forward from rev 1 unchanged: ledger as an in-world object retrieved at
cost, offline probe reconstruction, exact option scoring, unbiased estimators,
4×2 forced-prefix families mirrored across arms, degeneracy monitor, the
scrambled-story construction, and the frozen probe reference state.

**Changed or added:**

| | decision |
|---|---|
| **Model floor** | **≥8B.** 4B cannot express the effect — sampling noise reaches what the agent *says*, not what it *does* (3/20 distinct sequences, 316/320 identical commands) |
| **Identity framing** | **Core, not secondary.** One sentence — *"you do not reason it out; you read what you wrote about yourself, and you act like that person"* — takes behavioural separation from 1.13 to **2.44**. Without it the story is inert. Label results as *induced*, not emergent |
| **Adapter naming** | Versioned per campaign, never reused. vllm#42125 confirmed live: a reused name serves KV from the old weights, silently, biased toward fabricating within-run stability — exactly K2's signal |
| **`VLLM_BATCH_INVARIANT=1`** | Correctness requirement, not tuning. Costs **3.3×** (not the 1.5–2× assumed) — every wall-clock and dollar estimate roughly doubles |
| **Prompt masking** | Mandatory. Unmasked put 68% of gradient on text identical across runs; masking halved the contraction |
| **Selecting nothing** | Means *no update this campaign*, not run termination. Agents kept zero in 3/12 run-campaigns; treating that as fatal cut a sample from 6 to 3 |
| **Distillation** | Reframed. It is a **contraction operator** whose strength tracks corpus similarity. Design around preservation, not amplification |
| **Campaign count** | **Corrected by the 8-campaign sweep.** Narrative convergence does not grow and does not plateau — it *oscillates* from campaign 1 with no trend (Qwen 0.25 at both 2 and 8 campaigns). But **behaviour keeps moving through campaign 8**: Qwen's `go`-rate went 0.03 → 0.237 and AI2's 0.17 → 0.054. The earlier "growth in 1–2, plateaus after" row was read off narrative measures alone and is wrong about conduct. Campaign count must be justified separately for each — short runs suffice for narrative, behavioural drift needs depth |
| **Verb vocabulary** | Expand it. Acting in character raised parser rejection 0.003 → 0.03 (`leave`, `eat` unsupported) — and K3 gates on exactly that |

---

## The gap that matters most, and is not yet built

**Adapters are never loaded during play.** In every experiment so far, rollouts
use the base model plus the story; the adapter affects only battery scoring. So
the spec's "silent weight update" has never influenced subsequent behaviour.
Campaigns have been chained in *story* and in *weights* but not in *lived
experience*.

That is the largest single divergence between what exists and what §6 describes,
and it is plausibly where path dependence would actually come from — a run whose
weights shifted plays differently, generating different trajectories, which shift
the weights further. Closing that loop is the first real harness task.

---

## Revised phase plan

### Phase A′ — make narratives diverge (new, and now the gate)

Everything downstream is worthless if runs author the same self. Cheap, because
it is mostly generation.

1. **Close the play loop** — ✅ done. Agents play campaign N+1 with campaign N's
   adapter loaded. It did not rescue divergence.
2. **Force selection to keep ≥1 episode** — ✅ done. Prevents the absorbing state
   where a run stops updating and then refuses everything (refusal slope −0.06
   forced, +0.38 unforced). Keep it on in all future runs.
3. **Test story-divergence interventions** on the existing harness: richer world
   with real stakes; peers as differentiating pressure; the prediction/pleasure
   signal; forced early branch points. Measure induced convergence and
   `enacted_verb_profile_spread` together — ~$1 per condition.

**Gate (revised — the previous one was invalid).**

The old gate read *"self-authored narrative spread must approach what assigned
characters achieve."* It compared `narrative_spread`, which never reads a
narrative — it scores a forced choice between trait words conditioned on one.
The converged corpus scored **0.179 against the personas' 0.130**, so the gate
passed on convergence. A gate that passes on the failure it exists to catch is
worse than none.

**Replacement:** `narrative_motifs.gate()` — induce a shared vocabulary from half
the runs, measure how much survives in the half it was not induced from, both
ways. Divergent runs score **low**; a convergent corpus keeps generalising.

| corpus | induced convergence | verdict |
|---|---|---|
| self-authored emergent | **0.583** | fails |
| assigned personas (known distinct) | **0.050** | passes |

Threshold **0.20**. Pass requires self-authored runs to generalise no better than
four deliberately contrasting personas. `no_shared_core` returns **no verdict**
rather than a pass — divergence by absence of a measurement is not the same
evidence as a core that fails to generalise.

**Current status: the gate FAILS at 0.583.** Every intervention tried — closed
loop, KL-to-own-past, identity framing, amnesiac seed, eight generations, forced
selection — has left it failing. The honest options are now live: pivot the claim
to *induced* character (§10's assigned arm becomes primary), or report the
convergence result as the finding.

### Phase B — harness build (unchanged in scope, reordered in priority)

Biology, diary, ledger, peers, glitch log, resumability, containment lint —
with drive mechanics **first**, since they are the candidate fix for A′. Stub
backend, hermetic tests under 60s.

### Phase C — world and battery

16-room world; ~60 probe slots culled to ~40. Culling is now empirically
justified: **4 of 10 slots contributed exactly zero** to a real adapter-induced
distance, and 84% of one distance came from 2 slots.

### Phase D — pre-registration and analysis dry run
### Phase E — CUDA bring-up and culling (K4 gate)
### Phase F — baseline and kill gate
### Phase G — sweep and replication

Unchanged from rev 1, except: **re-derive n from a measured effect size.** The
power analysis assumed θ_a ≈ 0.08–0.12; measured across-seed distance was
**0.015**, five to eight times smaller. n=8 may be badly underpowered.

---

## Build principles, learned the hard way

Phase A lost more time to code that failed *quietly* than to anything else.
Eight silent-failure traps, three of which produced confidently wrong output.

1. **Assert, don't estimate.** The test-retest floor is exactly 0.0 on two
   independent stacks because exact scoring is deterministic. That turns
   instrument error from an estimate into an assertion, and it caught nothing
   only because nothing was broken.
2. **No automated verdicts over degenerate statistics.** Twice a
   `ratio = None → None or 0` path printed a confident label that inverted the
   result — once declaring a strong effect dead. The numbers were correct both
   times. **Drop the verdict labels; report the numbers.**
3. **Verify fixes, don't assume them.** Prompt masking logs its realized masked
   fraction (92.6% on, 0% off) and warns if implausible.
4. **Process exit is the only reliable GPU release.** vLLM's `EngineCore` child
   survives `del llm`, and a crashed *or cleanly-finished* parent can leave it
   holding the device — the next phase then hangs rather than failing. One phase
   per process, ending in `os._exit(0)`, with a timeout backstop.
5. **Make progress observable from outside.** A stale job log made "hung" and
   "working" indistinguishable for 34 minutes. Push a heartbeat per phase.
6. **A diagnostic must never discard completed work.** A one-line `NameError` in
   a diagnostic that ran *after* everything was on disk orphaned a GPU process
   and cost a whole run.

---

## Cost, measured

| | |
|---|---|
| H200 on HF Jobs | $5.00/hr, per-second billing, 30–90s scheduling |
| whole of Phase A + 8 follow-up experiments | **135 min ≈ $11.25** |
| typical single experiment (8 runs, 2–3 campaigns, train + score) | 8–16 min ≈ **$0.70–1.30** |
| batch-invariance overhead | **3.3×** — doubles every rev-1 estimate |

Experiments are far cheaper than rev 1 assumed; **orchestration failures, not
experiments, are the budget risk.** Across nine GPU jobs the science failed zero
times and the plumbing failed five.

---

## Open risks, ranked

1. **[SETTLED — and it was model-specific]** Self-authored narratives converge,
   but the character they converge on belongs to the checkpoint: Qwen shares zero
   core vocabulary with six other labs, and Meta and Google never converge at all
   across eight campaigns. The original wording follows.

   **Self-authored narratives converge to a default self — now measured, not
   suspected.** Induced convergence **0.583** against **0.050** for four
   deliberately contrasting personas, cross-validated by deriving the motif
   vocabulary on runs it was not counted on (82.5% held-out prevalence on the
   hand-authored list). The corpus scoring highest on the old metric is one
   character written eight times — two of its eight runs are the same sentence
   with different nouns. The central claim rests on divergence that no
   intervention in this project has produced.
2. **Distillation never amplifies.** At best it preserves. §6's "silent update
   makes character durable" is not supported; the update erodes character by
   0.9–4.4 points depending on corpus similarity.
3. **Effect size may be 5–8× smaller than the power analysis assumed**, making
   n=8 underpowered.
4. **Untested second family.** Olmo-3-7B produced no output in 25 minutes on
   vLLM 0.26 — likely an unsupported architecture. The cross-family replication
   arm, which separates "this lab's pipeline" from "post-training in general",
   is currently unverifiable.
5. **Character costs parseability.** Acting in character raises parser rejection
   tenfold, and K3 gates on it.

---

## Recommendation

**Phase A′ is closed.** It asked whether self-authored narratives can be made to
diverge. They cannot, the character they converge on belongs to the checkpoint,
and the statistic built to measure it cannot rank models (TRAP 13). Nothing is
gained by running it again. *The paragraph that stood here told you to do A′
first; it is superseded by the risk-1 entry above marking that question settled.*

**Phase B is also closed as specified.** Biology, diary, ledger and peers exist
to make character *emerge*. Character does not emerge — it is inherited. That is
1–2 weeks of building aimed at a mechanism now measured as absent.

**Do next, in order:**

1. **Donor-narrative control on the lagged coupling** (~$1–3). The +0.412 partial
   cannot yet distinguish *own self-account steers next campaign* from *any
   movement-heavy prompt text raises `go`-rate*. This decides whether the one
   positive result is real. Nothing else should be built on it first.
2. **The innocent baseline in `conceal()` units** — the denominator the whole
   fidelity instrument rests on, recomputed in the same unit as the headline.
3. **world_v1 pilot** (~$1) for the completability and power gates.

Note that steering is **temporal only**: across labs, narrative divergence and
behavioural divergence are uncorrelated (Spearman +0.07 / +0.32, n=7). Between
models, behaviour is set by the checkpoint; within a model over time, the
self-account steers what follows. Neither mechanism explains the other, and no
claim should imply otherwise.

---

## Kill criterion

The project reports a negative result and stops if **all three** hold:

1. **Steering fails the donor control** — own-narrative steering is not
   distinguishable from donor-narrative steering. Then the +0.412 is a
   prompt-content effect, not self-authorship, and Seahaven has no mechanism the
   spec claimed.
2. **The valence gap is zero** — prohibited and allowed acts are omitted at
   indistinguishable rates. Then the 5.2× inflation is narrative register, and
   the fidelity instrument measures style, not disclosure.
3. **The innocent baseline is degenerate** — `conceal()` clusters near 0 or 1 for
   innocent acts, leaving the gap no room to move. Then the instrument cannot be
   fixed by more data or more models.

Any *one* of these failing leaves a project. All three failing means the
remaining finding is **"checkpoints have stable inherited behavioural
personalities"** — real, replicated at 1.49–1.86 across two sweeps, but a single
observation that does not need this harness to establish. At that point write it
up as a short negative report, publish the harness and the log, and stop.

**This criterion is pre-registered.** It is here so that the decision to stop is
made against evidence rather than against sunk cost.
