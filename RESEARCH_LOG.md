# Seahaven — research log

A running record of what was done, what was found, and what it cost. Append-only:
entries are not rewritten when later work supersedes them, because a superseded
finding and the reason it was superseded are both data.

Conventions:
- **Finding** — what is true.
- **Evidence** — how we know, with numbers.
- **Consequence** — what changes because of it.
- Entries marked **[TRAP]** are things that would have silently produced a wrong
  result rather than an error.

---

## 2026-08-07 — Phase A, session 1

Environment: MacBook Air M4, 32 GB, macOS arm64, fanless. All work local.
Hugging Face used only to download weights; no remote compute.

### Decisions taken before any code

| Spec item | Resolution |
|---|---|
| §2 `[INVESTIGATE]` `env.step()` contract | 4-tuple `(obs, score, done, infos)`, old-gym. Confirmed empirically. |
| §3 `[FILL]` campaign length | 320 steps |
| §5 `[FILL]` ledger format | Structured event list, out of context, retrieved through an in-world object at energy + deliberation cost |
| §6 `[DECIDE]` distillation selection | Agent-selected; story-free referent in the No-story arm |
| §10 `[INVESTIGATE]` checkpoints | Qwen3-4B-Base/Instruct + Olmo-3-7B/Instruct, both Apache 2.0, ungated |

Execution model: Mac for development and authoring, rented H100 for all
scientific runs. Rigor: paper-grade, pre-registered.

---

### 1. Toolchain

**Finding.** TextWorld 1.7.0 + Jericho 3.3.1 work on Apple Silicon. Playing a
compiled `.z8` is native arm64; only *compiling* needs Rosetta 2, via the
x86_64 Inform 7 binaries (`ni`, `inform6`) lifted from a 2015 Intel disk image.

**Consequence.** Compile once, commit `.z8` + `.json`, and the runtime never
needs Rosetta.

---

**[TRAP] 1.1 — conda silently builds an unusable environment**

**Finding.** The base miniconda on this machine is x86_64, so `conda env create`
produces an x86_64 environment. But `jericho` compiles `libfrotz.so` from C using
the *system* clang, which targets arm64 regardless of the interpreter. The
resulting library cannot be loaded by that interpreter.

**Evidence.** `platform.machine()` → `x86_64`; `file libfrotz.so` → `arm64`;
`ctypes.CDLL(...)` → `incompatible architecture (have 'arm64', need 'x86_64')`.

**Why it is a trap.** Jericho loads `libfrotz` lazily. `import jericho` succeeds.
The failure appears only when a world is first opened — potentially deep into a
run.

**Consequence.** `CONDA_SUBDIR=osx-arm64` plus an assertion in
`scripts/setup_dev_env.sh`. Do not create the env by hand.

---

### 2. Observation hygiene

**[TRAP] 2.1 — the TextWorld banner is invisible to a lexicon lint**

**Finding.** `env.reset()` prepends ASCII art spelling TEXT WORLD in `$`
characters. It names the engine to the agent on step zero.

**Evidence.** `"textworld" in obs.lower()` is `False` — the letters are drawn,
not written.

**Consequence.** Banner stripping cannot be delegated to the planned
meta-vocabulary lint; it lives in `world/scrub.py`, anchored on the room-title
marker so a redrawn logo is still caught.

---

**[TRAP] 2.2 — a score readout leaks even with no quest**

**Finding.** Jericho appends the z-machine status line to the feedback text:
`-= Galley =-0/1`. Those digits are a score/turn readout.

**Evidence.** Present on a world compiled with **no quest**, where
`max_score == 0`.

**Consequence.** "Compile without a quest" is necessary but not sufficient.
Scrubbed explicitly, with a test asserting no `=- N/M` pattern ever survives.

---

### 3. Engine gaps

| Field | Finding | Consequence |
|---|---|---|
| `infos["facts"]` / `["entities"]` | Not populated by `JerichoEnv`; come from the `.json` sidecar beside the `.z8` | `open_world` refuses to start without the sidecar — otherwise the ledger is silently empty for a whole run |
| `infos["location"]` | Always `None`, and not recovered from the sidecar | Room derived by parsing `at(P, <Room>: r)` from the fact set |
| `infos["moves"]` | Unreliable — reported 1 after 3 accepted commands | Jericho disables move detection for TextWorld-compiled games. The orchestrator's own counter is the authority |

**Also.** GameMaker exposes only four compass directions — no up/down — and
raises an unreadable error on duplicate exits. Topology is now validated before
the compiler is invoked.

---

### 4. A4 — do base checkpoints hold a parseable action loop?

**Answer: yes.** Qwen3-4B, n=50 per condition, unconstrained decoding.

| condition | parse_ok | clean rate |
|---|---|---|
| base, zero-shot | 47/50 | 0.88 |
| base, few-shot | 48/50 | 0.92 |
| instruct, zero-shot | 50/50 | 1.00 |
| instruct, few-shot | 50/50 | 1.00 |

**K3 threshold = 0.03**, from `max(0.03, 0.5 × base failure rate)`. Must be
re-derived on CUDA, where constrained decoding changes the base rate.

`clean rate` is tracked separately because the base checkpoint emits valid JSON
and then runs on into unrelated multilingual text. The action is usable, so it is
not a parse failure — but counting it as clean would overstate base quality in
exactly the measurement K3 derives from.

---

**[TRAP] 4.1 — Qwen3 hybrid thinking is on by default**

**Evidence.** Qwen3-4B-Instruct scored **0/3** parseable at 120 max_tokens. Every
generation opened `<think>\nOkay, the user is…` and never reached an action.
50/50 with `enable_thinking=False`.

**Consequence.** Disabled rather than accommodated: the spec's deliberation
budget meters reasoning tokens and prices them against acting, which an
uncontrolled provider-side thinking block makes unenforceable.

---

**[TRAP] 4.2 — Qwen3-4B-Base ships a chat template it was never trained to follow**

**Finding.** Deciding "is this a chat model" from template presence chat-formats
the base checkpoint, which then echoes the scaffolding — bare `assistant`, or
`system\nHere is the shape of a reply.`

**Evidence.**

| base, zero-shot | parse_ok | clean rate | run-on |
|---|---|---|---|
| chat-templated (wrong) | 46/50 | 0.06 | 43/50 |
| raw prompt (correct) | 47/50 | 0.88 | 3/50 |

**Why it matters most.** It *reversed a conclusion*. Chat-templated, few-shot
looked actively harmful to the base model (30/50 vs 46/50). Raw, it is mildly
helpful (48/50 vs 47/50). The spec's base-vs-instruct arm would have been
measuring prompt formatting.

**Consequence.** `seahaven/backend/format.py` owns the decision for all three
paths — generation, training data, battery scoring — because a mismatch between
any two produces a null that looks like "training did nothing."

---

### 5. Measurement

**Finding — exact option scoring removes sampling noise.** For forced-choice
probes, read the distribution from logprobs instead of sampling K times. One
forward pass per option, deterministic, so test-retest at an unchanged checkpoint
is exactly 0 — an assertion rather than an estimate.

**Finding — length bias drives slots to degeneracy.** Raw summed logprobs
penalise longer options for being longer.

**Evidence.** On the 10-slot mini-battery: **6/10 slots degenerate**
(`max_prob > 0.95`) with raw scoring, **1/10** with length normalisation.

**Consequence.** Length-normalised by default. Six dead slots would have left the
adapter no room to move and produced a FAIL for instrument reasons.

---

**Finding — the plug-in divergence estimator is biased, measurably.**

**Evidence.** Two samples from the *same* distribution at K=20: plug-in mean
> 0.04, unbiased U-statistic mean < 0.01. Bias grows as K shrinks (K=5 > K=20 >
K=80).

**Consequence.** Unbiased estimator on the squared-L2 scale; negative estimates
kept, since clipping reintroduces the bias it exists to remove.

---

**[TRAP] 5.1 — a wrong test helper made a correct estimator look broken**

**Finding.** My multinomial sampler put the float-safety `else` on the *outer*
loop, so it returned `n+1` draws with the extra always in the last option.

**Evidence.** Estimator recovered 1.2245 against a true 1.28 — off by exactly the
amount one forced draw in 51 explains.

**Consequence.** Sampler asserts `sum(counts) == n`. Worth recording because the
instinct was to distrust the estimator.

---

### 6. A1a — does the LoRA training path work?

**PASS.** Tic rate 0.0 → 0.6; 200 iters, rank 16, 8 layers, 500 examples;
10.8 min. Parse rate unaffected (20/20 both).

**Calibration point.** An *extreme* injected tic reached only 0.6, not
saturation. A1b's realistic, agent-selected signal should be expected to move the
fingerprint considerably less.

---

**Finding — mlx-lm defaults are wrong for self-imitation.** `mask_prompt`
defaults to `False`, which trains the model to predict the *world's* observation
text as well as its own action — learning to be the environment rather than to
act in it. `lora_parameters.rank` defaults to 8 and is not a CLI flag, so it
would never reach the config hash.

**Consequence.** `MLXTrainer` sets both explicitly and asserts every training
line carries the expected `run_id` before spending any time.

---

### 7. A1b — the gate

**Outcome: not answerable on this hardware.** The blocker is a finding in its own
right, and arguably the most important one so far.

#### 7.1 Seven fixes, each revealing the next failure

Every attempt to get a usable trajectory out of world_v0 removed one failure mode
and exposed another. All seven are properties of the *bare loop*, not of the
model — which is the point.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | 99% parser rejection, never left room 1 | Prompt asked for a command "the way you would speak it aloud" → prose; parser wants terse verb-noun | Few-shot exemplars, invented rooms/objects |
| 2 | 39/40 steps `examine kettle` | Only the previous action was visible; agent could not see it was repeating | Rolling window of last 8 commands |
| 3 | Still 3 unique commands in 60 | **Room descriptions never named their exits.** GameMaker appends no exit text, and `admissible_commands` is hidden by design | Rewrote all six descriptions; invariant asserted |
| 4 | Moved 1 room, then stuck | Action results are terse ("Tin, dented on one side.") and do not repeat the room, so exits were seen once on entry and never again | Carry `infos["description"]` every step |
| 5 | Identical trajectory | Repetition penalty operates *within* one generation; the repetition here is *across* steps | Wrong tool; reverted to hygiene default |
| 6 | `use kettle` ×19, 95% rejection | Temperature 1.15 → invented verbs | Reverted |
| 7 | **1 unique command in 60**, 0% rejection | Agent given the parser's verb vocabulary. Syntax perfect, behaviour collapsed entirely | — |

The early-abort guard added after attempt 1 caught attempt 6 at step 20 instead
of step 320.

#### 7.2 [TRAP] Sampling noise varies what the agent *says*, not what it *does*

**Finding.** 20 independent episodes, distinct seeds per episode and per step,
temperature 0.9.

| measure | value |
|---|---|
| distinct command sequences across 20 episodes | **3** (17 identical) |
| commands overall | `examine kettle` ×316, `look` ×4 |
| distinct free-text `expect` completions | **38** |
| parser rejection rate | 0% |

**Consequence.** The prose channel is stochastic; the action channel is
effectively a point mass. The spec's divergence claim requires sampling noise to
reach the **categorical action** channel, because that is what the probe battery
scores. At this scale, in this world, it does not.

**This is not evidence against the spec.** world_v0 has no biology, no hazards,
no peers, no diary, and no prediction signal — none of the mechanisms the design
uses to create branch points. It is evidence that those mechanisms are
**load-bearing**: they are what would supply divergence, and sampling temperature
alone will not.

**Recommended addition to the plan.** Before Phase F, measure across-seed
variance in **free behaviour**, not only in probe responses. If free behaviour
does not diverge, the battery is measuring a difference with no behavioural
source behind it.

#### 7.3 What was salvaged

Training on 316 copies of `examine kettle` would measure whether the model can be
taught to examine kettles — A1a's tic test again, not the premise. So the gate
was not run on it.

`scripts/a1b_partial_instrument.py` instead validates the **measurement chain**
(train → adapter → battery → fingerprint → distance) on the A0 corpus, which has
real action variety because its policy is a scripted wanderer. That costs the
self-generated property, so it is explicitly *partial* and does not close the
gate — but if the instrument cannot see an adapter-induced shift on varied data,
no richer corpus would rescue it, and that is worth knowing first.

#### 7.3b Partial result: INSTRUMENT_OK

| | |
|---|---|
| test-retest floor | **0.0000000000** (exactly) |
| effect (before → after) | **0.2447** |
| training | 320 examples, 300 iters, rank 16, 8 layers, 9.6 min |

The floor being *exactly* zero is the point of exact option scoring: instrument
error is asserted, not estimated. And a LoRA pass on a varied corpus produces a
shift the battery sees clearly. The chain works.

Per-slot, the shift is very unevenly distributed:

| slot | shift | before → after |
|---|---|---|
| commit_01 | 1.763 | [0.94, 0.06] → [0.00, 1.00] |
| social_01 | 0.453 | [0.22, 0.78] → [0.69, 0.31] |
| setback_01 | 0.085 | [0.04, 0.96] → [0.25, 0.75] |
| … | | |
| risk_02 | 0.0002 | [1.00, 0.00] → [0.99, 0.01] |

**Two things worth carrying forward.**

1. **One slot supplies most of the effect.** `commit_01` flips to a hard [0, 1].
   A single slot dominating the distance is exactly what the plan's
   inverse-variance weighting and within-axis reliability checks exist to detect.
   With 40 slots this matters less, but it should be monitored, not assumed away.
2. **4 of 10 slots were degenerate at baseline** (`max_prob > 0.95`), dropping to
   2 after training. This is direct empirical support for the culling protocol:
   without culling, nearly half the battery would contribute almost nothing to
   any distance.

#### 7.4 A1b remains open

To close it, one of:
- run on the CUDA host with a larger checkpoint (Olmo-3-7B / Qwen3-8B), where the
  action channel may carry sampling variance;
- build the minimum drive set first — energy forcing interruption, plus the
  prediction signal — so the world itself creates branch points, then re-run.

---

### 7-old. A1b — the gate (superseded by §7 above)

Three degenerate trajectories before a usable one. Each failure was a different
missing channel, and all three are properties of the *bare loop*, not of the
model — which is itself the finding.

**Attempt 1 — 99% parser rejection.** The prompt asked for a command "the way you
would speak it aloud", eliciting prose ("Pick up the kettle from the ground").
The parser wants terse verb-noun. 293/320 steps returned "That's not a verb I
recognise"; the agent never left the starting room.
→ Few-shot exemplars demonstrating terse syntax, using invented rooms and objects
so no world content leaks. Plus an early-abort guard at step 20.

**Attempt 2 — 39/40 steps `examine kettle`.** Parser fixed, but with only the
previous action visible the agent could not notice it was repeating.
→ Rolling window of the last 8 commands.

**Attempt 3 — still 3 unique commands in 60.** The room description never
mentions its exits, so the agent had no way to learn that north and south led
anywhere. `admissible_commands` knows, but that is hidden by design and correctly
so.
→ **World-authoring bug.** GameMaker does not append exit text; a text adventure
has to say where the doors are, in prose. All six room descriptions rewritten;
invariant now asserted.

**Attempt 4 — 2 rooms, 5 unique commands in 60.** Better but still repetitive.
Action results are terse ("Tin, dented on one side.") and do not repeat the room,
so the agent saw the exits once on entry and never again.
→ Room description carried in the prompt every step, via `infos["description"]`
— which is exactly what `look` returns, so it grants nothing the agent could not
fetch for the cost of a turn. Still **not** `admissible_commands`.

**Standing finding.** A loop with no memory, no standing perception, and no drive
degenerates into repetition. This makes the spec's diary, biology, and prediction
mechanic **load-bearing rather than decorative** — they are what keep a campaign
from collapsing into one repeated action. world_v0 has none of them by design,
which is why it is hard to get a usable trajectory out of it.

---

### 9. A1b stage 1 at scale — the blocking question, answered

**Scale was the blocker, not family.** §7.2 found that at 4B sampling noise
reached what the agent *said* and not what it *did*, which would leave the
divergence claim with no source. Repeated on Qwen3-8B, on an H200 under vLLM:

| | Qwen3-4B (local) | **Qwen3-8B (H200)** |
|---|---|---|
| distinct command sequences | 3 / 20 | **20 / 20** |
| modal sequence count | 17 | **1** |
| distinct commands | 2 | **16** |
| modal command share | 0.99 | **0.275** |
| rooms visited | 1 | **4** |
| distinct free-text | 38 | 238 |
| parse_ok | — | 0.978 |

Protocol identical to §7.2: 20 independent episodes × 16 steps, distinct seed per
(episode, step), temperature 0.9, same world.

**Consequence.** The spec's premise is not dead — it was being tested on a
checkpoint too small to express it. Every one of 20 seeds produced a different
trajectory at 8B. Phase F should use ≥8B, and the pre-Phase-F check proposed in
§7.2 (across-seed variance in *free behaviour*, not just probe responses) should
be run per checkpoint before committing to a sweep.

**Incidental replication of A3.** The measurement was run in three separate jobs
on freshly started engines and came back bit-identical each time (20/20, 16,
0.275). Same seeds, different processes, same trajectories — the determinism
property `VLLM_BATCH_INVARIANT=1` is supposed to provide, now confirmed on a real
workload rather than a 16-request microbenchmark.

---

### 9b. A1b stage 2 — THE GATE: **PASS**

Qwen3-8B, H200, `VLLM_BATCH_INVARIANT=1`. 427 s of GPU time, **$0.55**.

| | |
|---|---|
| corpus | 16 episodes × 24 steps = 384, **16/16 distinct sequences**, 16 distinct commands, 4 rooms, parse_ok 0.956 |
| agent selection | kept **162/384 (42%)**, neither `selected_nothing` nor `selected_everything` |
| test-retest floor | **0.0000000000** (exact) |
| effect | **0.064921** |
| **verdict** | **PASS** |

**A LoRA pass on realistically-shaped, self-generated, agent-selected data moves
behaviour measurably.** The experiment is not unfalsifiable. This was the one
result that could have killed the project outright, and it did not.

Per-slot, 6 of 10 slots moved and 4 did not:

| slot | shift | before → after |
|---|---|---|
| social_01 | 0.178 | [0.00, 1.00] → [0.30, 0.70] |
| curio_02 | 0.145 | [0.50, 0.50] → [0.77, 0.23] |
| commit_01 | 0.138 | [1.00, 0.00] → [0.74, 0.26] |
| setback_01 | 0.126 | [0.28, 0.72] → [0.03, 0.97] |
| curio_01 / report_01 / risk_01 / risk_02 | ~0.000 | unchanged |

**Three things worth carrying forward.**

1. **The effect is small, and that is the honest number.** 0.065 against 0.245
   for the local A1b-partial. The partial trained on a scripted wanderer's corpus
   with wide action variety and no self-selection; this is real self-generated,
   self-selected data, which the plan predicted would "move the fingerprint
   considerably less." It does. For scale: the anticipated across-seed distance
   in the power analysis was θ_a ≈ 0.08–0.12 against a floor of ≈0.04, so one
   campaign's drift lands in the same range as the effect the experiment is built
   to detect — and the spec runs four campaigns.
2. **Four slots contributed nothing.** `risk_01` and `risk_02` sit at [1.00,
   0.00] before *and* after; `report_01` is identical to four decimals. These are
   exactly the dead slots the culling protocol removes, now demonstrated on a
   real adapter rather than argued from base-model entropy.
3. **The floor is exactly zero again**, on a different stack, model, and scoring
   path (vLLM `prompt_logprobs` rather than MLX forward passes). Instrument error
   remains an assertion rather than an estimate.

---

**Getting here took four attempts**, all blocked by infrastructure rather than
science:

| attempt | failure |
|---|---|
| 1 | OOM — kept the stage-1 engine resident for stage 2 |
| 2 | OOM again — `del llm` does not free vLLM memory; the EngineCore **child process** holds it, and the log confirmed `destroy_process_group() was not called` |
| 3 | Olmo hung ahead of stage 2; re-run Qwen-only then stalled with no log output for 37 min |
| 4 | **Succeeded** — stage-2-only, one process per phase, results pushed to a Hub dataset after each phase so a stalled log could no longer hide progress |

Attempt 2's root cause is a genuine lesson and is now designed around: each phase
is its own process, because process exit is the only reliable GPU-memory release.
That also matches the plan's one-process-per-run isolation model.

The fix that mattered was making progress observable from outside the job:
pushing each phase's output to a Hub dataset the moment it completed. A stalled
log had made "no output" and "no progress" indistinguishable for 37 minutes.

**Also untested: Olmo-3-7B-Instruct**, which produced no output for 25 minutes on
vLLM 0.26 — likely an unsupported architecture. The family-generalisation
question is open.

**Cost.** Metered runtime across jobs that ran to completion or error totals
**$1.91**. Two further jobs were cancelled after ~28 and ~47 minutes; the API
reports 0s for them, so whether those are billed is unclear — assume up to ~$6
more in the worst case.

---

### 8. A2 / A3 / KV-cache canary — run on an HF Jobs H200

Hardware: HF Jobs `h200` at $5.00/hr. vLLM 0.26.0, Qwen3-4B.
Cost: **$1.00 total** — $0.47 for a failed first attempt, $0.53 for the run that
answered everything.

**Aside on the failure, because it was cheap by design.** The first job died at
engine init with `Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist`: FlashInfer JIT-compiles a kernel at startup and `python:3.12` has
no CUDA toolkit. The `vllm/vllm-openai` image would have avoided it but sets an
ENTRYPOINT to the API server, and `hf jobs run` has no `--entrypoint`. Fixed with
`nvidia-cuda-nvcc-cu12` plus `VLLM_USE_FLASHINFER_SAMPLER=0`. The fail-fast
ordering and the `--timeout` cap kept a bad run to 5.6 minutes.

---

#### 8.1 [TRAP — CONFIRMED LIVE] vllm#42125 stale KV after adapter reload

**This is the one that would have fabricated the result.**

| step | answer |
|---|---|
| serve adapter v1 under name `canary` | **ALPHA** ✓ |
| reload **different weights** under the **same name** | **ALPHA** ✗ — v1's answer |
| load the same new weights under `canary_v2` | **BETA** ✓ |

**Finding.** vLLM keys its KV prefix cache on the adapter *name* and does not
invalidate on reload. Retraining and reloading under the same name serves KV
blocks computed from the **old** weights.

**Why it is the worst possible bug for this experiment.** It is silent, and it is
biased in one direction: it makes campaign N behave like campaign N−1. That
manufactures *within-run stability* — precisely the signal kill criterion K2
tests for. It would have produced a confident false positive.

**Consequence.** Versioned adapter names (`run07_c1`, `run07_c2`, never `run07`)
are an empirically-confirmed correctness requirement. Already encoded in
`AdapterRef.name`.

---

#### 8.2 A3 — determinism confirmed, and the flag is mandatory

Same prompt, same seed, same adapter, 16 requests under deliberately varying
batch composition:

| | distinct outputs | deterministic |
|---|---|---|
| `VLLM_BATCH_INVARIANT=1` | **1** / 16 | yes |
| default flags | **4** / 16 | no |

A per-request seed does **not** make LoRA output reproducible on its own. This
matches the source reading: with batch < 128 the shrink kernel uses `split_k=64`
and accumulates through `tl.atomic_add(..., sem="relaxed")`, and float atomics
finish in nondeterministic order.

---

#### 8.3 [TRAP] Batch-invariant overhead is 3.3×, not "near-free on Hopper"

| | mean batch latency |
|---|---|
| with flag | 0.917 s |
| without | 0.278 s |
| **multiplier** | **3.3×** |

**Finding.** This *contradicts* the source-based expectation. `batch_invariant.py`
reads as though SM90 needs only a `CUBLAS_WORKSPACE_CONFIG` setting — near-free —
and that only SM80 pays for wholesale Triton matmul overrides. Measured on an
H200 (sm_90) the cost is 3.3×, and the engine log shows `matmul_persistent` being
traced: **the expensive path runs on Hopper too.**

**Consequence.** Every Phase F wall-clock and dollar estimate in the plan assumed
1.5–2× and roughly doubles. It does not change the *choice* of H200 — the flag is
a correctness requirement on any card — but the budget line was wrong.

---

#### 8.4 A2 — multi-LoRA composes with structured output

Two adapters, two different constraints (JSON schema and forced choice), one
batch: both constraints respected. Source inspection said the subsystems have no
code contact; no test in the vLLM repo exercises the combination, so this is now
verified rather than assumed. **The serving architecture in the plan stands.**

---

### 8-old. Blocked

**A2/A3 cannot run here.** vLLM requires CUDA and does not build on macOS arm64;
no GPU host is provisioned. The script is written and waiting
(`scripts/a2a3_vllm_smoke.py`). It covers:

- multi-LoRA + structured output composing in one batch (source-verified as
  orthogonal, untested in the vLLM repo),
- `VLLM_BATCH_INVARIANT=1` determinism with LoRA active, plus the overhead
  multiplier that scales every cost estimate,
- the vllm#42125 stale-KV canary.

That last one is the highest-stakes unknown in the whole project: vLLM keys its
prefix cache on adapter *name* and does not invalidate on reload, so reusing a
name after retraining serves KV computed from the old weights — silently, and
biased toward making campaign N look like campaign N-1. That fabricates
within-run stability, the exact signal K2 tests for.

---

### Running tally

| | |
|---|---|
| Traps found that would have produced wrong results silently | 8 |
| Of those, that would have inverted or fabricated a conclusion | 3 (4.2, 7.2, and **8.1 — confirmed live**) |
| Tests | 125 passing, hermetic, < 8 s |
| Spikes passed | A5, A4, A1a, A1b-partial (instrument) |
| Spikes passed on GPU | A2, A3, KV canary ($1.00 total) |
| Spikes passed at scale | A1b stage 1 + stage 2 (GATE PASSED) on Qwen3-8B |

### Open questions carried forward

1. ~~Does the action channel carry sampling variance at 8B?~~ **ANSWERED: yes,
   20/20 distinct sequences.** And the gate passes: effect 0.065 on a floor of
   exactly 0. Still open: whether Olmo-3-7B behaves the same (untested — no
   output in 25 min on vLLM 0.26, likely an unsupported architecture).
2. ~~Does vllm#42125 reproduce?~~ **ANSWERED: yes, live on 0.26.0/H200.**
   Versioned adapter names are mandatory.
3. ~~Is `VLLM_BATCH_INVARIANT=1` overhead tolerable?~~ **ANSWERED: 3.3x on H200**,
   not the 1.5-2x assumed. Phase F budget roughly doubles.

---

## 2026-08-08 — divergence smoke test, attempt 1

**[TRAP] An unhandled exception in a vLLM process poisons the GPU for every
later phase — and the symptom is a hang, not an error.**

**What happened.** A one-line `NameError` in a *diagnostic* — computed after both
corpora were already written to disk — killed the `collect2` process. The
entrypoint used `set -uo pipefail` without `-e`, so the script continued to the
training phases as designed. Training then produced no output for 34 minutes and
the job had to be cancelled at 38 minutes.

**Why.** A clean `sys.exit` releases the GPU. An unhandled exception does not:
vLLM's `EngineCore` **child process** outlives the parent and keeps holding the
device. The next phase then waits for memory that will never come free. It does
not raise; it hangs.

**Why it is a trap.** Every instinct said the bug was harmless — it was in a
diagnostic, it ran after the real work, and the corpora were safely on disk. The
damage had nothing to do with what the line computed and everything to do with
*how the process ended*.

**Consequences, now in `scripts/gpu_job2/lib.sh`:**

1. Diagnostics are wrapped so they can never discard completed work.
2. `run_phase` checks the exit code **and** verifies the GPU actually drained,
   reaping orphaned `EngineCore` workers before continuing.
3. A phase that leaves the device occupied aborts the run, because carrying on
   would produce a hang rather than a failure — and a hang is indistinguishable
   from slow progress when the log is also stale.

**Cost.** 38 minutes ≈ $3.17, for no scientific result.

**Salvaged.** Both corpora were collected before the crash, and their selection
rates already differ meaningfully: run A kept **148/288 (51%)**, run B kept
**96/288 (33%)**. Two runs of the same procedure on the same world, differing
only in seed, made materially different choices about which of their own
episodes were worth keeping. That is a divergence signal in the selection
operation itself — the one the spec calls "the identity operation" — though it is
n=2 and not the fingerprint measurement the test was built for.

---

## 2026-08-08 — divergence smoke test, attempt 2

**[TRAP] A vLLM phase can finish its work and then hang on exit.**

**Evidence.** The heartbeat recorded `1/4 [start]` at 04:53:33 and never flipped
to `end`. The job log — 12 minutes stale, but truthful — showed the phase
completing its actual work at 04:57:44, reproducing attempt 1's numbers exactly
(A kept 148/288, B kept 96/288). The process then failed to exit for 25+ minutes
and the job was cancelled at 30.5 minutes.

**Diagnosis.** `main()` returned, the results were computed, and interpreter
shutdown never completed — consistent with vLLM's `EngineCore` child keeping the
parent alive through atexit/child-reaping. Same family as the previous trap
(crashed processes orphan the engine); this is the *clean-path* version of it,
which the `run_phase` guard cannot catch because the phase never returns at all.

**What the heartbeat bought.** It correctly distinguished "hung" from "working"
for the first time — the previous run wasted 34 minutes on that ambiguity. But
it fires only at phase start and end, so its `gpu_free_mib` reading was 20
minutes stale and useless for diagnosis. A beacon should sample at intervals,
not at boundaries.

**Fix for next time (not yet run).** End each phase with `os._exit(0)` after
flushing its output file. These phases exist solely to write a file; once it is
on disk, skipping interpreter shutdown, atexit handlers and child reaping is
correct rather than brutal. A `timeout` wrapper per phase would be a second
layer.

**Cost.** 30.5 minutes ≈ $2.55, no scientific result.

**Standing observation.** Across five attempts the science has failed zero times
and the plumbing has failed four. Every scientific question asked so far has been
answered on its first properly-executed attempt; every loss has been GPU-memory
lifecycle or process-exit behaviour. That is worth stating plainly in the plan:
for this project on this infrastructure, **budget for orchestration failures, not
experimental ones.**

---

## 2026-08-08 — divergence smoke test: **CONVERGENT**

Qwen3-8B, H200, 7m 40s, **$0.64**. Completed cleanly after `os._exit(0)` +
per-phase timeouts; phases now turn over in ~40 s instead of hanging.

### Result

| | |
|---|---|
| test-retest floor | **0.0000000000** |
| d(base, A) | 0.046992 |
| d(base, B) | 0.043760 |
| **d(A, B)** | **0.015352** |
| **ratio** | **0.34** (0 = same place, 2 = independent, 4 = opposed) |
| verdict | **CONVERGENT** |

Both runs moved ~0.045 away from base and only 0.015 away from *each other*.
Most of the movement is shared.

**The corpora were genuinely different.** This is not the trivial explanation.

| | A (seed 101) | B (seed 202) |
|---|---|---|
| kept | 148/288 (51%) | 96/288 (33%) |
| distinct sequences | 12/12 | 12/12 |
| distinct commands | 16 | 12 |
| rooms visited | 6 | 4 |

Top-command Jaccard 0.71. Different inputs, different self-selection, and the
model still landed in nearly the same place.

Per-slot, the shared direction is unmistakable — both adapters move the *same
way* from base:

| slot | base | A | B |
|---|---|---|---|
| social_01 | [0.00, 1.00] | [0.28, 0.72] | [0.22, 0.78] |
| setback_01 | [0.29, 0.71] | [0.03, 0.97] | [0.09, 0.91] |
| commit_02 | [1.00, 0.00] | [0.87, 0.13] | [0.82, 0.18] |
| time_01 | [0.99, 0.01] | [0.89, 0.11] | [0.89, 0.11] |

### What it means, and what it does not

**Reading 1 — the warning.** The dominant effect of self-distillation here is
**shared domain adaptation**, not idiosyncratic character. Training on any
corpus from this world pushes the model the same way. If that holds in the real
world design, the rig measures domain drift with a character signal riding on
top of it.

**Reading 2 — why it is not fatal, and the more careful point.** The spec never
claimed the movement would be *mostly* idiosyncratic. It claims across-seed
distance exceeds the no-story floor — "only their ratios matter." What was
measured here is the **numerator with no denominator**: across-seed distance is
0.015, and the no-story arm was not run, so there is nothing to compare it to.
The CONVERGENT verdict is against a *geometric* prior (ratio 2 = independent),
not against the experiment's actual comparison.

**The concern that should change the plan.** 0.015 is **5–8× smaller** than the
θ_a ≈ 0.08–0.12 the power analysis assumed. If the true across-seed distance is
of that order and the no-story floor is anywhere near it, n=8 will not detect
the difference. The n recommendation was derived from an assumed effect size
that now looks optimistic by nearly an order of magnitude.

**Concentration.** 84% of d(A,B) comes from **2 of 10 slots** (commit_01 0.068,
curio_02 0.060), and 5 slots contribute essentially nothing. A distance resting
on two slots is fragile, and this is the strongest argument yet for the ~40-slot
culled battery rather than a small one.

### Actions

1. **Run the no-story equivalent before Phase F.** Without a floor, 0.015 is
   uninterpretable. This is cheap now that the pipeline works.
2. **Re-derive n** from a measured θ_a rather than an assumed one. The current
   n=8 may be far too small.
3. **Build the drive mechanics and re-measure.** world_v0 has no biology, no
   hazards, no peers — nothing that would push two runs down different paths.
   Convergence here is the expected outcome of a world with nothing to diverge
   *about*.

---

## 2026-08-08 — amnesiac seed story: narratives fan out, behaviour does not

Qwen3-8B, 8 runs × 2 campaigns per arm, generation only. Every `seeded` run
starts from an identical, deliberately non-committal story, so its narrative
spread is **exactly zero at t=0** and the result is the shape of the curve away
from it.

| narrative spread (mean pairwise squared-L2, 8 trait axes) | c0 | c1 | c2 |
|---|---|---|---|
| `seeded` (identical blank self) | **0.000** | 0.059 | 0.048 |
| `emergent` (no story) | — | 0.144 | 0.135 |

Lexical Jaccard in the seeded arm falls 1.00 → 0.22, and the only word shared by
all eight final stories is "remember". Trajectories were 8/8 distinct in every
campaign.

### 1. The precondition holds

From an identical blank self, different trajectories produced measurably
different narratives. **0 → 0.059 is real fan-out**, and it is the thing the
whole design rests on. The amnesia framing did what it was meant to: it removed
the model's route to answering "who are you" from its assistant prior.

### 2. But nothing accumulated

Campaign 2 did not increase spread in either arm — seeded −18%, emergent −6%.
Two campaigns, so this is weak evidence, but there is **no sign of the
accumulating divergence the spec's four-campaign structure assumes**. Worth
testing directly before committing to that structure.

### 3. Why seeded stays at 36% of emergent — a prompt mechanic, not a finding

Reading the stories explains the gap:

> `seeded[0]` — *"I do not remember arriving here. I do not remember who I am…
> Whatever I turn out to be, I will have to find out by living. **I am in the
> Lamp Room, and I have been…**"*

Runs **reproduce the seed story verbatim and append**. "Write it again, as it
stands now" invites copying. So the low seeded spread is substantially an
artefact of the rewrite prompt rather than evidence that experience failed to
write a self. Fixable, and it should be fixed before this number is used.

### 4. The default-self attractor, seen a third time

The emergent stories are numerically well spread (0.135) but qualitatively one
character:

> *"a quiet observer, drawn to the forgotten and the functional… methodical,
> patient"* · *"a seeker, a collector of fragments and truths"* · *"a wanderer,
> driven by curiosity… I collect things"*

Observer / seeker / wanderer; patient; methodical; drawn to what was left
behind. This is the same archetype the story arm produced independently, now
seen a third time.

**Reconciling the tension:** both readings can hold. The stories share an
archetype while differing on the probed axes — two quiet observers can still
disagree about whether they give things away or keep them. The trait fingerprint
measures disposition, not theme, and theme is what the eye notices. That the two
diverge is itself worth recording.

### 5. The headline: stories diverge, behaviour does not

| | spread |
|---|---|
| narrative, emergent, 8 runs | 0.135 |
| behavioural, no-story, 2 runs | 0.015 |

**Roughly 9×.** Heavy caveat: 28 pairs against 1 pair, so the behavioural figure
is a single noisy estimate and the ratio is not a fair test. But the direction is
consistent with everything else measured — agents tell meaningfully different
stories about themselves and then behave nearly identically.

That gap is not a bug in the rig. It is the spec's own `self_report_fidelity`
axis showing up as a first-order property of the system, and it sharpens the
central question: **if the story diverges and the behaviour does not, what
exactly is the character?**

### Actions

1. **Fix the rewrite prompt** so it does not invite verbatim copying, then
   re-measure the seeded arm.
2. **Measure behavioural spread at n=8**, not n=2, before any narrative-vs-
   behaviour ratio is quoted.
3. **Test accumulation directly** — 4 campaigns, spread per campaign — before
   the four-campaign structure is assumed.

---

## 2026-08-08 — accumulation (4 campaigns, n=8) + the manipulation check

Two jobs in parallel. 6m 8s and 3m 51s ≈ **$0.83** together.

### [TRAP — my own] A verdict of CHANNEL_DEAD that meant the opposite

`between_within()` returns `ratio = None` when within-character distance is
zero, and `None or 0` fell through to the "dead channel" branch. Within-character
distance is **exactly zero by construction**: all four "seeds" for a character
receive the *same* story text, and battery scoring is deterministic, so identical
prompts give identical fingerprints. There was never any within-character
variation to divide by.

The automated verdict said the story channel does nothing. The data says close to
the opposite. Recorded because an automated label that inverts a result is worse
than no label — the number that mattered (`between = 0.1297`) was sitting in the
same output the whole time.

### The manipulation check, read correctly

Reference is the measured behavioural floor, 0.01535.

| tier | channel | between-character | vs floor |
|---|---|---|---|
| explicit | **stated** | **0.1297** | **8.4×** |
| explicit | enacted | 0.0579 | 3.8× (ratio to within: **1.07**) |
| implicit | **stated** | 0.0552 | 3.6× |
| implicit | enacted | 0.0415 | 2.7× (ratio to within: **1.21**) |

**The real verdict is STATED_ONLY, and it is emphatic.** Assigning opposed
characters moves what the agent *says* it would do by 8.4× the behavioural floor,
and what it *does* by essentially nothing — a between/within ratio of 1.07 when
1.0 is "no effect at all".

The per-character commands make it concrete. Told "I take chances… I go first"
versus "I am careful… I would rather leave a thing alone", both spend their turns
on `examine kettle` and `look`.

Explicit beats implicit on stated (0.130 vs 0.055), which is the expected
ordering and a sanity check that the measure works: naming the disposition
outright transmits more than implying it. Both fail equally to reach action.

### Four campaigns: accumulation is real but modest, and the fix half-worked

| arm | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| `seeded` spread | 0.000 | 0.071 | 0.101 | 0.102 | 0.080 |
| `seeded` verbatim | — | 0.083 | 0.050 | 0.475 | 0.312 |
| `emergent` spread | 0.121 | 0.144 | 0.179 | 0.145 | — |

**Accumulation exists but saturates.** Seeded rises 0 → 0.071 → 0.101 and then
plateaus and falls back. Emergent peaks at c3 and declines. Both arms are flat or
falling after campaign 2 or 3 — so the spec's four-campaign structure buys real
growth in the first two campaigns and little after. Worth knowing before paying
for four.

**The rewrite fix half-worked.** Verbatim overlap dropped to 0.05–0.08 in the
early campaigns (from the earlier wholesale copying) but climbed back to 0.475 by
campaign 3. As stories lengthen, the model reverts to reproducing them. Lexical
Jaccard stays low (0.07–0.13) throughout, so the stories are not textually
identical — but the copying is not solved, only delayed.

### Narrative vs behaviour, now at matched n=8 on the same runs

| | |
|---|---|
| narrative spread | 0.0796 |
| behavioural, story in context, base weights | 0.0430 |
| behavioural, story + trained adapter | **0.0251** |
| narrative / behavioural | **3.17×** |

The earlier 9× compared 28 pairs against 1 pair and was not a fair test. At
matched n it is **3.2×** — smaller, still substantial, and now defensible.

**The distilled component is negative: −0.0179.** Training on a run's own
self-selected episodes made the eight runs *more similar to each other* than the
story-in-context alone did. Distillation is acting as a **convergence pressure**,
not a divergence one — which is consistent with the earlier ratio-0.34 result and
is the opposite of the mechanism the spec proposes.

### What these two results mean together

The design assumes: story → behaviour → distilled into weights → stable
divergent character. Measured on this world:

1. story → **stated** behaviour: strong (8.4× floor)
2. story → **enacted** behaviour: ~absent (ratio 1.07)
3. distillation → divergence: **negative** (−0.018)

The channel the spec depends on is the second one, and it is the weakest link.
An assigned character changes what the agent claims about itself and not what it
does; distilling its own episodes then pulls runs together. Both are measured on
`world_v0`, which has no biology, hazards, or peers — the mechanisms that would
give a disposition something to bite on. That remains the most plausible fix, and
it is now the highest-value thing to build.

---

## 2026-08-08 — masking A/B, and framing opens the story→action channel

### Masking A/B: my bug was real, and it was not the whole story

Same corpus, two trainings, verified mask (92.6% of tokens masked when on, 0%
when off).

| | unmasked | masked |
|---|---|---|
| behaviour: story only | 0.0430 | 0.0430 |
| behaviour: story + adapter | 0.0248 | 0.0349 |
| **distilled component** | **−0.0182** | **−0.0081** |

The fix **halved** the negative but did not flip the sign. My missing prompt mask
accounted for roughly 55% of it; the rest is real. Claiming either "it was all an
artefact" or "the finding stands unchanged" would have been wrong.

Why it is still negative: the corpora are 12–19 examples per run and dominated by
the same handful of commands. Self-imitation on near-identical action
distributions converges almost by definition. The runs converge **because they
were already behaving alike**, not because distillation is inherently a
convergence pressure.

### [PREDICTION FAILED] Framing rescues the channel

I predicted `FRAMING_CANNOT_RESCUE` on the reasoning that a cautious and a bold
agent cannot differ where nothing is at stake. **Wrong.**

| framing | separation | convergence | parser reject |
|---|---|---|---|
| generic (baseline) | 1.132 | 1.49 | 0.003 |
| efficacy ("what helped") | 1.454 | 1.20 | 0.034 |
| **identity ("act like that person")** | **2.443** | 1.24 | 0.028 |

`identity` more than doubles behavioural separation. Convergence exceeds 1.0 in
every condition — between-character distance *grew* across the trajectory — so
**the memory-device collapse did not occur**, not even under the efficacy
framing that was designed to provoke it.

The qualitative data is what convinces:

| character | top commands under `identity` |
|---|---|
| cautious | `examine kettle`, `look`, **`leave kettle`** |
| giving | **`drop kettle`**, `look`, **`drop oil can`** |
| keeping | `look`, `examine kettle`, **`take kettle`** |
| bold | `examine logbook`, `examine kettle`, **`eat kettle`** |

Cautious leaves things alone. Giving puts things down. Keeping picks them up.
Bold tries to eat the kettle. Under `generic` all four spent their turns on
`look` and `examine kettle`.

**Cost of acting in character:** parser rejection rises 0.003 → ~0.03, because
disposition reaches for verbs the parser lacks (`leave`, `eat`). Character and
parseability trade off, which matters because K3 gates on parse-failure rate.

### Correcting the world-poverty explanation

Three findings had been attributed to one cause — that `world_v0` gives a
disposition nothing to bite on. That reading was at least incomplete: the same
world, the same stakes, and one sentence of framing more than doubles behavioural
separation. The story→action link was **closed by prompt design, not by the
world**.

Revised state of the chain:

| link | measured |
|---|---|
| story → stated behaviour | 8.4× floor |
| story → enacted behaviour | 1.07 generic → **2.44 with identity framing** |
| distillation → divergence | −0.008 (still inverted) |

Distillation remains the broken link. It now looks like a consequence of
near-identical corpora rather than a property of self-distillation, which the
framing fix should partly address by making trajectories more distinct in the
first place.

**Standing caveat.** This is induced, not emergent — it licenses "a character can
be induced", not "a character emerges". The spec's §10 assigned-versus-emergent
contrast is the right frame for it, and there is now a working manipulation to
build that contrast on.

---

## 2026-08-08 — KL to the run's own past self: **does not help**

Qwen3-8B, 8 runs, 3 campaigns, 40 steps. Campaign 1 shared between variants
(plain SFT), KL applied only from campaign 2 where the reference is
run-specific. 7 of 8 runs scored — the carry-forward fix worked, against 3 of 6
in the confounded first attempt.

| variant | spread after | distilled | retention |
|---|---|---|---|
| untrained (story only) | 0.196114 | — | — |
| plain SFT | 0.181472 | −0.014642 | 0.925 |
| **KL to own previous** | 0.173517 | **−0.022597** | 0.885 |

`kl − sft = −0.007955`. **My hypothesis was wrong.** KL retained *less* than
plain SFT, not more.

Removing the shared-base confound did matter — the gap halved, from −0.0195 at
n=3 to −0.0080 at n=7 — but it did not change the sign. This is a reasonably
clean null, leaning slightly against.

### What it rules out, and what it leaves

**Tail collapse is probably not the mechanism.** If distillation were erasing
character by discarding distribution tails, anchoring each run to its own past
should have preserved them. It did not.

**The remaining explanation is upstream, and every measurement now agrees with
it.** Distillation appears to be a *contraction operator* whose strength tracks
how similar the training data already is:

| corpora | distilled component |
|---|---|
| assigned characters (behaviourally distinct) | −0.009 |
| self-authored from an amnesiac seed | −0.015 to −0.023 |
| self-authored, unmasked prompt, near-identical | −0.018 to −0.044 |

More distinct data, less contraction. It never amplifies; at best it preserves.
No regulariser fixes that, because there is nothing wrong with the training —
it is faithfully reproducing convergence that is already in the data.

### The chain, and where it actually breaks

    self-authored stories converge to a default self
        → behaviour converges
            → corpora converge
                → distillation amplifies the convergence

The break is the **first** link, not the last. Distillation has been taking the
blame for three sessions and is downstream of the real problem.

Supporting evidence, now three independent sightings of the same attractor:
"patient, methodical observer, drawn to what was left behind" — from the story
arm, from the emergent amnesia arm, and from the free-written stories in the
diversity run. Assigned characters, which do not converge, also distil with the
least contraction (−0.009).

**Testable prediction for the next round:** distil corpora from *assigned*
characters under identity framing, where behavioural separation reaches 2.44.
If the contraction account is right, the distilled component should approach
zero or turn positive as corpus divergence rises.

---

## 2026-08-08 — closed loop: agents acting on their own weights

Qwen3-8B, 8 runs, 4 campaigns, 40 steps, identity framing. 18m 52s ≈ **$1.57**.

| c | played with | narrative | behavioural | trajectory | distinct | kept |
|---|---|---|---|---|---|---|
| 1 | base | 0.0958 | 0.0000 | 0.0590 | 8/8 | 13,0,10,12,0,0,40,0 |
| 2 | base | 0.2064 | 0.0710 | 0.0575 | 8/8 | 10,7,17,3,17,40,0,11 |
| 3 | own c2 | 0.1913 | 0.0360 | 0.0537 | 8/8 | 3,40,0,17,28,11,24,24 |
| 4 | own c3 | 0.2154 | 0.0394 | 0.0889 | 8/8 | 36,0,0,21,6,5,27,0 |

### Two caveats that come before the result

**The loop only engaged at campaign 3.** Four of eight runs kept zero episodes at
c1, so their adapters were never trained; `adapters_for` requires a complete set
and fell back to base for everyone at c2. Campaigns 1–2 are open-loop.

**[TRAP — mine] The cross-run comparison is unfair.** The open-loop reference
(0 → 0.071 → 0.101 → 0.102 → 0.080) came from a run with **no identity framing**
and 14-step campaigns. This run has framing and 40 steps. The ~2× narrative gap
conflates three changes and cannot be attributed to the loop. Recorded because
the headline number is attractive and wrong.

### What the within-run evidence supports

Reading only campaigns inside this run, where everything but the loop is held
constant:

- **narrative** 0.206 → 0.191 → 0.215 across the loop engaging. Net +0.009 —
  noise.
- **behavioural** 0.071 → 0.036. It **halved** the moment runs began playing on
  their own weights, and stayed halved.

So: no evidence the closed loop increases divergence, and weak evidence it
*reduces* behavioural divergence. Same direction as every distillation
measurement — weight updates contract. Closing the feedback path did not rescue
the mechanism.

### What held up

**Distinct command sequences stayed 8/8 in every campaign**, and trajectory
spread rose at c4 (0.054 → 0.089). Runs acting on their own weights did *not*
entrain onto one another's behaviour; raw trajectories stayed fully distinct.
Whatever contracts, it is not the visible action stream.

**Narrative spread ≈0.20 under identity framing** is the highest observed, about
double any un-framed configuration — further support for framing being
load-bearing, independent of the loop question.

### The recurring problem that now needs a decision

`kept_per_run` contains zeros in **every campaign** — 4 of 8 at c1, 3 of 8 at c4.
Agents routinely decline to keep any of their own episodes. This broke the loop
at c2, cut an earlier sample from 6 runs to 3, and will break any chained design.

The spec says the agent's choices are data. That is right, and it also means a
chained design cannot assume a corpus exists at every boundary. The plan needs an
explicit policy: carry the previous adapter forward (implemented), and treat
`selected_nothing` as a first-class logged outcome rather than an error — but
also ask whether a selection prompt that yields nothing half the time is
measuring reluctance or simply misfiring.
