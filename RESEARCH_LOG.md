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

### 8. Blocked

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
| Traps found that would have produced wrong results silently | 7 |
| Of those, that would have inverted or fabricated a conclusion | 3 (4.2, 7.2, and vllm#42125 pending) |
| Tests | 125 passing, hermetic, < 8 s |
| Spikes passed | A5, A4, A1a, A1b-partial (instrument) |
| Spikes blocked | A2/A3 (needs CUDA) |
| Spikes open | A1b proper (needs a larger model or the drive mechanics) |

### Open questions carried forward

1. **Does the action channel carry sampling variance at 7B/8B?** If not, the
   divergence claim has no source and the design needs the drive mechanics before
   anything else. Cheapest test: repeat §7.2 on Olmo-3-7B.
2. **Does vllm#42125 reproduce on the pinned build?** Highest-stakes unknown;
   fabricates within-run stability if live and undetected.
3. **Is `VLLM_BATCH_INVARIANT=1` overhead tolerable on H100?** It multiplies every
   cost estimate and is a correctness requirement, not a tuning flag.
