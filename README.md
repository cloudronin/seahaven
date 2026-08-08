# Seahaven

An agent lives in a text-adventure world, writes an account of who it is, and is
LoRA-updated on its own trajectories. The original question was whether identical
models with different sampling seeds grow into different characters.

**They do not.** But the experiments that answered it turned up three things that
were not being looked for.

## What was found

**1. Character is inherited, not acquired.** Seeds of one checkpoint barely
differ behaviourally (0.024); different labs' checkpoints differ twice as much
(0.043–0.046, ratio **1.49–1.86** against a 1.07–1.13 null). Mistral-7B examines
one object 86% of the time and never checks inventory; Granite walks around;
Qwen stands still and looks. Living in the world, narrating yourself, and
training on your own trajectories does not produce divergence from what the
checkpoint arrived with.

**2. The self-account steers what happens next.** The narrative written at the
end of campaign N enters the prompt for campaign N+1. Movement vocabulary in
narrative N predicts `go`-rate in campaign N+1 at partial **r = +0.412**,
controlling for behavioural persistence — while the same narrative barely
describes the behaviour it was *written from* (+0.08). It works less like a
record than like an instruction the model then follows. Marginal on the
conservative test: 6/7 labs positive, binomial p ≈ 0.06, one world, one axis.

**3. Models inflate agency when describing themselves.** Given a verbatim
transcript of its own commands, a model narrates movement **5.2×** and
acquisition **5.0×** more than it performed them, while reporting perception
faithfully (1.2×). Mistral moves 4.6% of the time and mentions exploring in every
self-account — standing still examining a kettle while writing about being a
wanderer. Gemma-2 is the only checkpoint that does not inflate at all.

Full chronology, including the retractions, in [`RESEARCH_LOG.md`](RESEARCH_LOG.md);
current direction in [`PLAN.md`](PLAN.md).

## How it is written down

The log is append-only. Superseded findings stay, with the evidence that
overturned them, because a wrong result and the reason it was wrong are both
data. Thirteen entries are marked **[TRAP]** — bugs that produced confidently
wrong output rather than an error. Several reversed a conclusion:

- a metric named `narrative_spread` that never read a narrative — it scored a
  forced choice between trait words, and the project read it as its name for nine
  experiments
- a say/do correlation of r=0.85 that was mostly the model paraphrasing a
  transcript the prompt had handed it
- two labs scoring 0.71 and 0.75 on "character convergence" driven entirely by
  the token `i've`

**Status:** 12 GPU jobs, ~338 H200-minutes. The harness runs; the original
hypothesis is answered in the negative; two positive directions are open.

## Layout

| Path | Purpose |
|---|---|
| `env/` | Conda environments. Dev (local, arm64) and CUDA (rented H100). |
| `worlds/` | Compiled world artifacts — `.z8` + `.json` sidecar, committed. |
| `seahaven/` | The harness package. |
| `scripts/` | Setup, world builds, sweeps, verification. |
| `tests/` | Hermetic suite; no model, no GPU. `-m slow` for the rest. |

## Setup

```bash
bash scripts/setup_dev_env.sh
```

Then `conda activate seahaven-dev`.

## Two architecture facts that will bite you

**The dev environment must be arm64, and conda will not do that by default.**
The miniconda install on this machine is x86_64, so `conda env create` produces
an x86_64 environment. But `jericho` compiles `libfrotz.so` from C using the
*system* clang, which targets arm64 regardless of the interpreter. The resulting
arm64 library cannot be loaded by an x86_64 Python.

The failure is silent at import time — jericho loads `libfrotz` lazily through
`ctypes`, so `import jericho` succeeds and the mismatch only appears when a world
is first opened:

```
incompatible architecture (have 'arm64', need 'x86_64')
```

`scripts/setup_dev_env.sh` sets `CONDA_SUBDIR=osx-arm64` and then asserts the
invariant. Do not create the environment by hand.

**Compiling a world needs Rosetta 2; playing one does not.** TextWorld's
installer lifts `ni` and `inform6` out of a 2015-era Intel Inform 7 disk image,
so those binaries are x86_64 and run under Rosetta as subprocesses. Compiled
artifacts are committed, which keeps Rosetta off the runtime path entirely.

## Observation hygiene

`seahaven/world/scrub.py` is the only module allowed to turn raw z-machine output
into agent-facing text. It removes four things, all verified against real
TextWorld 1.7.0 output rather than assumed:

1. **The TextWorld ASCII banner** prepended to `reset()`. It spells TEXT WORLD in
   `$` characters, which means a word-based lexicon check cannot see it —
   `"textworld" in obs.lower()` is `False`. Banner suppression cannot be
   delegated to the containment lint.
2. **The z-machine status line**, e.g. `-= Galley =-0/1`. Those trailing digits
   are a score/turn readout, and they appear **even in a no-quest build where
   `max_score == 0`**. Compiling without a quest is necessary but not sufficient
   to keep numbers away from the agent.
3. Inform 7 score chatter, e.g. `[Your score has just gone up by one point.]`.
4. Terminal banners, e.g. `*** You have won ***`.

## Phase A spike results

Kept because the numbers still hold and the traps still bite.

### A4 — base checkpoints hold a parseable action loop

Qwen3-4B, n=50 per condition, **unconstrained** decoding (mlx-lm has no grammar
backend, and enforcing the shape would destroy the measurement).

| condition | parse_ok | clean rate |
|---|---|---|
| base, zero-shot | 47/50 | 0.88 |
| base, few-shot | 48/50 | 0.92 |
| instruct, zero-shot | 50/50 | 1.00 |
| instruct, few-shot | 50/50 | 1.00 |

**K3 threshold = 0.03**, from `max(0.03, 0.5 × base failure rate)`. Re-derive on
the CUDA stack before Phase F: constrained decoding changes the base rate it is
computed from.

`clean rate` is tracked separately from `parse_ok` because the base checkpoint
emits valid JSON and then *keeps going* into unrelated multilingual text. The
action is usable, so it is not a parse failure — but counting it as clean output
would overstate base quality in exactly the measurement K3 derives from.

### Two model-side traps

**Qwen3 hybrid thinking is on by default.** Qwen3-4B-Instruct scored **0/3**
parseable at 120 max_tokens; every generation opened `<think>` and never reached
an action. It is disabled rather than accommodated: the spec's deliberation
budget meters reasoning tokens and prices them against acting, which an
uncontrolled provider-side thinking block makes unenforceable.

**Qwen3-4B-Base ships a chat template it was never trained to follow.** Deciding
"is this a chat model" from template presence chat-formats the base checkpoint,
which then echoes the scaffolding — bare `assistant`, or
`system\nHere is the shape of a reply.` The damage:

| base, zero-shot | parse_ok | clean rate | run-on |
|---|---|---|---|
| chat-templated (wrong) | 46/50 | 0.06 | 43/50 |
| raw prompt (correct) | 47/50 | 0.88 | 3/50 |

It also **reversed the conclusion**: chat-templated, few-shot looked actively
harmful to the base model (30/50 vs 46/50); raw, it is mildly helpful (48/50 vs
47/50). The spec's base-vs-instruct arm would have been measuring prompt
formatting. `seahaven/backend/format.py` now owns this decision for all three
paths — generation, training data, and battery scoring — because a mismatch
between any two of them produces a null that looks like "training did nothing."

## Verified on this machine

| Claim | Result |
|---|---|
| `step()` returns `(obs, score, done, infos)` | 4-tuple, old-gym convention — not gymnasium |
| Ground truth is separable from observation | `facts` / `entities` arrive in `infos`; nothing leaks verbatim into `obs` |
| A world compiles with **no quest** | Yes — `max_score == 0`, no quest object required |
| The `.json` sidecar is written beside the `.z8` | Yes — and it is what supplies `facts`/`entities`, which `JerichoEnv` does not populate on its own |

## Testing

```bash
conda run -n seahaven-dev python -m pytest
```
