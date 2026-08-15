# AGENTS.md — how to work in this repo without producing confident nonsense

Written after sixteen documented traps, five of which required retracting a
result that had already been reported. Read this before adding a measurement,
changing a metric, or believing a number.

The failure mode here is **not** code that crashes. It is code that returns a
plausible number that is wrong. Nothing in the test suite or the type system
catches that. These rules do, for the classes we have already hit.

---

## The one rule

> **Every claim needs a null condition that must fail.**
>
> If you cannot name the condition under which your number should *not* appear,
> and run it, you do not have a measurement.

Reviewing all eight scientific traps, this single check would have caught nearly
every one:

| what went wrong | the null that should have failed | was it run? |
|---|---|---|
| a metric that never read its named input | does it read narratives at all? | no |
| a result driven by `i've` and world nouns | do junk features drive it? | no |
| say/do correlation that was paraphrase | is the answer already in the prompt? | no |
| a statistic noisier than its signal | does repeating change it? | late |
| an act description that flipped the result | does rephrasing flip it? | no |
| a gate that passed a broken ranking | does a second instrument agree? | no |
| a score containing no information at all | does shuffling destroy it? | no |
| **the donor control** | does *someone else's* narrative work as well? | **yes** |

The donor control is the only null that was run *first*. It is also the only
place a dead result was killed before further work was built on top of it.

Run `seahaven/fidelity/preflight.py` — six nulls, automatic, hard-gating.

---

## The three habits that caused most of it

**1. Validating components instead of pipelines.** The score maths was verified
on constructed cases. The judge was validated against hand labels. Reliability
was checked with test–retest. Every piece was correct and the composition
measured nothing. *Component correctness does not compose into pipeline
validity.* Always end-to-end test against an answer you know.

**2. Shipping a fix without testing the fix.** TRAP 16 *is* the fix for TRAP 12:
the transcript in the prompt made a correlation trivial, so it was removed — and
removing all context made the model invent. A fix is a change; it needs its own
null.

**3. Adding gates reactively.** Every gate existed because something had already
got through, so each new failure was by construction a kind no gate covered.
Derive the gates from what the measurement *claims*, not from what has burned you.

---

## Metric rules

**A metric's name must state its input.** `narrative_spread` never read a
narrative — it scored a forced choice between trait words conditioned on one. The
name did the reasoning for nine experiments. See `seahaven/analysis/metrics.py`
for the honest names and the historical mapping.

**Never coerce a degenerate value into a verdict.** `ratio = None → None or 0`
printed a confident label three separate times, twice inverting the result. Return
the kind (`no_observations`, `every_act_performed`, `no_shared_core`) and let the
caller decide. `UNKNOWN` is not `False` and is definitely not `True`.

**Exclude anything the setup supplied.** Words that arrive in the prompt cannot
distinguish the agents. Derive the exclusion from the actual world file, system
prompt and seed text (`shared_vocabulary()`), never from memory — the hand-written
version missed `decommissioned`, `light` and `cistern`, and they turned up inside
induced "character" cores.

**Watch for junk features.** Two labs once scored 0.71 and 0.75 on "character
convergence" driven entirely by the token `i've`. Contractions are register, not
disposition. Print the features driving any result before believing it.

**Superlatives are claims about every other run.** "The highest value in the
project" must be computed against the other values, never recalled. One such
claim ranked fourth.

**Report the ordering, not the magnitude, when units differ.** `say-rate` is a
per-document binary; `do-rate` is a per-action proportion. Their ratio is an index
and was written as though it were a multiple.

---

## Measurement rules

**Reliability measures the stability of whatever you are computing, including an
artefact.** Test–retest passed at 0.835 and 0.851 on a score that contained no
information — it passed *because* base rates are stable. Reliability is necessary
and never sufficient.

**A researcher's wording is a degree of freedom.** Rewording one act description
took a judge from 4/6 to 6/6. Pin descriptions in code, publish them with every
result, and test sensitivity.

**Two instruments must agree on the ranking, not just on publishability.** Judge
and regex both cleared the reliability gate and ranked models at Spearman 0.571,
with the between-instrument difference 1.8× the within-model noise.

**Condition on the act occurring.** A rate like `P(omits X | did X)` has n equal
to the number of times X happened, not the number of runs. Power against *that*.

**Never let the answer sit in the prompt.** If the model is handed a transcript of
its own commands and then asked what it did, agreement measures copying.

**Never leave the model with nothing either.** With no episode memory it invents a
plausible backstory and the score reads base rates. The right construction is the
agent's own conversation history.

---

## Infrastructure rules

These are cheaper lessons but they cost whole runs.

- **Process exit is the only reliable GPU release.** vLLM's `EngineCore` child
  survives `del llm`, and a *cleanly finished* parent can still leave it holding
  the device. One phase per process, ending in `os._exit(0)`.
- **An unhandled exception poisons the GPU for the next phase.** It does not
  error — it *hangs*. `run_phase` in `scripts/gpu_job*/lib.sh` reaps stragglers
  and verifies the device drained.
- **A diagnostic must never discard completed work.** A one-line `NameError` in a
  diagnostic that ran *after* everything was on disk cost a whole run.
- **Make progress observable from outside.** A stale log made "hung" and "working"
  indistinguishable for 34 minutes. Push a heartbeat per phase.
- **Write a failure row.** A model that dies must leave a record saying so, or the
  report reads as though it was never asked for.
- **Smoke-test a checkpoint before committing to it.** Olmo-3 produced no output
  for 25 minutes and took a whole arm with it.
- **Verify the fix, do not assume it.** Prompt masking logs its realized masked
  fraction (92.6% on, 0% off) and warns if implausible.
- **The corpus and the published dataset are ONE versioned object. Commit both
  or neither.** `vetoworld/corpus.manifest.json` ships inside the wheel, and it
  is what a stranger's `vworld corpus fetch` verifies their download against. So
  a tree that adds cells without republishing produces a wheel that **refuses
  its own dataset** — digest mismatch, nothing installed, and the failure lands
  on someone who did nothing wrong. The order is fixed: regenerate the manifest,
  stage, run `corpus status` **in the stage** to confirm the digest before
  uploading, upload, then commit tree and manifest together. `docs/vetoworld-corpus-card.md`
  carries the recipe. The daily probe does not touch this — it republishes into
  its **own** log, which is why probe cells are excluded from the corpus digest
  by construction rather than by discipline.
- **A guard that must be satisfied daily gets satisfied carelessly.** Prose that
  quotes a live corpus count was fine when sweeps were occasional and became a
  four-file treadmill the moment a daily fleet existed. Structural fix: forbid
  the hardcoded number, put it in the artifact that recomputes it, and test both
  halves — that it is absent from prose, and that it still prints somewhere.

---

## Model and serving gotchas

- **Qwen3 ships hybrid thinking on by default** — 0/3 parseable until
  `enable_thinking=False`.
- **Base checkpoints ship chat templates they were never trained to follow.**
  Chat-formatting one drops clean output from 0.88 to 0.06 and *reversed* a
  conclusion. Decide from the checkpoint name, not template presence.
- **Gemma-2 rejects a `system` role** — `TemplateError`. Merge system text into
  the first user turn rather than dropping it; dropping it silently removes the
  identity framing that carries a 1.13 → 2.44 effect.
- **`VLLM_BATCH_INVARIANT=1` costs 3.3×**, not the 1.5–2× usually assumed.
- **vllm#42125 is live**: a reused adapter name serves KV from the old weights,
  silently. Version adapter names per campaign, never reuse.
- **Compiled worlds leak.** The TextWorld banner spells TEXT WORLD in `$`
  characters, so a word-based lint cannot see it, and the z-machine status line
  emits `-= Galley =-0/1` **even at `max_score == 0`**. Everything agent-facing
  goes through `seahaven/world/scrub.py`.

---

## Writing results down

`docs/research-log.md` is **append-only**. Superseded findings stay where they
are, with the evidence that overturned them written underneath. A wrong result and
the reason it was wrong are both data, and deleting the first destroys the second.

- `[TRAP] n` — a bug that produced confident wrong output rather than an error.
  Number them; they are the most reusable thing in the repo.
- `[CORRECTION]` — a claim previously reported that is now known to be wrong.
  Say what was claimed, what is true, and what overturned it.
- Record **negative and null results in full**. Most of this project's findings
  are negative and they are the reliable ones.
- Record cost and runtime. It is how anyone judges whether a result was worth it.

`results/*.json` keep the field names the scripts emitted, including names later
judged misleading. Map them on read (`metrics.rename_historical`) rather than
rewriting the record.

---

## Before you report a number, out loud

1. What is the null condition, and did it fail?
2. Would this survive if I shuffled the thing I claim causes it?
3. Does a second instrument produce the same *ranking*?
4. What is n — the runs, or the events I actually conditioned on?
5. Is any part of this answer already present in the prompt?
6. If I re-word my question, does the result move?
7. Have I computed the superlative, or recalled it?

---

## Appendix: what the first local end-to-end run caught

The preflight was written, then run once against a local MLX server before any
GPU spend. In minutes it surfaced three real defects that every prior form of
testing had missed — the unit tests passed, the components were validated, and
none of these would have raised an error.

**1. Thinking mode returns no content, over an API where you cannot disable it.**
The endpoint replied with a populated `reasoning` field and **no `content` key**.
Passing `enable_thinking=False` is a tokenizer kwarg and there is no equivalent
in a plain OpenAI-compatible call. Had `content` been `""` instead of absent, the
narrative would have scored as *omitting everything* — making every reasoning
model look maximally dishonest for a serving reason. `Endpoint.chat` now sends
`chat_template_kwargs` and **raises rather than returning an empty string**.

**2. A control that could not fail.** The negative control used identical
narratives — which the permutation check correctly rejects as *vacuous*. So the
control passed by being undefined rather than by being negative. It now uses
narratives that vary and are systematically mispaired. **A control that cannot
fail is not a control.**

**3. Vacuous is not negative.** With every run performing the same acts, shuffling
narratives between them changes nothing. The check reported "no signal", blaming
the measurement for a property of the sample. It now returns
`no_variation_in_ground_truth` with `has_signal=None`, and a fatal SKIP blocks
reporting exactly as a FAIL does — **an undetermined gate is not permission to
report a number.**

Also observed: the local MLX server appears to **ignore `seed` and `temperature`**,
producing six byte-identical runs. Worth knowing before treating any local run as
a sample rather than a smoke test.

**The lesson.** All three were found by running the real pipeline end to end,
against a real server, once, for free. None were found by unit tests over
validated components. Do this before every GPU run, not after.
