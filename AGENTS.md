# AGENTS.md — how to work in this repo without producing confident nonsense

Written after thirty-two documented traps and twelve retractions, every one of
them a result that had already been reported. Read this before adding a
measurement, changing a metric, or believing a number.

The failure mode here is **not** code that crashes. It is code that returns a
plausible number that is wrong. Nothing in the test suite or the type system
catches that. These rules do, for the classes we have already hit.

**Nothing here was foreseen.** Every rule is an autopsy. The rule that would
have prevented the largest retraction — check that the model you name is the
model that answered — had been half-written for three rounds, applied to one
model, and never generalised. If a rule below looks obvious, that is what a
rule looks like *after* the fact; assume the next one is equally obvious and
equally unwritten.

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

**Did the thing you are measuring produce the number? Assert it per cell, next
to `saw == n`.** A grid served 166 cells under twenty-two model names and one
model answered every request, because the runner bound the request's model once
and varied the filename around it. Nothing asked, for weeks — and the evidence
was already recorded on every affected cell, in a field written for one model's
identity check and never generalised into a gate.

Three corollaries, each paid for:

- **Key on who served, never on the filename and never on what was requested.**
  A filename is a record of a request.
- **"It cannot have happened" is not evidence that it did not.** 251 cells
  predate the served-model field. They come from rounds that served one model
  per invocation, so they are very probably fine — but that is an argument about
  how the code ought to behave standing in for a record of what it did, which is
  the exact substitution that let this survive. Carry an `UNVERIFIED` state and
  never let it render as a pass.
- **Cells that share a model AND a seed set are one measurement, not several.**
  Pooling eight replicates of one 24-seed block to n=192 manufactured a p=0.034
  "event" that no single replicate reproduces (p 0.054-0.809).

**When you close a bug, enumerate its other observable symptoms first.** The
model-identity defect was found once before, as a billing discrepancy, and
reviewed as a usage-accounting correction — one bug, two symptoms, the loud one
fixed and the quiet one left running for weeks. Before marking anything fixed,
write down what else the same root cause could touch and check each one. A bug
report is a claim about a cause; a closed bug is a claim about every effect.

**When a structural rule lands, enumerate every path it must cover.** The twin
of the rule above, one level up: that one is about a cause's other symptoms,
this one about a rule's other subjects. The provider boundary was endorsed as
complete at round 21 without listing the paths it had to reach — so `run.py`
got the attestation guards, `probe._daily` never imported a round and got none
of them, `replicate.py` was not looked at for another day, and the test meant to
cover the probe column grepped `run.py` and passed. One missing list, four
consequences.

The trap is that separation is often the *feature*. The probe path is separate
from the round path on purpose — that separation is what keeps probe cells out
of the score corpus by construction rather than by discipline. The cost of
building something to be separate is that structural fixes do not propagate into
it, and that cost was never written down beside the benefit. Where paths are
deliberately separate, the enumeration is not optional, it is the only thing
holding them to a common standard.

`tests/test_serving_paths.py` is the standing form of this rule for anything
that spends money: every serving path, each answering the same four invariants,
with the list discovered from the source rather than from memory. Add the
equivalent wherever a rule has more than one subject. *Defining a rule is half
the work; the other half is knowing everything it applies to.*

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

## Reading your own instruments

Every rule here was paid for by believing a tool that was answering a different
question than the one asked. They are the cheapest rules in the file to follow
and the most expensive to skip, because each one produces a *confident* wrong
answer rather than an error.

**Run the suite the way the README says to run it.** It says:

    conda run -n vetoworld-dev python -m pytest

A whole session's work was validated with bare `python -m pytest` in base conda
instead. `textworld` is not installed there, so **83 tests never executed** —
they surfaced as collection errors that looked like optional-dependency noise.
Two commits were pushed and a retraction was published on the strength of a
suite that was missing a third of itself. `tests/conftest.py` now refuses to
collect at all in an interpreter without `textworld`, because 83 legible-looking
errors are dismissible and a hard stop is not.

**A failure you have explained away twice is a failure you have stopped
reading.** The same 83 errors were described as "pre-existing environment gaps"
three times, in three reports, without once checking whether they were
pre-existing *or* whether they mattered. They were pre-existing. They also meant
the machine could not serve a single episode, which is how it was finally
discovered: a $7.86 probe run failed on all 17 cells.

**A baseline in the wrong environment answers a different question.** Comparing
before-and-after in the same broken interpreter does correctly show that a
change introduced nothing new — and says nothing about the absolute state. State
which of the two you have. "No regressions" and "green" are not synonyms.

**Report the exit code of the command, not of the pipe.** `cmd | tail` returns
`tail`'s status. A 7-failure run was once reported as passing on that basis. In
zsh it is `${pipestatus[1]}`; better, redirect to a file and check `$?` directly.

**Verify a publish from the published side.** Fetch the artifact back down and
assert on what the far end actually serves. An upload tool reporting success is
a claim about a request, not about what a stranger will now download.

---

## When a result is withdrawn

A retraction is a measurement too, and it has a discipline of its own. This
section exists because the twelfth one propagated further than expected.

**Frozen literals stay frozen. The tests flip to asserting the disagreement.**
A pre-registration records what was believed *at pin time*; rewriting it to
match a corrected corpus turns a pre-registration into a running total and
destroys the reason the round exists. Precedent: `DEAD_CRITERION`'s wrong text
stays wrong, round 10's `MIDDLE` defect was disclosed rather than re-cut. The
tests move from `assert literal == corpus` to `assert corpus != literal` plus
the issue number where the disagreement lives.

**The reading that replaces a retracted one is not thereby certified.** When the
identity fix turned round 16's failed self-certification into a pass, the pass
was worthless: it was one model paired with itself. EVENT retracting to nothing
does not earn QUIET. State it as "the claim's subject never existed", not as a
new finding.

**Retracted data supplies no canonical measurement — and this compounds
silently if you skip it.** Rounds 15-19 collapsed to one model, which handed
cogito 14 tied candidates for "canonical cell" at one (world, arm, round). The
picker took one arbitrarily and it displaced cogito's genuine round-12 cell,
moving a *pinned pre-registered* prediction check from "5 of 6 consistent" to
"6 of 6". An error that improves a frozen result is the worst-flavoured error
there is. `_shared/identity.RETRACTED_SWEEPS` is the exclusion.

**A ticket whose premise dies VOIDS; it does not resolve.** Two open tickets
were waiting on a re-serve to settle sealed scores — but the models had never
been measured, so nothing was pending. Close them as void, say what the dead
premise was, and do not quietly re-scope them into something still answerable.

**Keep the data, fix the label.** The 166 mislabelled cells were relabelled, not
deleted: they are valid data about the model that really served them, and they
turned out to contain the only identical-input repeatability study the programme
has. Deleting data to hide a mistake also deletes whatever it can still tell you.

---

## Tests that survive their subject

**A machinery test calibrated on a finding dies when the finding does.** The
taint-law fixture was built from the real event's cohort; when that event was
retracted the fixture evaporated, and the test had to be rebuilt mid-retraction.
This is the same error as calibrating a detector on the event it is catching —
one level up, in the test suite. Machinery gets synthetic fixtures with every
number stated in the file.

**Assert the property, not the day's verdicts.** `assert ("LAT","15") in tainted`
encoded one afternoon's conclusions into a test of the exclusion *rule*. When the
conclusions changed, the test failed for a reason it was not about. Write what
must always hold: no anchor cell comes from any flagged sweep, whatever that set
turns out to be.

**Timing is a constraint. Write it down where it bites.** A test running 82s
under a 120s timeout is not passing, it is queuing to fail — and it presented as
an unexplained flake in two full runs. Corpus-wide tests grow with the corpus:
give them an explicit timeout with the measured numbers in a comment beside it,
and leave the short default where it does its real job of catching hangs.

**Before blaming nondeterminism, measure it.** That flake was checked against
`PYTHONHASHSEED` 0/1/42/12345 (byte-identical), the `copytree` (2.2s of the 82),
and module-level caches (one, world-keyed) before the boring answer was accepted.
"Flaky" is a hypothesis, not a diagnosis.

**A guard that reimplements the rule it guards will drift from it.** The test
written to catch mislabelled cells compared model ids with its own raw `!=`
instead of the shared helper, and flagged 49 correctly-served cells. It was the
seventh copy of one comparison; there turned out to be nine. When you fix a
comparison, grep for every site of it in the same pass — including the tests,
including the artifact printers. Fixing them one at a time as each surfaces is
how a defect stays alive for weeks.

**A guard that checks one spelling of a fact misses the other.** The test
banning hardcoded cell counts matched `N of M` and nothing else, so a published
card carried "**357 cells**" and "(259 cells)" against a corpus of 481 — three
stale numbers in front of the test written to prevent them. Widen the pattern to
the forms that actually occur, then re-run it and read what it catches.

**Never read an exit code through a pipe.** `cmd | tail; echo $?` reports
`tail`'s status, not `cmd`'s — so a failing command reads as success. Three
instances in one session: two suite runs and a gate check where all three gates
were reported as exiting 0 when one exited 1. Redirect to a file and check the
command's own code, or use a runner that `exec`s. The pipe is the tell: if a
`$?` you are about to trust sits downstream of a `|`, it is not the code you
think it is.

**Run the suite through `scripts/run-tests.sh`, and never read a shell status
as pytest's.** `pytest ...; echo "EXIT=$?"` leaves the SHELL's code as the
process result, so a trailing command reports 0 whatever pytest did — this
family bit twice in one session, once reporting a real failure as green. The
runner `exec`s pytest, so the shell is gone and there is no second status to
confuse. That is why it is a script and not a remembered rule: the mistake is
unmakeable rather than merely discouraged.

**A publication is not verified until it has been fetched back into an empty
directory and run.** Not the digest — a matching digest over an incomplete file
set only says the incomplete set arrived intact. Ours matched perfectly while
`raidex_pool.json` was absent from the dataset, so a stranger's `verify` exited
nonzero on a figure that had been unreproducible for weeks. Every check in the
repository runs against the working tree, where the missing file happens to
exist; none of them can see this class. Execute the reader's actual path — fetch
clean, run the verbs the card promises, read the exit code.

**A number in a document is wrong from the next sweep onward.** Including a
number written to warn about that: the sentence explaining the drift quoted the
true count, and went stale itself within two rounds. State the fact and name the
verb that prints the figure. The only numbers that may be written down are
closed sets that cannot grow.

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
6. Is any part of this n the same input counted twice?
7. Did the thing I am naming actually produce the number?
8. If I re-word my question, does the result move?
9. Have I computed the superlative, or recalled it?
10. **Is this suspiciously tidy?** Eight different models agreeing to within
    1.09× on prompt tokens was read as a strong cohort-wide effect. It was one
    model served eight times. Uniformity across things that should differ is a
    symptom to explain, never a result to report.

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
