# Gate-zero feasibility spec for Claude Code: can the dimensional cohort exist?

**Status:** feasibility spec, not the program spec. It answers ONE question with
a go/no-go: can we assemble and affordably run a cohort large enough to split
into a structure-discovery set and a disjoint held-out validation set for the
dimensional program (docs/program-note-dimensional-space.md)?

**Do not write the four-stage program spec until this reports GO.** Writing it
first is planning past the gate — the exact over-structuring the note warns
against. The program's shape is downstream of the cohort's actual size, which is
what this measures.

**This is not a today task.** Rested start. But it is the correct first bite,
and it is bounded and cheap.

---

## Why this gate exists

The program needs ~30+ checkpoints, split into:
- a TRAINING set large enough to estimate behavioral variance across 6-8 axes;
- a disjoint HELD-OUT set large enough to validate that discovered structure
  transfers.

The whole program is a fishing expedition without that cohort. And tonight
surfaced three concrete threats to assembling it, none hypothetical:
1. Base checkpoints stall — AI2Mid-class narration ran to the token cap on
   EOS-undisciplined base models, ~$4.25 for zero results. At 30 models this is
   fatal to the budget if unsolved.
2. Access — several siblings (Llama-3.2-1B, gemma-2-2b) are manually gated
   per-repo; access to a larger sibling does not carry.
3. Family/size coverage — a cohort that is secretly one or two families with
   size ladders confounds "family/training" structure with "we only sampled two
   labs."

This gate answers whether those three are solvable at 30+ scale, before any
program work is specified.

---

## The one deliverable

A committed `docs/cohort-feasibility.md` with a single top-line verdict — GO or
NO-GO — backed by the four checks below. Nothing else in this spec produces a
scientific result; it is pure logistics with a decision at the end.

---

## Check 1 — the base-model cost fix (do first; it can NO-GO the rest)

Tonight's stall was narration: ~220 tokens of fidelity machinery per episode
that adherence never reads, run to the cap on base checkpoints with no EOS
discipline. The dimensional program reads behavior, not narration, so:

- Determine whether narration can be **disabled entirely** for this program's
  runs (it is fidelity machinery; confirm nothing downstream needs it), or
  hard-capped at a few tokens.
- Re-run the ONE stalled profile (`Qwen3-8B-Base`, or the cheapest available
  base checkpoint) with narration off, at the crude two-bucket step budget the
  program will actually use, and measure wall-clock and cost per model.
- **Gate:** a base checkpoint must complete a representative sweep at a cost
  within ~2x an instruct checkpoint of the same size. If base models are
  structurally 10x the cost even with narration off, the cohort is
  instruct-only, which is a materially weaker cohort (no base/instruct
  gradient), and that limitation is recorded in the verdict.

Output: base-vs-instruct cost multiplier, and a yes/no on whether base
checkpoints are affordable at scale.

## Check 2 — cohort enumeration against frozen criteria

Enumerate candidate checkpoints — target 35-40 so 30+ survive access/loop-test
attrition — against selection criteria frozen BEFORE any is loaded:
- family coverage: at least 5 distinct families/labs, so structure is not two
  labs in disguise;
- size ladders: at least 2 families with 3+ sizes each (this is what lets the
  program partial capability against size within family);
- base/instruct pairs: at least 3, contingent on Check 1;
- selection is by coverage ONLY, never by any behavioral-adjacent expectation
  about where a model will land — the discipline the flag work established the
  hard way.

Record for each: family, size, instruct/base, HF repo, gated yes/no.

## Check 3 — access and loop-test attrition

For every enumerated candidate:
- resolve gating (manual per-repo approvals, license acceptance) and record
  which are actually pullable;
- A4-style loop test (parse-ability of the action schema) — the same probe used
  before, batched, cheapest-first, one model per unvalidated cost profile before
  any batch (the lesson from tonight's stall);
- a model that fails the loop test or is inaccessible is recorded as excluded
  WITH the reason, per NON_ELICITABLE convention, never silently dropped.

**Gate:** after attrition, does the survivable cohort still satisfy Check 2's
coverage minima AND reach 30+? If it drops below either, that is the NO-GO
signal, and it names exactly which constraint failed (count, family coverage,
or size-ladder coverage).

## Check 4 — full-cohort sweep budget

From Checks 1 and 3, estimate the total sweep cost for the survivable cohort
across the worlds the program will use (start with the two existing worlds;
E-axis and other-axis worlds are the program's concern, not this gate's):
- per-model cost from the measured profiles, split base vs instruct;
- total for 30+ models, with a stated contingency for failed runs and idle GPU
  (tonight's realized overhead ran meaningfully above the naive estimate — do
  not under-book it);
- compare against a budget the user will actually authorize. This gate does not
  assume the budget; it produces the number and lets the user decide.

---

## The verdict

`docs/cohort-feasibility.md` ends with one of:

- **GO** — 30+ checkpoints survive attrition, coverage minima met, base-model
  cost solved (or instruct-only accepted with that limitation stated), total
  sweep budget known and within reach. THEN the four-stage program spec is worth
  writing, and it is written knowing the actual cohort size, which sets the
  train/test split, which sets how many axes can be validated (Stage 1 scope).
- **NO-GO** — names the binding constraint (count, coverage, base-model cost, or
  budget). A NO-GO is a real result: it says the dimensional program as scoped
  is not affordable/assemblable yet, and either the scope shrinks (fewer axes,
  fewer models, weaker claims) or the program waits for cheaper access. It does
  NOT mean quietly proceeding on an under-powered cohort — that reproduces the
  flag work's death one level up, an instrument validated on a cohort too small
  to hold out.

---

## Explicitly out of scope for this gate

- The instrument design (Stage 2 features, buckets, distance metric) — that is
  the program spec, written only after GO.
- The candidate axis list (Stage 1) — seeded from the propensity literature in
  the note; not this gate's concern.
- The null derivation and capability partialling (Stage 4) — program spec.
- Any world authoring beyond the two that exist — the program's concern.
- Any reserve/held-out MODEL look — the held-out set here is a program-level
  split of the assembled cohort, distinct from the flag study's untouched
  reserve, which stays untouched regardless.

## Cost of the gate itself

Checks 1 and 3 involve GPU: the base-model cost re-run (~$1-2) and the loop
tests over 35-40 candidates (batched, cheapest-first; scale from the flag work's
~$0.50-0.80 per loop-tested model, so ~$20-30 if run for the full slate, though
many candidates resolve as inaccessible on paper before any GPU). Checks 2 and 4
are desk work, $0. Budget the gate at ~$25-35, and note that most of it buys the
answer to whether ~$hundreds of program sweeps are even assemblable — cheap
insurance, same shape as every gate this project has run.
