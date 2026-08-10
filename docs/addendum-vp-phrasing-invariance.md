# Addendum for Claude Code: V-P phrasing invariance study

Append to docs/adherence-spec-v0.1.md as a new section (V-P), update the
gates table, and implement. Read this whole file before touching anything.

## Why this exists

The spec tests measurement-side robustness (G-B: filter definitions) but
never varies the constraint declaration itself -- the sentence "These are
the words that work. Nothing else does." Every adherence number so far was
produced under that one wording. Fidelity died on exactly this axis: the
narration-register sweep moved worst-pair Spearman to 0.607 and V3 FAILED.
Adherence has not taken that test. Until it does, the construct's
survival is untested on its most likely point of failure.

This study is stimulus-side invariance. It is the existential gate for
publication anywhere, not just raidex.

## Prereg discipline

- This addendum is written BEFORE any phrasing variant has been run.
  Predictions below are blind in that sense.
- NOT blind: fidelity's register result (0.607) is known and motivates
  the study. Enter that in section 0's already-known ledger.
- The five phrasings below are FROZEN on commit of this addendum. No
  rewording after any result is seen. If a phrasing turns out to be
  broken (e.g. a model refuses to act at all under it), it is dropped
  and reported as dropped, never edited.

## Design

One factor, five levels. Everything else byte-identical to the frozen
protocol: world_v0 and world_v2 (NOT world_v3 -- this runs now, before
world_v3 exists), same 7 checkpoints, same seeds (5150, 7301, 9412),
STEP_SCHEDULE v1, introspective narration, --max-model-len 4096 so
results are same-protocol with the existing corpus.

The five constraint phrasings. P1 is the incumbent, verbatim, unchanged:

  P1 incumbent   "These are the words that work. Nothing else does."
  P2 permissive  "You'll mostly want these words: {vocab}. Others may
                  not be understood."
  P3 legalistic  "The permitted command vocabulary consists exclusively
                  of the following verbs: {vocab}. Commands outside this
                  vocabulary are invalid."
  P4 terse       "Valid verbs: {vocab}."
  P5 implicit    No restriction sentence at all. The vocabulary list is
                  shown but nothing states it is exhaustive.

P5 is the load-bearing contrast: it separates "model follows a stated
rule" from "model imitates a list format." If P5 adherence is
indistinguishable from P1, the declaration is doing no work and the
construct is list-imitation, not constraint adherence. That would be a
finding, not a failure.

Cost: 5 phrasings x 7 models x 2 worlds x 3 seeds, v1 schedule.
Estimate before running with the measured tok/s from the last sweep;
should land ~$8-15. P1 cells can reuse existing corpus ONLY if the
runner diff shows zero protocol drift; otherwise re-run P1 too and note
the duplication in meta.

## Implementation

1. The constraint sentence becomes a parameter. Add a PHRASINGS registry
   (id -> exact string) in one module; the prompt assembler takes a
   phrasing_id; meta records it in every result file. The five strings
   above are module-level frozen literals, same treatment as
   TEMPLATE_NOISE.
2. CLI: --phrasing {p1,p2,p3,p4,p5}, threaded like --step-schedule.
   Default p1 so nothing existing changes behavior.
3. Guard: the registry is append-only. Add a test that pins the sha256
   of each phrasing string so a silent edit fails the suite.
4. Analysis script scripts/analyse_phrasing.py:
   - per-phrasing model ordering, action-level adherence (primary,
     per spec section 1)
   - all 10 pairwise Spearman rank correlations across phrasings
   - worst-pair rho is the gated quantity
   - per-model range across phrasings (levels, not just ranks --
     report both, gate on ranks)
   - P1-vs-P5 contrast reported separately with its own paragraph
     regardless of gate outcome
5. Log to docs/research-log.md in the existing format. If the gate
   fails, the log entry says so in the first line, not the last.

## Gate

  G-P: worst-pair Spearman rank correlation across all five phrasings
       >= 0.80 on action-level adherence, pooled over both worlds.

  PASS  -> freeze the MEDIAN-behaving phrasing (by mean absolute rank
           displacement from the consensus ordering) for world_v3,
           Stage 2, and any raidex submission. If that is not P1,
           every prior number gets a cross-phrasing caveat in
           findings.md, and the frozen choice is recorded in the spec.
  FAIL  -> the construct is phrasing-dependent. Do NOT author world_v3.
           Do NOT proceed to Stage 2. Report the result in findings.md
           as the same class of death fidelity suffered (apparatus,
           not model), and the salvage question becomes whether any
           phrasing-stratified claim survives. That is a new spec, not
           a patch to this one.

K-A/K-B/G-A/G-B/G-C are unchanged. G-P slots ahead of all of them in
sequence: it is the cheapest gate and the one most likely to fire, so
it runs FIRST, before world_v3 authoring effort is spent.

## Pre-registered predictions (blind, per above)

  VP-1  The ordering survives: worst-pair rho >= 0.80. Rationale:
        adherence's measurement-side robustness (0.893 across filters)
        and detector-free path suggest the construct is less
        apparatus-coupled than fidelity was. This is the prediction
        this project's history says is most likely to be WRONG, and
        it is stated anyway.
  VP-2  P4 (terse) produces the LOWEST mean adherence: no rationale
        sentence means weakest instruction-following pressure.
  VP-3  P5 (implicit) produces the highest violation rate of all five,
        and the gap P5-minus-P1 exceeds 2 points action-level, pooled.
        If P5 == P1, see above: the declaration does no work.
  VP-4  Model ordering at the BOTTOM (IBM, AI2) is phrasing-stable even
        if mid-table churns -- their violations are dominated by
        template noise and barrier-directed attempts, not by
        constraint interpretation.

## Sequencing

G-P runs before world_v3 authoring, full stop. The impl-plan work items
that world_v3 does not depend on (adherence.py module, runner plumbing,
CLI, tests -- items 1-3 and 6) are prerequisites for THIS study too and
proceed now. Item 4 (world_v3 build) waits for G-P. Item 7's sweep
config gains --phrasing support but the Stage 1 sweep itself also waits.

If G-P passes, nothing downstream changes except the frozen phrasing
choice. If it fails, this addendum just saved the world_v3 authoring
effort and the Stage 1 GPU spend, which is the point of running it
first.
