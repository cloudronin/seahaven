# Phase-switch entry + directives for Claude Code

Paste the entry into docs/research-log.md as-is. The directives section
supersedes the F0a-2r implementation plan and governs work until a FREEZE
entry replaces it.

---

## 2026-08-09 — PHASE SWITCH: exploration declared, freeze-and-validate to follow

This entry changes the governing discipline, not a result.

### The decision

The goal, restated: determine whether any imitation anchor and boundary
rule can separate the universe of models in a useful, stable way. Whether
that is possible is unknown. Therefore: maximum flexibility now, validation
over fresh cohorts later, with the boundary between the two phases marked
in this log by a FREEZE entry when a candidate instrument is frozen.

### The boundary

Everything after this entry, until FREEZE, is exploration. Nothing
produced during it is confirmatory, and nothing from it may be described
as pre-registered, blind, or confirmatory in any log entry, paper draft,
badge, or conversation. Enforcement is one field: every result file
carries phase: "exploration".

### The sandbox

- Models: the seven dev checkpoints. Additions permitted, base and small
  checkpoints encouraged for dynamic range; each addition is recorded in
  the burn ledger before first use.
- Worlds: v0 and v2 only.
- Inside the sandbox, anything goes: any anchor family (n-gram,
  class-conditioned, trained micro-imitators), any statistic, any boundary
  rule, any phrasing subset. No blindness bookkeeping.

### The burn ledger, initial state

| touched | items |
|---|---|
| models | Alibaba, MistralAI, AI2, IBM, TII, Meta, Google |
| worlds | v0, v2 |
| anchor families | bigram R1, previewed at n=2 (~89.14, SE 1.77) |
| instruments | V-P five-phrasing sweep, C-RAND, C-NOISE |

Every new model, world, anchor family, or instrument variant is appended
before use. The ledger is what the eventual confirmatory claim stands on:
it defines what the freeze must treat as development.

### Held-out reserves, established now

- world_v3: parameters stay frozen as specced. It may be authored at any
  time; it must not be swept by any model until after the freeze.
  Authoring does not burn it; play does.
- Reserve cohort: 4 to 6 checkpoints selected by the frozen breadth
  criteria: family coverage including at least one family absent from the
  dev seven, at least one base/instruct pair, A4-style loop test before
  inclusion, and never selected by any containment-adjacent result.
  Claude Code enumerates candidates, loop-tests them, and pins the passing
  list in the commit adjacent to this entry, before any anchor iteration
  begins. Scoring budget: at most two looks at the reserve, and only when
  a candidate has cleared the possibility bar. Reserve models never enter
  the sandbox.

### The possibility bar, written before iterating

A candidate (anchor family, statistic, boundary rule) clears the bar iff,
in the sandbox:

1. Non-trivial split: at least two models on the minority side.
2. Cross-world: memberships identical on v0 and v2.
3. Seed-stable: no model's side flips under episode bootstrap at 90
   percent confidence.
4. Phrasing-robust: memberships unchanged under leave-one-phrasing-out
   recomputation.

Only then: one reserve look. The freeze claim, if the look supports it:
the frozen instrument produces non-degenerate, seed-stable,
cross-world-stable memberships on checkpoints never seen during
development. That is the possibility result. Bounds and reproducibility
across further cohorts are the later program, run under version
discipline.

### What does not relax during exploration

- Stooge bracketing on every fitted anchor: C-NOISE below it, C-RAND
  above it, else the fit or the pipeline is wrong, full stop.
- Anchor-location SE at or below 0.30; comparing candidates requires
  locations known finer than their separations. The doubling clause
  applies as hygiene, not as a gate.
- VLLM_BATCH_INVARIANT=1 on every GPU run.
- The per-world loader regression against the pooled table.
- Append-only log, dated corrections, TRAP numbering.

### Status of prior confirmatory artifacts

flag-spec v0.1, F0a-2r, and predictions PF-1, PF-2, PF-L1 remain in the
record and govern nothing during exploration. PF-1 and PF-2 are reported
when R1's full fit lands, labeled development observations, read as
calibration of expectations. The freeze produces spec v1.0 with a fresh
confirmatory battery over cohorts and worlds absent from the burn ledger.
Version discipline carries forward from there: any post-freeze instrument
change reopens development and requires fresh cohorts for confirmation.

### Two constraints carried from the superseded plan into the freeze

1. One anchor family across all worlds in any frozen instrument.
   Per-world anchor families confound world with family, the two-factor
   failure that killed the fidelity ordering.
2. No dev-derived constant leaks into held-out evidence. Anything tuned
   in the sandbox is development by definition.

---

## Directives to Claude Code, superseding the F0a-2r plan

| plan item | disposition |
|---|---|
| ladder commit: trigger, ceiling, verdict fn, escalation choreography | drop, superseded by the survey |
| R1/R2/R3 implementations shipped in one commit | keep; they are the first three survey candidates |
| "unused rungs are never fit" | inverted: fit all three rungs, both worlds, one pass; report three anchor locations side by side |
| trigger arithmetic and escalation commits | drop |
| claim-language fields rung / blindness / status_language | replace with phase: "exploration" on every result file |
| F0b per-world loader and its regression | keep unchanged, do first |
| F0c PF-1/PF-2 reporting | keep, labeled development observations |
| stooge bracketing check | keep, on every fitted anchor |
| F2 world_v3 sweep (~$12) | drop; v3 is a held-out reserve |
| hash assertions on new exploration code | not required; existing pins on flag.py remain as historical record |

Sequence: commit this entry, then F0b, then the three-anchor survey
(about an hour of CPU, $0), then reserve enumeration and loop tests, then
read the survey against the possibility bar and decide sandbox widening
(two or three base or small checkpoints, roughly $10 of the ~$16
remaining) only if the ceiling compression turns out to bind.
