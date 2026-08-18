# Manuscript data bundle — everything the skeleton needs, emitted not remembered

Goal: one `manuscript-data/` directory, produced entirely by `vworld` verbs
and register functions, that Claude (chat) and Vishnu can write the skeleton
from without touching the repo. **Every number comes from an emitted
artifact. Nothing is transcribed from conversation memory — several figures
quoted in past sessions predate corrections and are wrong now.** If the
skeleton needs a number no artifact emits, that is a register gap: file it,
add the emitting function, then emit. The paper cites functions; this bundle
is the paper's inputs.

Target venue: TMLR (methods). Working title area: expedience under terminal
stakes; the instrument, its corrections, and the serving layer it exposed.

## Spec corrections — 2026-08-18

**This spec misstated its own inputs in three places**, which is the failure
mode it exists to prevent, and the premise check that found them is the one it
asks the bundle to institutionalise. Recorded here rather than silently edited;
they are corrections-section material.

| Was written | Actually | Where |
|---|---|---|
| Exhibit 1 "build if not yet built — it has been ordered and is $0" | Built and shipped in 0.3.0. Reports **ZERO**, and money does **not** unblock it: all 28 cross-provider pairs are 7 models whose Together side is retracted (r15/r16), and all seven are unservable on Together — six non-serverless, Kimi-K3 empty-content under the pinned request form | §5 |
| `emit corrections` "all 13" | `artifacts.CORRECTIONS` holds **5** rows; research-log numbers `[CORRECTION]` to **11**. Three disagreeing counts, and the 13 was a hand-maintained number living in conversation prose — the exact family this programme has caught repeatedly in code. Resolution: backfill the register from the log, every row verified against its commit, and emit whatever the true count is | §4 |
| Bands (#118) "if implemented by bundle time… else TBD" | Implemented: `register/c5.py:report`, claim `vetohold.bands`, `vworld emit bands` | §2 |

Also narrower than it reads: the #113 identity-state census largely exists
already as `vworld corpus identity`.

## Bundle layout

    manuscript-data/
      MANIFEST.md          figure -> emitting function -> file (the claims map)
      register/            per-artifact outputs, one file each
      tables/              formatted tables ready to drop into LaTeX/md
      log-extracts/        corrections ledger, trap index, named rules
      exhibits/            exhibit-1 mechanism table + any figures

Every file header: generating command, commit hash, corpus digest, date.

## Section-by-section: what to emit

### 1. The instrument (methods)

- **Worlds table**: LAT, W2, W3 (+ LAT2 twin, one-phrase diff): decay, start
  health, horizon, forbidden restore, permitted objects and restores, S,
  Held-Karp optimum, greedy_min, price-of-the-rule. The 1-object vs 2-object
  structure column. Source: `vworld worlds` + lock constants.
- **COMP world** params and the gate criterion (mean min health vs
  greedy_min), with round-10 exclusion reproduction noted.
- **Arms and pairing**: A1/A0 seed-pairing scheme, m per arm, temperature,
  max_tokens, seed derivation, burned-block registry size.
- **Metric definitions**: intent_rate, rate_any, the gap; C1 formula; C3
  difference form; funnel stage definitions (saw/took/attempt/eat).
- **The five decision rules** as pinned, each with its outcome and the data
  that resolved it (rule 2's self-contradiction included verbatim; rule 5's
  activation with TRAP 38 caveat).

### 2. The register (results core)

- **Full register table**: 17 models x {C1, C3, per-world intent + rate_any,
  A0 rates, provider, occasion date(s), deviation flags (MiniMax-M2.7
  reasoning; Terra temperature), identity status}. Source:
  `vworld read` / register emitters.
- **vetohold.cohort** (scored, joined) and the six never-joinable by name.
- **Spread and ceiling**: current spread tuple; ceiling count with per-model
  0/144 and Wilson uppers, provider list.
- **Bands (#118)**: IMPLEMENTED — emit `vworld emit bands` with basis and
  boundary flags per model. Do not hand-compute.
- **Rule 5 artifact**: separating pairs / total pairs, threshold, the
  direction argument (intervals omit occasion+provider variance -> count is
  an upper bound), TRAP 38 text.
- **GLM family row**: the four versions with C1, provider, occasion —
  emitted as its own table since it carries the post-training-choice
  argument.

### 3. Mechanisms and the funnel

- **floor_mechanisms** artifact: route per (model, world) for every
  zero-reach model; Terra's world-dependence explicit; the NEVER_TOOK
  vacancy statement with its witness (no measured model fails to engage;
  every zero is the rule's work, A0-verified).
- **Funnel table** for the ceiling four + 2-3 mid-scale exemplars:
  saw/took/attempt/eat counts both arms.
- **A0 saturation summary** per world: pooled A0, the licence-gate stats,
  and the current two-low observation (cogito + Qwen2.5-7B vs the six, with
  p) labeled single-occasion-uncertified; cogito's four-day trace with its
  paired p, labeled unresolved.

### 4. Corrections as contribution (its own section)

- `vworld emit corrections`: **the emitted count, not a remembered one** — see
  the corrections table above. Backfill `artifacts.CORRECTIONS` from the log
  first; each row carries its commit ref and one-line what-changed, and the
  emitter already verifies every row against the commit it cites. Add the
  13 -> emitted supersession line to MANIFEST per rule 4. 
- **Trap index extract**: count, plus the named families with instance
  counts (can't-fire checks; duplicated-identity sites; hand-maintained
  numbers; observer-environment: mtime/pipe-exit/timezone; membership-
  dependent derivation).
- **#113 numbers**: cells affected, relabelled count, identity-state census
  (VERIFIED / UNVERIFIED / MISLABELLED counts now), the repeatability gift
  (identical-input pairs, zero fires, min p, per-level spread), detector
  re-read result (zero events on corrected corpus).
- **Named rules extract** (log-extracts/): requested==served per cell;
  bug closure enumerates symptoms; structural rule lands -> enumerate paths;
  reference disjoint; taint law; no verdicts off unfinished runs; exit codes
  never through a pipe.

### 5. The serving layer (the seismograph section)

- **occasion-health --all**: the base-rate artifact — sweeps examined,
  events, dates, scope, false-alarm expectation at the pinned alphas.
- **The one confirmed event**: DS-V4-Flash step, before/after, p, the
  dissociation evidence (within-session agreement, cross-day disagreement).
- **Fleet day-one row(s)**: cells, cost, verdicts, identity attestation
  count; the matched-pair design summary (probe/control, both columns,
  levels-never-compare rule).
- **Serving-layer findings table**: tiering deletion (6 of 14, date);
  enable_thinking cross-provider asymmetry (Together truncates / DeepInfra
  honors); warmup-vs-sustained (GLM-5.2 mid-game rate); timezone label
  defect (recorded, narrowed claim stated); attestation design (header vs
  body, both identities per cell).
- **Exhibit 1** — BUILT (`vworld emit exhibit-1`). Emit as-is: it reports ZERO
  admissible flips, and the zero is the content. The Together side of all 28
  cross-provider pairs is retracted, and no budget reopens it. The table prints
  `SUBSTRATE_PENDING`: the matched-pair fleet is the route out, since Flash
  carries the A1 decision cell on both columns daily, so route labels accrue
  from PROBE cells over the coming weeks — descriptive, probe-sourced, and
  labelled so, because probe cells are excluded from the score corpus by
  construction and must never read as round measurements.

### 6. Program-level numbers (abstract/intro/discussion)

- `vworld corpus status` + manifest: cells, digest, providers, identity
  census.
- **Spend ledger**: program total, per-round, fleet daily cost. Source:
  emit spend (if no such artifact exists, that is a gap to file — the
  number is quoted in the paper).
- Suite size, figure count recomputing, pin count (open/closed/retired,
  retired digests recomputing), rounds served.
- **E1 correlations**: current n, every coefficient with CI, permutation p,
  Bonferroni threshold, tie counts, range-restriction disclosure, and the
  power window at the real n (recompute the closed-arms bounds at n=11;
  the n=17 bounds in past chat are stale).
- E2 spread stats; E3 candidate-convergence at the real cohort (recompute —
  all remembered values are pre-retraction).

### 7. Limitations inputs

Single-occasion middle; mixed-provider register with per-row provenance;
cohort-rental (tiering) sentence; E1 underpowered-by-construction; the
timezone provenance narrowing; LAT two-low observation uncertified; what the
instrument cannot attribute (mechanism vs internals — the three-internals
equivalence statement).

## Rules for the bundle build

1. CLI/register functions only; no ad-hoc scripts. A needed-but-missing
   emitter is filed and built first (the standing rule).
2. Any figure that cannot be recomputed from committed cells does not enter
   the bundle — same bar as the paper.
3. MANIFEST.md maps every number the skeleton will quote to its function,
   per the claims-register discipline; the skeleton will cite functions.
4. Nothing in the bundle from memory or chat history. Where a past-quoted
   number differs from the emitted one, add a line to MANIFEST noting the
   supersession (these become corrections-section material, not errors).
5. $0 total. If anything wants serving, it is out of scope for the bundle.

## Skeleton (CC drafts after the bundle, placeholders bound to MANIFEST)

Abstract / 1 Intro / 2 Related work (stub) / 3 The instrument (worlds,
arms, metrics, gates) / 4 Validity machinery (pins, register, identity,
occasion, licence) / 5 Results (floor + mechanisms; the middle; family
contrast; funnel decomposition) / 6 The serving layer (events, base rate,
attestation, cross-provider) / 7 Corrections as contribution / 8
Limitations / 9 Discussion (one MRL/grid paragraph). Every number as
`[[claim-name]]` bound to MANIFEST — no literals in prose.
