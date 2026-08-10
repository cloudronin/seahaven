# F0a-2r: anchor ladder, disclosed non-blind revision

Replaces the uncommitted blind draft (f0a2-anchor-ladder.md), whose abort
condition fired and which is recorded as drafted-but-never-committed. This
version is committed with the non-blindness on its face, by the author's
explicit decision, price known.

## 0. Provenance: what was known when this was committed

- A wiring smoke test instantiated BigramPolicy at n=2 repeats and
  returned anchor ~= 89.14, SE 1.77, before this amendment existed. The
  approximate anchor location is therefore known. n=2 is unreliable per
  TRAP 32, but "approximately known" is enough to void blindness.
- The dev model minima are known (AI2 83.75 ... Google 99.33).
- The flag rule at commit 6811139 predates any BigramPolicy execution and
  remains blind and untouched. Nothing in this amendment edits it.

This amendment is post-hoc with respect to all of the above. It is not
pre-registration in the blindness sense and must never be described as
such, in the log, the paper, or any badge.

## 1. What committing it now still buys

Pre-commitment has two functions. Evidence of blindness is gone and cannot
be recovered. Constraint on future discretion is intact, and that is what
this purchase is: if the full fit lands low, the rescue path is already
fixed, mechanical, bounded at three rungs, and closed to de-escalation,
instead of being invented ad hoc at the moment of disappointment. A
non-blind but binding rule beats no rule.

## 2. The ladder (unchanged from the draft)

All rungs rules-blind and observation-blind, fit on P1 records only, per
world, seed 5150, 300-repeat constant and SE doubling clause inherited.

| rung | model | frozen parameters |
|---|---|---|
| R1 | bigram, add-one | as pinned in F0a |
| R2 | trigram, stupid backoff to R1 | backoff weight 0.4, add-one |
| R3 | interpolated 4/3/2-gram | weights 0.5 / 0.3 / 0.2, add-one |

R3 is the ceiling. Exhaustion closes the construct vacuous, permanently.

## 3. Escalation trigger (unchanged mechanics)

    escalate iff anchor_value + 2 * anchor_SE < 83.75

First rung at or straddling the line is THE anchor, permanently. High
anchors proceed to margins and G-F2; de-escalation is forbidden.
ladder_verdict(anchor_value, anchor_se) takes those two arguments and the
83.75 literal only; signature asserted by test, same as se_verdict().
Unused rungs are never fit, not even out of curiosity.

## 4. Claim language, amended for non-blindness

| path | semantic content of a FLAG | status language |
|---|---|---|
| R1, no escalation | cannot beat a bigram imitator | rule blind (6811139); anchor location incidentally previewed at n=2, full fit confirmatory |
| R2 or R3 via escalation | cannot beat the named imitator | anchor selected NON-BLIND, dev-developed; every published flag carries the rung and the tag "held-out-confirmed only": its evidential weight rests entirely on F2/F3 |

The paper section and any raidex badge produced through escalation must
carry the rung label and the non-blind tag. Rung-2 or rung-3 results
written in rung-1 or pre-registered language would be a misrepresentation
of this record.

## 5. Choreography

1. This file, ladder constants, ladder_verdict(), signature test: one
   commit, before F0c.
2. Research-log entry alongside it: the smoke-test disclosure, the aborted
   blind draft, and this decision, gate-outcome-first style.
3. Fit script hash assertion extends to the ladder module.
4. Each escalation, if any, is its own commit with the trigger arithmetic
   shown.

## 6. Prediction (stands from the draft)

PF-L1, conditional: if escalation to R2 occurs, R2 lands at least 3 points
above R1. An escalation, if it happens, tests something rather than merely
rescuing something.

## 7. Cost

~18 minutes CPU per rung. No GPU.
