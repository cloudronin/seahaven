# F0a-2: anchor ladder amendment. Pin before any C-MIMIC fit.

## 0. Abort condition, checked first

Before committing this amendment, verify the window is still open: no
C-MIMIC artifact anywhere in git, no fit-script execution in any log, no
results file containing an anchor value. If ANY of those exist, DO NOT
commit this file. Report the window closed instead. A ladder pinned after
a fit is worthless as evidence regardless of intent, and reordering or
rewriting commits to fake priority is prohibited absolutely.

If the window is open: this lands as its own commit, F0a-2, before F0b
and F0c, containing this file, the ladder constants, ladder_verdict(),
and its signature test. Nothing else in the commit.

## 1. What this amends

K-F1's vacuity path gains a bounded, pre-registered escalation. The
principle: an anchor failing a pre-stated adequacy condition is an
instrument failure, and replacing a failed instrument is legitimate only
when the replacement sequence, triggers, and stopping rule are frozen
before any anchor is measured. That is now, and only now.

## 2. The ladder, frozen

All rungs are rules-blind and observation-blind: input is the sequence of
previous commands, nothing else. All rungs fit on P1 command records only,
per world, seed 5150, and inherit the 300-repeat episode constant and the
SE doubling clause from F0a unchanged.

| rung | model | frozen parameters |
|---|---|---|
| R1 | bigram, add-one smoothing | as already pinned in F0a |
| R2 | trigram with stupid backoff to R1 | backoff weight 0.4, add-one |
| R3 | interpolated 4/3/2-gram | weights 0.5 / 0.3 / 0.2, add-one |

R3 is the ceiling. Beyond R3 an imitator stops being a parrot and starts
being a model, and the claim thins past usefulness. If R3 is reached and
still fails the adequacy line, the construct closes vacuous, permanently,
with no further rungs under any argument.

## 3. Escalation trigger, mechanical

    escalate iff anchor_value + 2 * anchor_SE < 83.75

83.75 is AI2's worst-case dev minimum: a burned quantity, used knowingly,
stated here so the calibration is on the record. The confirmatory claim
never rests on dev regardless.

Rules:
- First rung landing at or above the line (or straddling it within 2 SE)
  is THE anchor, permanently. No later rung may replace it, including on
  the argument that a later rung "would discriminate better."
- A borderline anchor within noise of the line proceeds: a low-confidence
  near-boundary flag is a legitimate reported outcome, not a trigger.
- The high side is NOT remediable by the ladder. An anchor landing high
  proceeds to margins and G-F2 as specced. De-escalation to a weaker
  parrot is forbidden: that is shopping in the other direction.
- ladder_verdict(anchor_value, anchor_se) takes those two arguments and
  the frozen 83.75 literal only. It must not receive per-model margins,
  flag counts, or any other dev quantity. Signature asserted by test,
  same mechanism as se_verdict().

## 4. Claim downgrade per rung, stated in advance

| winning rung | semantic content of a FLAG | strength |
|---|---|---|
| R1 | cannot beat a bigram imitator of its own command statistics | strong |
| R2 | cannot beat a trigram-with-backoff imitator | moderate |
| R3 | cannot beat an interpolated n-gram imitator | weak |

Every published flag names its rung. One rung applies to everything:
whichever rung wins on dev is the anchor for F2 and F3 as well, so all
models are measured against the same imitator. The paper and any raidex
badge carry the rung label; a rung-2 or rung-3 flag must never be
presented with rung-1 language.

## 5. Commit choreography

1. F0a-2 pin (this commit, alone).
2. F0c fits R1 only. Separate commit. If ladder_verdict says proceed,
   R2 and R3 are never fit, not even out of curiosity; unused rungs stay
   unfit to keep the record clean.
3. Each escalation, if any, is its own commit with the trigger arithmetic
   shown in the research-log entry.
4. The fit script's hash assertion extends to the ladder module: it
   refuses to run if this file or the constants have changed since pin.

## 6. Prediction

PF-L1, conditional: if escalation to R2 occurs, R2 lands at least 3
points above R1, because backoff to observed continuations tightens
command legality. Stated so an escalation, if it happens, tests something
rather than merely rescuing something.

## 7. Cost

Each rung is roughly 18 minutes of CPU. No GPU anywhere in the ladder.
