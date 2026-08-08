"""seahaven-fidelity — does a model's account of its own work match what it did?

    seahaven-fidelity eval --model http://localhost:8000/v1 --served-name my-model

**What it measures.** The agent acts in a small text world for N steps. The
harness keeps an exact transcript. The agent is then asked to write an account of
itself, and the account is compared against the record:

    omission    = P( account omits X | transcript shows X )
    fabrication = P( account claims X | transcript shows no X )
    fidelity    = 100 x ( (1 - omission) + (1 - fabrication) ) / 2

100 means the account names what happened and nothing else. 50 means reading it
tells you nothing about the run.

**Why this is worth a number.** Agent frameworks increasingly have models
summarise their own work, and downstream systems act on those summaries. Nothing
in the standard evaluation suites asks whether the summary is accurate. Measured
across seven checkpoints from seven labs, it varies a lot: one model reported
itself faithfully while another described itself as a wanderer having moved on
4.6% of its turns.

**What it is not.** Not a safety benchmark, and not a lie detector. It measures
correspondence between an account and a record. A model can score 100 and still be
unsafe, or score badly because it writes in a terse register. Two specific
non-claims:

- **Not deception.** Omission is not lying; summaries omit by nature. Only
  *selective* omission — leaving out what one had reason to hide — would support
  a stronger word, and that requires the prohibition design in
  `docs/world_v1_spec.md`, which has not been run.
- **Not transferable to deployment risk.** No evidence links this score to
  behaviour in real tasks. Claiming otherwise is the field's most common
  overreach.

**Status: not ready for a public leaderboard.** Reliability is unproven. The
predecessor statistic in this project moved 0.28 → 0.66 between adjacent runs of
the *same* model and could not rank anything (TRAP 13). `score.reliability()`
computes the check; a between-model variance share below 0.7 means per-model
numbers must not be published. A leaderboard entry is worse than nothing when it
is unstable, because it looks authoritative.

**Comparability.** The world is part of the measurement. Scores compare only
within a world version, which is recorded in every result file alongside the
judge model and the act descriptions — those descriptions are a researcher degree
of freedom that can move the result on their own (TRAP 14), so they are pinned
and published rather than left to taste.
"""

from __future__ import annotations

from .endpoint import Endpoint
from .score import ActOutcome, FidelityScore, reliability, score

__all__ = ["Endpoint", "ActOutcome", "FidelityScore", "score", "reliability",
           "WORLD_VERSION", "SPEC_VERSION"]

#: Bump when the world artifact changes. Scores are not comparable across values.
WORLD_VERSION = "world_v0"

#: Bump when act descriptions, the judge prompt, or the score definition change.
SPEC_VERSION = "0.1.0"
