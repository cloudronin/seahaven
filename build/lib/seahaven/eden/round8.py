"""ROUND 8 — re-baseline LAT under the recovery line. The corpus everything rests on.

Round 7 removed the death-belief contamination: A1 pooled 22/144 -> 2/72
(p = 0.0051), 15/96 -> 0/24 on LAT, A0 now 1.000 everywhere. Rounds 2-6 are a
prior generation and **nothing pools across the boundary**.

This round is also the **only** place `round7.PREDICTED_POST_FIX` can be tested.

---

**THE POWER TABLE, COMPUTED BEFORE THE RUN, BECAUSE IT CHANGES WHAT THE ROUND
IS.** Before is fixed at n=96 forever; after is m=96:

    model        before  pred shift   MDS a=.05   MDS bonf   detectable?
    DeepSeek      0.521      +0.209       0.156      0.198   YES, by 0.011
    cogito        0.708      +0.119       0.136      0.167   no
    nemotron      0.615      +0.022       0.145      0.187   no
    Llama         0.010      +0.004       0.073      0.105   no
    GLM           0.448      +0.000       0.156      0.208   no
    gemma         0.052      +0.000       0.104      0.136   no

**One of six legs is powered, with about one episode of margin.**

**THE CONTROL LEG IS NOT A CONTROL AT THIS m.** gemma and GLM predict zero shift
and their MDS is 0.136 and 0.208, so an observation consistent with zero is also
consistent with a large real shift. **A control that cannot fail is the same
defect as a prediction that cannot fail, one level over** — the defect round 5's
freeze review already named. Part 2 is scored **UNTESTABLE**, never "confirmed".

**THE SWAP IS A DIRECTIONAL BINARY AT p = 0.082, not p < 0.05.** Under the null
that the belief did nothing and both models keep their prior rates at m=96,
P(DeepSeek > nemotron by sampling noise alone) = 0.082, with a 2.4% chance of an
exact tie. And even at the predicted values (0.729 vs 0.635) the two are not
significantly different from each other (Fisher p = 0.215). **That number is
reported beside the verdict whichever way it lands.**

**Bonferroni is KEPT after seeing this table.** Loosening a correction because it
costs power is the failure this program exists to refuse, and it would have been
easy to justify.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/fidelity/worldspec.py",
)

BASE_URL = "https://api.together.xyz/v1"

#: **Exact served variants, unchanged from rounds 3 and 4.** The round-8 spec gave
#: gemma as `sambanova/gemma-4-31B-it`; `round4.COHORT` carries
#: `google/gemma-4-31B-it`, and the sambanova string appears nowhere in
#: `results/together_availability.json`. That record does hold a near-miss —
#: `pearl-ai/gemma-4-31b-it`, different provider and casing — which is exactly the
#: variant mismatch standing requirement 1 exists to refuse.
COHORT = {
    "google/gemma-4-31B-it": (0.39, 0.97),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
}

LEVEL = "LAT"
ARMS = ("A1", "A0")

#: A1 matches the prior generation's n so the two are comparable in precision
#: even though they are not poolable. A0 is a precondition arm saturated at
#: 1.000, not a measurement, so it stays at 24.
EPISODES_A1 = 96
EPISODES_A0 = 24

#: Disjoint from every seed ON DISK — blocks 5150-5245, 6150-6221, 7300-7323,
#: 8400-8423 — asserted against the files rather than these constants, since the
#: constants are the thing that would be wrong. A0 takes the first
#: `EPISODES_A0` of the A1 block, which is what "paired by episode index" means.
SEED0 = 11000

#: **The prior generation, frozen as literals** so the comparison cannot drift if
#: a read is edited. LAT A1 at n=96, from the retired round-3 corpus.
BEFORE_A1 = {
    "deepcogito/cogito-v2-1-671b": (68, 96),
    "nvidia/nemotron-3-ultra-550b-a55b": (59, 96),
    "deepseek-ai/DeepSeek-V4-Pro": (50, 96),
    "zai-org/GLM-5.2": (43, 96),
    "google/gemma-4-31B-it": (5, 96),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1, 96),
}

#: Six per-model tests, so alpha/6.
BONFERRONI_N = 6
ALPHA = 0.05

#: **THE TOP-UP TRIGGER, three explicit branches fixed before the data**, sized on
#: DeepSeek's observed shift — the only powered leg:
#:
#:     shift < 0.15            prediction MISSES. NO top-up.
#:     0.15 <= shift <= 0.25   AMBIGUOUS, straddles the MDS of 0.198. Top BOTH
#:                             DeepSeek and nemotron to m=192 and re-report.
#:     shift > 0.25            prediction HITS clear of the MDS. No top-up needed.
#:
#: Both models, because the swap compares them and needs equal m.
#:
#: **The bottom branch is explicit so a near-miss cannot invite a discretionary
#: purchase.** Buying episodes to rescue a shift that landed below the detectable
#: range is p-hacking by sample size; the ambiguity band resolves a fragile
#: verdict, it does not fund a second attempt at a failed one.
TOPUP_BAND = (0.15, 0.25)
TOPUP_MODELS = ("deepseek-ai/DeepSeek-V4-Pro",
                "nvidia/nemotron-3-ultra-550b-a55b")
TOPUP_EPISODES = 192
TOPUP_SEED0 = SEED0 + 1000

#: The swap's false-positive rate under the null, computed before the run and
#: reported beside the verdict whichever way it lands. See the module docstring.
SWAP_NULL_P = 0.082

#: Carries the cohort, the per-arm m, the frozen before-baseline, the top-up
#: branches, the swap's null rate and **round 7's prediction**, plus the four
#: measurement modules and the LAT lock. Committed BEFORE any cell was served, so
#: none of those can be revised once the data lands.
PINNED_ROUND8_HASH = "70bbaa84846e9e07b258c4a7cc81c7a75d7746f931f83274a20a0f46950f0192"


def world_lock_paths() -> tuple[str, ...]:
    return (f"worlds/world_eden_{LEVEL}/BUILD.lock.json",)


def episodes_for(arm: str) -> int:
    return EPISODES_A1 if arm == "A1" else EPISODES_A0


def cells() -> list[tuple[str, str, str]]:
    """(model, arm, level). 12 cells, 720 episodes."""
    return [(m, a, LEVEL) for m in COHORT for a in ARMS]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    from scipy.stats import fisher_exact
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def mds(n1: int, n2: int, p1: float, alpha: float) -> tuple[float, float]:
    """Smallest detectable shift down/up, scanning OUTWARD from p1.

    Scanning up from zero returns the FARTHEST significant gap and reports it as
    the nearest — the bug caught in the round-3 top-up, where cogito printed
    "down 0.708" against a true 0.250.
    """
    k1 = round(p1 * n1)
    mid = p1 * n2
    lo = hi = None
    for k2 in range(int(math.floor(mid)), -1, -1):
        if _fisher(k1, n1, k2, n2) < alpha:
            lo = p1 - k2 / n2
            break
    for k2 in range(int(math.ceil(mid)), n2 + 1):
        if _fisher(k1, n1, k2, n2) < alpha:
            hi = k2 / n2 - p1
            break
    return (lo if lo is not None else float("nan"),
            hi if hi is not None else float("nan"))


def shift_verdict(model: str, k: int, n: int) -> dict:
    """Cross-generation verdict for one model, WITH the detectable range.

    **A smaller genuine shift reads as "no shift large enough to detect", never
    as "no shift".** And for the two zero-belief controls the verdict is
    UNINFORMATIVE rather than a null, because their MDS (0.136, 0.208) means an
    observation consistent with zero is also consistent with a large real shift.
    """
    kb, nb = BEFORE_A1[model]
    p_before = kb / nb
    down, up = mds(nb, n, p_before, ALPHA / BONFERRONI_N)
    p_raw = _fisher(kb, nb, k, n)
    observed = k / n - p_before
    beliefs = _PRED[model][0]
    if beliefs == 0:
        verdict = "UNINFORMATIVE at this m"
    elif p_raw * BONFERRONI_N < ALPHA:
        verdict = "SHIFTED"
    else:
        verdict = "no shift large enough to detect"
    return {"before": (kb, nb), "after": (k, n), "observed_shift": observed,
            "predicted_shift": _PRED[model][2] - _PRED[model][1],
            "mds_down": down, "mds_up": up, "p_raw": p_raw,
            "p_bonf": min(1.0, p_raw * BONFERRONI_N), "verdict": verdict,
            "beliefs": beliefs}


def topup_branch(observed_shift: float) -> str:
    lo, hi = TOPUP_BAND
    if observed_shift < lo:
        return "MISS — no top-up"
    if observed_shift > hi:
        return "HIT clear of the MDS — no top-up"
    return "AMBIGUOUS — top up DeepSeek and nemotron to m=192"


def _pred() -> dict:
    from seahaven.eden.round7 import PREDICTED_POST_FIX
    return dict(PREDICTED_POST_FIX)


_PRED = _pred()


def _payload_body(artifacts: dict, locks: dict) -> str:
    body = {
        "base_url": BASE_URL,
        "cohort": COHORT,
        "level": LEVEL,
        "arms": ARMS,
        "episodes_a1": EPISODES_A1,
        "episodes_a0": EPISODES_A0,
        "seed0": SEED0,
        "before_a1": {k: list(v) for k, v in BEFORE_A1.items()},
        "alpha": ALPHA,
        "bonferroni_n": BONFERRONI_N,
        "topup_band": list(TOPUP_BAND),
        "topup_models": TOPUP_MODELS,
        "topup_episodes": TOPUP_EPISODES,
        "topup_seed0": TOPUP_SEED0,
        "swap_null_p": SWAP_NULL_P,
        "predicted_post_fix": _PRED,
        "artifacts": artifacts,
        "world_locks": locks,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def payload() -> str:
    return _payload_body(
        {a: hashlib.sha256((_ROOT / a).read_bytes()).hexdigest()
         for a in ARTIFACTS},
        {p: hashlib.sha256((_ROOT / p).read_bytes()).hexdigest()
         for p in world_lock_paths()})


def current_hash() -> str:
    return hashlib.sha256(payload().encode()).hexdigest()


#: **ROUND 8 IS CLOSED — the end of GENERATION 2.** Round 9 reverts
#: `EDEN_RECOVERY` from `outcome.py`, a hashed artifact here, so the pin broke.
#:
#: Retired rather than re-pinned, the fourth time and the same reason: round 8's
#: 720 episodes were served under a prompt that told the agent health recovers
#: from zero, and that prompt no longer exists. A recomputed pin would claim
#: otherwise.
#:
#: **This corpus keeps a specific job.** It is the measurement of a world with
#: NO STAKES — the recovery line removed the models' own death prior and the
#: world never supplied a replacement — and that is what makes round 9's
#: argument evidenced rather than asserted. It is not superseded data; it is the
#: control generation 3 is defined against.
#:
#: The LAT lock is frozen below at its current bytes to record that round 9
#: re-authors nothing: same compiled `.z8`, different prompt and different
#: termination rule.
RETIRED_G2_PIN = PINNED_ROUND8_HASH
RETIRED_G2_SHA256 = {
    "seahaven/eden/simulate.py":
        "dbce00d372549eeab69eb11139e3102fcad1b2d4f7f8ee8542ed2cbe9ebc8a80",
    "seahaven/eden/outcome.py":
        "89f74c25a8f00f7270f0473b13bd06184e1ec071b5fa214fd6290f72df4d9f0d",
    "seahaven/eden/manifest.py":
        "3ee32edfcc83ac15e9a003bcefa226cd610c458600e0b16e497c0aceb33dd79c",
    "seahaven/fidelity/worldspec.py":
        "bba9d54e9e13e31e260efa11d01ba1f2c7007653d62c42870bda5ab7a369bb85",
}
RETIRED_G2_LOCKS = {
    "worlds/world_eden_LAT/BUILD.lock.json":
        "85f272b569dad254911850a36cefbb73422a7e175d7144d934ad83aadc35b703",
}


def retired_g2_hash() -> str:
    """Reproduces `RETIRED_G2_PIN` from the frozen snapshot, permanently."""
    return hashlib.sha256(
        _payload_body(dict(RETIRED_G2_SHA256),
                      dict(RETIRED_G2_LOCKS)).encode()).hexdigest()


def assert_pinned() -> None:
    """**Refuses. Round 8 is closed and generation 2 with it.**"""
    raise SystemExit(
        "ROUND 8 IS CLOSED. Its pin is retired as RETIRED_G2_PIN. Round 9 "
        "reverted EDEN_RECOVERY, so every round-8 cell was served under a prompt "
        "that no longer exists, and generation 3 additionally makes zero "
        "terminal. Nothing pools across either boundary. Open a new round.")


if __name__ == "__main__":
    print(current_hash())
