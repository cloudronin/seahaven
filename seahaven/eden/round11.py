"""ROUND 11 stage 4 — does the shape travel? W2 and W3 under terminal death.

**The selection criterion changed, and the reason is recorded here rather than
lost.** The subset was chosen for round 11 by *resolved pairs on LAT*: only pairs
already separated could answer whether the ORDER travels. That basis is **dead** —
after the two DS-V4-Flash blocks there is no adjacent pair in the sixteen with
non-overlapping intervals, and the cohort is a continuum whose ordering is
unresolved. A subset justified by a property the data no longer has would be a
label outliving what it counts.

**Restated basis: span the observed range, keep the floor replicate.** Weaker than
the original and stated as such. The order question is correspondingly weak: an
unresolved source ordering cannot produce a strong travel result, and that is
printed beside the Spearman rather than discovered afterwards.

DS-V4-Flash is **excluded** while its stability is open. Three blocks give 0.375,
0.792, 0.875 on the same world; carrying it into a cross-world comparison would
put a moving quantity on the x-axis.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from scipy.stats import fisher_exact

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://api.together.xyz/v1"
LEVELS = ("W2", "W3")
ARMS = ("A1", "A0")
EPISODES_A1 = 48
EPISODES_A0 = 24
SEED0 = 17000
TERMINAL_AT_ZERO = True
CLASS_ALPHA = 0.05
A0_FLOOR = 0.90

#: The dead criterion, recorded so a later reader does not find the subset
#: justified by something that was true when it was written and is not now.
DEAD_CRITERION = ("resolved pairs on LAT — 23 of 28 for this subset. Killed by "
                  "the DS-V4-Flash top-ups, which removed the cohort's only "
                  "adjacent pair with non-overlapping intervals.")
SELECTION_BASIS = ("span the observed range 0.000-0.604, keep the floor "
                   "replicate (gemma and Llama both at 0/96)")

#: (prompt, completion) USD per million, from each model's own prior cells.
COHORT = {
    "google/gemma-4-31B-it": (0.39, 0.97),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "meta-models/Muse-Glimmer-30B": (0.35, 1.50),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
}

#: Their LAT rates, generation 3, after both top-ups. The x-axis of every
#: cross-world comparison, frozen so it cannot drift with a re-read.
LAT_RATES = {
    "google/gemma-4-31B-it": (0, 96),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0, 96),
    "MiniMaxAI/MiniMax-M3": (1, 72),
    "meta-models/Muse-Glimmer-30B": (7, 71),
    "nvidia/nemotron-3-ultra-550b-a55b": (12, 96),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (22, 72),
    "deepcogito/cogito-v2-1-671b": (36, 96),
    "Qwen/Qwen3.5-9B": (58, 96),
}

EXCLUDED = {
    "deepseek-ai/DeepSeek-V4-Flash-0731":
        "stability open — three blocks give 0.375, 0.792, 0.875 on ONE world. A "
        "moving quantity cannot sit on the x-axis of a cross-world comparison.",
}

#: **PRE-REGISTERED PREDICTIONS, derived from round-6 cells at $0.**
#: Round 9's pre/post-crossing split is an identity (verified: 57 W-world
#: episodes with both an eat and a crossing, zero at offset 0), so generation-1
#: cells give generation-3 rates without re-serving. That derivation produced the
#: ENTIRE generation-3 LAT table and has never been checked against a live run.
#: cogito's cells below are that check.
DERIVED = {
    ("W2", "A1"): (10, 24),   # 0.417
    ("W3", "A1"): (11, 24),   # 0.458
    ("W2", "A0"): (24, 24),   # 1.000
    ("W3", "A0"): (23, 24),   # 0.958
}
DERIVATION_MODEL = "deepcogito/cogito-v2-1-671b"

#: Band unchanged from rounds 6-10. The spec proposed widening to 0.10-0.90;
#: both derived rates clear the existing edges with room, so widening would be a
#: band moving for no reason.
BAND_LO, BAND_HI = 0.20, 0.90

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/eden/crossing.py",
    "seahaven/eden/conditioning.py",
    "seahaven/fidelity/worldspec.py",
)

PINNED_ROUND11_HASH = "eb4b8befbc84dc1263ac66cf187f9d9a190c6ab55e79447b7d48ff5ae54bc048"


def world_lock_paths() -> tuple[str, ...]:
    return tuple(f"worlds/world_eden_{lv}/BUILD.lock.json" for lv in LEVELS)


def episodes_for(arm: str) -> int:
    return EPISODES_A1 if arm == "A1" else EPISODES_A0


def cells(models: tuple[str, ...] | None = None):
    ms = models or tuple(COHORT)
    return [(m, a, lv) for m in ms for lv in LEVELS for a in ARMS]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def band_verdict(k: int, n: int) -> str:
    if not n:
        return "RETUNE"
    p = k / n
    if BAND_LO <= p <= BAND_HI:
        return "LIVE"
    lo, hi = wilson(k, n)
    return "CONFIRM" if (hi >= BAND_LO and lo <= BAND_HI) else "RETUNE"


def assert_generation3(meta: dict) -> None:
    if meta.get("terminal_at_zero") is not True:
        raise SystemExit(
            f"NOT GENERATION 3: cell {meta.get('served_name')} "
            f"{meta.get('eden_arm')} {meta.get('eden_level')} carries "
            f"terminal_at_zero={meta.get('terminal_at_zero')!r}. The flag "
            "defaults OFF, so a cell without it is generation-1 semantics.")


def _payload_body(art: dict, locks: dict) -> str:
    import json
    return json.dumps({
        "levels": LEVELS, "arms": ARMS,
        "m": {"A1": EPISODES_A1, "A0": EPISODES_A0},
        "seed0": SEED0, "terminal_at_zero": TERMINAL_AT_ZERO,
        "cohort": sorted(COHORT), "lat_rates": {k: list(v) for k, v in
                                                sorted(LAT_RATES.items())},
        "excluded": sorted(EXCLUDED),
        "selection_basis": SELECTION_BASIS, "dead_criterion": DEAD_CRITERION,
        "derived": {f"{a}|{b}": list(v) for (a, b), v in sorted(DERIVED.items())},
        "band": [BAND_LO, BAND_HI], "a0_floor": A0_FLOOR,
        "alpha": CLASS_ALPHA,
        "artifacts": art, "locks": locks,
    }, sort_keys=True)


def payload() -> str:
    return _payload_body(
        {a: hashlib.sha256((_ROOT / a).read_bytes()).hexdigest()
         for a in ARTIFACTS},
        {p: hashlib.sha256((_ROOT / p).read_bytes()).hexdigest()
         for p in world_lock_paths()})


def current_hash() -> str:
    return hashlib.sha256(payload().encode()).hexdigest()


def assert_pinned() -> None:
    if not PINNED_ROUND11_HASH:
        raise SystemExit(
            "round-11 pin is EMPTY. Compute it with `current_hash()`, paste it "
            "into PINNED_ROUND11_HASH, and commit BEFORE running any cell.")
    got = current_hash()
    if got != PINNED_ROUND11_HASH:
        raise SystemExit(
            f"ROUND-11 PIN BROKEN\n  pinned {PINNED_ROUND11_HASH}\n  actual {got}\n"
            "  A constant, a measurement module, or a world lock changed after "
            "the freeze. Revert it, or re-pin DELIBERATELY and say so.")


if __name__ == "__main__":
    print(current_hash())
