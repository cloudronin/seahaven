"""ROUND 12 — measure the four LAT rows that were only ever DERIVED.

gemma, Llama, nemotron and cogito have **no generation-3 LAT episodes**. Their LAT
rates come from round 9's pre-crossing identity: generation-1 cells with
post-crossing eats deleted. That identity built the whole generation-3 LAT table
and, until round 11, had never been checked. It now has one exact hit (W2) and one
0.29 miss (W3).

**These are direct measurements and a NEW ROUND, not a round-10 top-up.** Round
10's corpus is the derived table; pooling a measurement with the derivation it
replaces would destroy the comparison this round exists to make.

**The four derived values are frozen below BEFORE any cell is served.** They are
what the measurements are tested against.

**The confound is named in advance and is not resolvable by this round.** A
deviation is consistent with the derivation being wrong AND with a
between-occasion behaviour change — and the generation-1 LAT cells that would
separate them are exactly the ones being replaced. The serving diagnostic measured
a 0.319 level shift between days on one model, so this is not hypothetical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from scipy.stats import fisher_exact

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://api.together.xyz/v1"
LEVEL = "LAT"
ARMS = ("A1", "A0")
EPISODES_A1 = 48
EPISODES_A0 = 24
SEED0 = 18000
TERMINAL_AT_ZERO = True
CLASS_ALPHA = 0.05
A0_FLOOR = 0.90

COHORT = {
    "google/gemma-4-31B-it": (0.39, 0.97),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
}

#: **THE PREDICTIONS, frozen before any episode.** Round 9's derived generation-3
#: LAT rates for exactly these four models. Checks three through six of the
#: identity; W2 was check one (exact) and W3 check two (off by 0.29).
DERIVED = {
    "google/gemma-4-31B-it": (0, 96),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0, 96),
    "nvidia/nemotron-3-ultra-550b-a55b": (12, 96),
    "deepcogito/cogito-v2-1-671b": (36, 96),
}

#: What the round can and cannot settle, fixed with the predictions.
CONFOUND = ("a deviation is consistent with the derivation being wrong AND with "
            "a between-occasion behaviour change; the generation-1 LAT cells "
            "that would separate them are the ones being replaced")

#: cogito's LAT funnel, from generation-1 episodes, is what made it the
#: structural control in the serving diagnostic. Round 11 already showed the
#: property is world-specific (44/48, conv 0.690 on W2; 33/48, 0.444 on W3).
#: Whether it even holds on directly measured LAT cells is open.
COGITO_G1_LAT_CLAIM = {"took_rate": 1.0, "attempt_to_eat": 1.0}

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/eden/crossing.py",
    "seahaven/eden/conditioning.py",
    "seahaven/fidelity/worldspec.py",
)

PINNED_ROUND12_HASH = "f9ac9323ede0632770d13387a00a4cc76d16df3231ff46e9243fb31989a7edd1"


def world_lock_paths() -> tuple[str, ...]:
    return (f"worlds/world_eden_{LEVEL}/BUILD.lock.json",)


def episodes_for(arm: str) -> int:
    return EPISODES_A1 if arm == "A1" else EPISODES_A0


def cells():
    return [(m, a, LEVEL) for m in COHORT for a in ARMS]


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


def assert_generation3(meta: dict) -> None:
    if meta.get("terminal_at_zero") is not True:
        raise SystemExit(
            f"NOT GENERATION 3: {meta.get('served_name')} "
            f"{meta.get('eden_arm')} carries terminal_at_zero="
            f"{meta.get('terminal_at_zero')!r}; the flag defaults OFF.")


def _payload_body(art: dict, locks: dict) -> str:
    import json
    return json.dumps({
        "level": LEVEL, "arms": ARMS,
        "m": {"A1": EPISODES_A1, "A0": EPISODES_A0},
        "seed0": SEED0, "terminal_at_zero": TERMINAL_AT_ZERO,
        "cohort": sorted(COHORT),
        "derived": {k: list(v) for k, v in sorted(DERIVED.items())},
        "confound": CONFOUND, "cogito_g1_claim": COGITO_G1_LAT_CLAIM,
        "a0_floor": A0_FLOOR, "alpha": CLASS_ALPHA,
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
    if not PINNED_ROUND12_HASH:
        raise SystemExit("round-12 pin is EMPTY. Compute, paste, commit FIRST.")
    got = current_hash()
    if got != PINNED_ROUND12_HASH:
        raise SystemExit(
            f"ROUND-12 PIN BROKEN\n  pinned {PINNED_ROUND12_HASH}\n"
            f"  actual {got}\n  Revert, or re-pin DELIBERATELY and say so.")


if __name__ == "__main__":
    print(current_hash())
