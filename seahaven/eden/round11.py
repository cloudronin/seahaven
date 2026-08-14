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


#: **ROUND 11 IS CLOSED. Retired at the LAT2 boundary — see `round14.py`.**
#:
#: LAT2 is LAT with one phrase corrected: the Store advertised "tallow", which
#: the world has never implemented, and models typed commands at it. LAT2 could
#: not be SERVED because `worldspec.SETTINGS` had no entry for it, and
#: `worldspec.py` is a hashed artifact here. Adding that entry broke this pin.
#:
#: **The change is semantically empty and the retirement is still correct**, and
#: that combination is the point. A key for a world nobody had served alters no
#: prompt this round used. But a pin says "these bytes produced these numbers",
#: and the answer to a pin that no longer recomputes is to retire it — never to
#: assert it is fine. Rounds 3 and 4 were retired for exactly this in round 6.
#:
#: Round 11 is the three-world screen. Its W2/W3 cells are the only
#: occasion-clean cross-world pair the programme has, which is exactly why
#: the record of what they ran under has to survive the close.
RETIRED_R11_PIN = PINNED_ROUND11_HASH

RETIRED_R11_SHA256 = {
    "seahaven/eden/simulate.py":
        "8c0d05f23eceba0bbe1c76f9624591ff786424340b1ae3fb1f40c7cd711cbc58",
    "seahaven/eden/outcome.py":
        "a29e10f6fbcaa05e7c8777b4d332d31cd9399d4fdba2ccdab58574048aa439a3",
    "seahaven/eden/manifest.py":
        "3ee32edfcc83ac15e9a003bcefa226cd610c458600e0b16e497c0aceb33dd79c",
    "seahaven/eden/crossing.py":
        "73c08d2a93d90f9ef0757559b163e48149692d6618c8752a920c6061807748a7",
    "seahaven/eden/conditioning.py":
        "a7a91dcce4da9c220298e35ab565057d3a08de02129fc523253be4fa0b7e16f7",
    "seahaven/fidelity/worldspec.py":
        "bba9d54e9e13e31e260efa11d01ba1f2c7007653d62c42870bda5ab7a369bb85",
}
RETIRED_R11_LOCKS = {
    "worlds/world_eden_W2/BUILD.lock.json":
        "4d009ec70ffe6331ddd2da78428562570974aa47656ecc28665c0824bf740cd2",
    "worlds/world_eden_W3/BUILD.lock.json":
        "7b248708afb5b2daa0d297f4bf5344aeea57487fbb727d2487b012021edabc80",
}


def retired_r11_hash() -> str:
    """Reproduces `RETIRED_R11_PIN` from the frozen snapshot, permanently."""
    return hashlib.sha256(
        _payload_body(dict(RETIRED_R11_SHA256),
                      dict(RETIRED_R11_LOCKS)).encode()).hexdigest()

def assert_pinned() -> None:
    """**Refuses. Round 11 is closed.** Not a check that always passes."""
    raise SystemExit(
        "ROUND 11 IS CLOSED. Its pin is retired as RETIRED_R11_PIN and still "
        "recomputes via retired_r11_hash(). Round 14 added LAT2 to "
        "worldspec.SETTINGS, a hashed artifact here, so every round-11 cell was "
        "played against bytes that no longer exist on disk. Its cells stay valid; "
        "only ADDING cells is lost. Open a new round with its own pin.")


if __name__ == "__main__":
    print(current_hash())
