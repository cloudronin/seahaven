"""ROUND 18 — hold the cohort fixed and move only the world.

Round 16 found LAT falling for the fourteen's anchored six: 0.9712 -> 0.8194 ->
0.7292 across three days. Round 17 found W2 and W3 **rising** for a disjoint
eight on the third of those days, 0.887 -> 0.974 and 0.937 -> 0.984, with COMP
suggestive-but-underpowered in the same upward direction.

**That pair of results is confounded, and the confound is the whole reason this
round exists.** Round 17 changed the cohort AND the world at once. "LAT is
broken" and "the fourteen are broken" both fit the data exactly as well, and
nothing in either round separates them.

So: the SAME eight models, at LAT, on the same day they were measured at W2, W3
and COMP. Cohort held fixed, world varied, one thing moving.

  THE EIGHT FALL AT LAT  -> the world. Cohort is held fixed, so the fall cannot
  be a property of who was measured. LAT-specific and persistent, and the fault
  is in that world rather than the provider.

  THE EIGHT HOLD AT LAT  -> the cohort. The fourteen moved, not LAT. This
  reopens the composition question from the opposite direction and makes round
  16's trace a fact about those models rather than about that world.

---

**DIRECTION IS NOT THE SAME AS CHANGE, AND THIS ROUND EXISTS BECAUSE OF THAT.**
The paired Fisher is two-sided: EVENT means *moved*. Reading EVENT as *fell*
inverted round 17's conclusion on its first pass — the co-located worlds had
gone UP, and the matched branch said they had fallen. `READING` below is keyed
on direction explicitly, and every branch names which way.

**THE REFERENCE IS STILL DISJOINT.** This cohort is the nine the admission gate
scores, minus the one on another provider — none of them is among the fourteen
under test. That satisfies `round17.REFERENCE_DISJOINT`, which was pinned as a
general rule precisely so this round would not have to think about it again.

**COMP'S 8-OF-8 WAS AN INVALID COMPARISON AND IS NOT REPEATED HERE.** Four of
its eight baselines were gen1 round-4 cells, and the programme forbids pooling
across a generation boundary. Restricted to valid gen3 pairs it is 4 of 4 rising
at p=0.125 — suggestive, underpowered, and one of those four has a lossy n=12
baseline. It is recorded below at that strength and no more.
"""

from __future__ import annotations

from pathlib import Path

from . import round17 as R17
from ._shared import pinning as P

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://api.together.xyz/v1"
LEVELS = ("LAT",)
ARMS = ("A0",)
EPISODES_A0 = 24
EPISODES_A1 = 0
SEED0 = 24000
TERMINAL_AT_ZERO = True
A0_FLOOR = 0.90
OCCASION_ALPHA = 0.05

#: **Round 17's cohort exactly**, taken from its literal so the two cannot
#: drift. Changing one model would reintroduce the confound this round removes.
COHORT = dict(R17.COHORT)

#: Clean pre-event LAT A0 anchors, event sweeps excluded, `(ate, n)`. All eight
#: are anchored — unlike the fourteen, where only six were.
CLEAN_LAT_ANCHOR = {
    "MiniMaxAI/MiniMax-M3": (23, 24),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (19, 24),
    "Qwen/Qwen3.5-9B": (24, 24),
    "deepcogito/cogito-v2-1-671b": (21, 24),
    "google/gemma-4-31B-it": (24, 24),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (24, 24),
    "meta-models/Muse-Glimmer-30B": (24, 24),
    "nvidia/nemotron-3-ultra-550b-a55b": (24, 24),
}
ANCHOR_POOLED = (183, 192)          # 0.9531, all from 2026-08-13

#: The two prior results this round is here to disentangle, frozen so the
#: comparison does not get recomputed differently later.
ROUND16_LAT_TRACE = {              # the FOURTEEN's anchored six, at LAT
    "2026-08-13": (101, 104),      # 0.9712
    "2026-08-14": (118, 144),      # 0.8194
    "2026-08-15": (105, 144),      # 0.7292
}
ROUND17_RESULT = {                 # THIS cohort, elsewhere, on 2026-08-15
    "W2": {"prior": [165, 186], "now": [187, 192], "direction": "UP",
           "p": 8.95e-04},
    "W3": {"prior": [179, 191], "now": [189, 192], "direction": "UP",
           "p": 1.87e-02},
    "COMP": "4 of 4 valid gen3 pairs rose, two-sided sign p=0.125 — "
            "SUGGESTIVE AND UNDERPOWERED, not evidence. Four further pairs "
            "were excluded as gen1 baselines, and one included pair has a "
            "lossy n=12 baseline that inflates its delta.",
}

#: **FROZEN BEFORE THE CELLS RUN, and keyed on DIRECTION.**
READING = {
    "the eight FALL at LAT":
        "THE WORLD. Cohort is held fixed against round 17, so a fall here "
        "cannot be a property of who was measured. LAT-specific and "
        "persistent; the fault is in that world's lock or scenery, not the "
        "provider — which the three rising sites already made unlikely to be "
        "the provider anyway. LAT needs re-anchoring; W2, W3 and COMP keep "
        "their references",
    "the eight HOLD at LAT":
        "THE COHORT. The fourteen moved and LAT did not. Round 16's trace "
        "becomes a fact about those models rather than about that world, the "
        "composition fork reopens from the opposite direction, and the "
        "'persistent LAT finding' from round 16 is withdrawn to 'persistent "
        "finding about the fourteen'",
    "the eight RISE at LAT":
        "LAT moved WITH the other three sites for this cohort, so the whole "
        "environment improved and only the fourteen fell. Strongest possible "
        "form of the cohort account; treat identically to HOLD but record that "
        "the divergence between cohorts is larger still",
    "mixed":
        "Report per model against its own anchor. No pooled verdict, and no "
        "attribution to world or cohort until a cleaner split exists",
}

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/eden/crossing.py",
    "seahaven/eden/conditioning.py",
    "seahaven/eden/intent.py",
)

#: **THE PRE-BOUNDARY ARTIFACT TUPLE.** `worldspec.py` was hashed WHOLE,
#: and `SETTINGS` inside it is a registry keyed by world — so registering
#: any new world broke this pin for reasons no served prompt depended on.
#: Kept so the retired hash below stays permanently recomputable.
RETIRED_ARTIFACTS = ARTIFACTS + ("seahaven/fidelity/worldspec.py",)

#: The pin as frozen before the worldspec boundary. RESTORED, NOT RE-FROZEN.
RETIRED_R18_PIN = "37bf3d72faabb66e568771018a1a1dffea8595c4afd6eb28b2f0078b6c6cacc6"


#: Computed on a clean tree, BEFORE any cell was served.
PINNED_ROUND18_HASH = "37bf3d72faabb66e568771018a1a1dffea8595c4afd6eb28b2f0078b6c6cacc6"


def world_lock_paths() -> tuple[str, ...]:
    return P.world_lock_paths(LEVELS)


def episodes_for(arm: str) -> int:
    return P.episodes_for(arm, EPISODES_A1, EPISODES_A0)


def cells():
    return [(m, "A0", lv) for m in COHORT for lv in LEVELS]


def assert_generation3(meta: dict) -> None:
    if meta.get("terminal_at_zero") is not True:
        raise SystemExit(
            f"NOT GENERATION 3: {meta.get('served_name')} "
            f"{meta.get('eden_arm')} {meta.get('eden_level')} carries "
            f"terminal_at_zero={meta.get('terminal_at_zero')!r}")


def _payload_body(art: dict, locks: dict,
                  specs: dict | None = None) -> str:
    import json
    return json.dumps({
        "base_url": BASE_URL, "levels": LEVELS, "arms": ARMS,
        "m": {"A0": EPISODES_A0}, "seed0": SEED0,
        "terminal_at_zero": TERMINAL_AT_ZERO, "a0_floor": A0_FLOOR,
        "occasion_alpha": OCCASION_ALPHA,
        "cohort": {k: list(v) for k, v in sorted(COHORT.items())},
        "clean_lat_anchor": {k: list(v)
                             for k, v in sorted(CLEAN_LAT_ANCHOR.items())},
        "anchor_pooled": list(ANCHOR_POOLED),
        "round16_lat_trace": {k: list(v)
                              for k, v in sorted(ROUND16_LAT_TRACE.items())},
        "round17_result": ROUND17_RESULT,
        "reading": READING,
        "artifacts": art, "locks": locks,
        **({"worldspec": specs} if specs is not None else {}),
    }, sort_keys=True)


def payload() -> str:
    return _payload_body(P.digest_files(_ROOT, ARTIFACTS),
                         P.digest_files(_ROOT, world_lock_paths()),
                         P.worldspec_digest(LEVELS))


def retired_r18_hash() -> str:
    """Recomputes the PRE-BOUNDARY payload, byte-for-byte.

    A retired pin is restored, never re-frozen: the digest it produced
    must stay computable forever or the record is a number nobody can
    check."""
    return P.hash_payload(_payload_body(
        P.digest_files(_ROOT, RETIRED_ARTIFACTS),
        P.digest_files(_ROOT, world_lock_paths())))


def current_hash() -> str:
    return P.hash_payload(payload())


def assert_pinned() -> None:
    """**Refuses. Round 18 is closed by the worldspec boundary.**

    Not a check that always passes. Every round-18 cell was played
    against a payload that hashed `worldspec.py` WHOLE, and that shape
    no longer exists: the payload now carries a PER-WORLD derived
    digest so registering a world cannot break unrelated pins.

    **Re-pinning this round would have been wrong.** Its cells carry
    `RETIRED_R18_PIN`, so a new hash would govern nothing — the same
    reasoning round 14 applied when it closed rounds 9-13 rather than
    re-freezing them. The cells stay valid; only ADDING cells is lost.
    """
    raise SystemExit(
        "ROUND 18 IS CLOSED by the worldspec boundary. Its pin is "
        "retired as RETIRED_R18_PIN and still recomputes via "
        "retired_r18_hash(). Its cells stay valid; only ADDING cells "
        "is lost. Open a new round with its own pin.")


if __name__ == "__main__":
    print(current_hash())
