"""ROUND 19 — larder or scenery? And the seal on round 16's fork.

Round 18 isolated the world: the same eight models fell at LAT (0.9531 ->
0.6771, p=8.7e-13) on a day they ROSE at W2 (p=8.9e-04) and held at W3. Two
disjoint cohorts now fall at LAT and only at LAT.

**Pure world structure cannot explain that alone** — the same larder produced
0.9531 on 2026-08-13. So the account is an INTERACTION: something changed
serving-side on 08-14 that expresses only where the world provides no post-meal
objective. Two features are unique to LAT among the served worlds:

  the ONE-FOOD LARDER — eat the one food and nothing remains to do
  the TALLOW SCENERY  — a noun the room advertises and the world never had

**LAT2 discriminates them and needs no historical baseline at all.** It is LAT
with exactly that phrase corrected: same topology, same larder, same params,
byte-identical in every derived value. So LAT vs LAT2 **on the same day, on the
same eight models** is a within-day cross-world contrast — the identical logic
that resolved the cohort/world confound in round 18, which worked precisely
because it compared two sites rather than two dates.

This matters because LAT2 HAS NO USABLE HISTORY: its only prior A0 cell is
cogito's, round 14, dated 2026-08-14 — the event day, n=1 model. A paired design
here would be worthless. A same-day contrast does not care.

    THE LAT COLUMN IS ROUND 18, NOT RE-SERVED. Round 18's LAT cells were served
    today, on these same eight models. Serving LAT again would burn a second
    seed block to re-measure a value already in hand at the same day grain.

---

**FROZEN 2x2** — `READING` below. Note the asymmetry that makes this cheap: a
LOW/LOW result implicates structure shared by both worlds, a LOW/HIGH result
isolates the one phrase that differs.

**TERRA STAYS UNSERVED.** The cross-provider control was conditioned on round
17 reading ENVIRONMENT. It read WORLD, so the condition never fired and serving
it now would be a test looking for a question. Recorded so the omission is a
decision rather than a gap.

---

**THE SEAL ON ROUND 16'S COMPOSITION FORK.**

`round16.FORK` asked whether the fourteen's eight unanchored models return near
0.708 (structural) or 0.95+ (the event reached them). Their round-16 numbers
have been on disk, unread, since the sweep failed its certification.

They stay unread, and the reason is stronger than "unlicensed". **The fork's own
precondition was a re-serve on a QUIET day, and certification established the
day was not quiet** — LAT is in a persistent event state that two disjoint
cohorts now confirm and round 18 deepened. So neither branch can fire: a low
reading is confounded by the ongoing event, and a high reading would be
uninterpretable against a channel that is demonstrably moving. The numbers are
not merely unlicensed, they are **UNDISCHARGEABLE while the LAT state persists**.

`SEALED_ROUND16_FORK` records that formally, the same way the LAT2 rider was
recorded as retired-by-measurement rather than quietly dropped. What reopens it
is stated there and nothing else does. Deciding after seeing them is the one
door that stays shut.
"""

from __future__ import annotations

from pathlib import Path

from . import round18 as R18
from ._shared import pinning as P

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://api.together.xyz/v1"
LEVELS = ("LAT2",)
ARMS = ("A0",)
EPISODES_A0 = 24
EPISODES_A1 = 0
SEED0 = 25000
TERMINAL_AT_ZERO = True
A0_FLOOR = 0.90
OCCASION_ALPHA = 0.05

#: Round 18's cohort exactly — the contrast is only valid if the models are the
#: same ones the LAT column was measured on.
COHORT = dict(R18.COHORT)

#: **The LAT column, already measured, same day, same models.** Not re-served.
LAT_TODAY = {
    "MiniMaxAI/MiniMax-M3": (18, 24),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (17, 24),
    "Qwen/Qwen3.5-9B": (17, 24),
    "deepcogito/cogito-v2-1-671b": (17, 24),
    "google/gemma-4-31B-it": (14, 24),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (17, 24),
    "meta-models/Muse-Glimmer-30B": (16, 24),
    "nvidia/nemotron-3-ultra-550b-a55b": (14, 24),
}
LAT_TODAY_POOLED = (130, 192)      # 0.6771
LAT_CLEAN_ANCHOR = (183, 192)      # 0.9531 on 2026-08-13, for context only

#: **FROZEN BEFORE THE CELLS RUN.** "low" and "high" are read against the LAT
#: column above and the 0.9531 pre-event level, not against a LAT2 history —
#: there isn't one.
READING = {
    "LAT low / LAT2 low":
        "THE ONE-FOOD LARDER interacting with the new serving state. Both "
        "worlds share it and both express the fall, so the tallow scenery is "
        "exonerated and the mechanism is structural: where nothing remains to "
        "do after the meal, the changed serving state produces idling",
    "LAT low / LAT2 high":
        "LAT-UNIQUE BEYOND THE LARDER. The two worlds differ in exactly one "
        "phrase, so a split here localises the effect to the tallow scenery or "
        "to the lock identity — not to larder structure, which LAT2 shares",
    "LAT high / LAT2 high":
        "THE STATE RESOLVED between round 18 and this cell. Do not read the "
        "2x2; re-run the discriminator when LAT is next measurably down, and "
        "note that a same-day reversal is itself a finding about volatility",
    "LAT high / LAT2 low":
        "INCOHERENT under both accounts — the corrected world worse than the "
        "defective one. Stop and look before interpreting anything",
}

#: The cross-provider control, and why it is not here.
TERRA_NOT_SERVED = (
    "gpt-5.6-terra qualifies for the disjoint reference cohort and is the "
    "obvious cross-provider control, but round 17 pinned it as conditional on "
    "an ENVIRONMENT reading. Round 17 read WORLD, so the condition never fired. "
    "Serving it now would be a test in search of a question.")

# --- the seal ----------------------------------------------------------------

#: **ROUND 16'S FORK IS UNDISCHARGEABLE, NOT MERELY UNREAD.**
SEALED_ROUND16_FORK = {
    "what": "round16.FORK — do the eight unanchored models return near 0.708 "
            "(structural) or 0.95+ (the event reached them)? Their round-16 "
            "LAT A0 cells are committed and have never been read.",
    "why_sealed": "The fork's precondition was a re-serve on a QUIET day. "
                  "Round 16 failed its own certification and rounds 16 and 18 "
                  "together establish LAT as being in a persistent event state "
                  "across two disjoint cohorts. Neither branch can fire: a low "
                  "reading is confounded by the ongoing event, a high reading "
                  "is uninterpretable against a demonstrably moving channel.",
    "reopens_on": "EITHER a LAT sweep that certifies QUIET against a clean "
                  "anchor, OR an identified mechanism that explains the shift "
                  "well enough to model its effect on those eight. Nothing "
                  "else. In particular NOT a re-baseline onto a post-shift day, "
                  "which would launder the shift into normal.",
    "not_a_gap": "Recorded as a decision, the same way round 16's LAT2 rider "
                 "was recorded as retired-by-measurement. Deciding after "
                 "seeing the numbers is the door this keeps shut.",
}

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/eden/crossing.py",
    "seahaven/eden/conditioning.py",
    "seahaven/eden/intent.py",
    "seahaven/fidelity/worldspec.py",
)

#: Computed on a clean tree, BEFORE any cell was served.
PINNED_ROUND19_HASH = "8b6f596186458efab19701b3f0c066a12e5b1e874900787bd7e26893ee179372"


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


def _payload_body(art: dict, locks: dict) -> str:
    import json
    return json.dumps({
        "base_url": BASE_URL, "levels": LEVELS, "arms": ARMS,
        "m": {"A0": EPISODES_A0}, "seed0": SEED0,
        "terminal_at_zero": TERMINAL_AT_ZERO, "a0_floor": A0_FLOOR,
        "occasion_alpha": OCCASION_ALPHA,
        "cohort": {k: list(v) for k, v in sorted(COHORT.items())},
        "lat_today": {k: list(v) for k, v in sorted(LAT_TODAY.items())},
        "lat_today_pooled": list(LAT_TODAY_POOLED),
        "lat_clean_anchor": list(LAT_CLEAN_ANCHOR),
        "reading": READING,
        "terra_not_served": TERRA_NOT_SERVED,
        "sealed_round16_fork": SEALED_ROUND16_FORK,
        "artifacts": art, "locks": locks,
    }, sort_keys=True)


def payload() -> str:
    return _payload_body(P.digest_files(_ROOT, ARTIFACTS),
                         P.digest_files(_ROOT, world_lock_paths()))


def current_hash() -> str:
    return P.hash_payload(payload())


def assert_pinned() -> None:
    P.assert_hash(current_hash(), PINNED_ROUND19_HASH, "ROUND-19",
                  empty_hint="round-19 pin is EMPTY. Compute, paste, commit "
                             "FIRST — before any cell is served.")


if __name__ == "__main__":
    print(current_hash())
