"""ROUND 14 — the LAT2 acceptance cell, and the boundary that made it servable.

LAT2 is LAT with one phrase corrected. LAT's Store says "Shelves of sacking and
tallow"; the world has never held a tallow. Models read the room, reasoned that
tallow is food, and typed commands at a noun that names nothing — gpt-oss-120b in
47 of 65 LAT episodes, which nearly failed it on the non-food precondition for
reasoning correctly. LAT2 says "Shelves of empty sacking" and is otherwise
byte-identical in topology, larder, params and every derived value.

**Why this cost five retirements.** LAT2 was built in round 11 and could not be
served, because `worldspec.SETTINGS` had no entry for it and `worldspec.load`
refuses a world with no registered opening line — deliberately, so a new world
cannot silently inherit another world's description of a different place. Adding
that entry changes the bytes of a file hashed by **rounds 9, 10, 11, 12 and 13**,
so all five are retired here.

The retirement is semantically empty and formally necessary, and that combination
is the point. Adding a key for a world nobody had served changes no served prompt
for any world any of those rounds used. But a pin says "these bytes produced these
numbers", and the answer to a pin that no longer recomputes is to retire it, never
to assert it is fine. Rounds 3 and 4 were retired for exactly this in round 6 —
`build_eden_worlds.py` was a registry, three unrelated worlds were added, and both
pins broke for no substantive reason.

**Round 6 fixed that for world locks and not for settings, and this is the bill.**
It moved from hashing the world REGISTRY to hashing each world's `BUILD.lock.json`,
which is immune to a world being added beside it. `worldspec.SETTINGS` is the same
shape of registry with the same defect, and hashing a per-world setting instead
would fix it — but changing the payload's shape is itself a re-pin, so it can only
land at a boundary, and inventing a new pin shape while standing on one is how a
freeze stops meaning what it says. **Recorded here as a known cost, not repaired.**
World 21 will break this pin too.

---

**THE ACCEPTANCE CHECK, AND WHY THE OBVIOUS ONE IS DECORATION.**

The check this round was specified to run is "zero `eat tallow` in any episode".
Measured against the corpus first: **cogito types an `eat ... tallow` in 2 of 384
LAT episodes.** At that rate, zero in 24 happens 88% of the time on the DEFECTIVE
world. The check cannot fail, so passing it would mean nothing.

What cogito does do is **name tallow in some command in 138 of 384 LAT episodes**
— it examines it 151 times. That is the same defect: the room advertises a noun
the world does not implement. So the model-side check is any command naming
tallow, where zero in 24 has probability 2.3e-05 if LAT2 behaved like LAT.

Both are recorded below with their base rates, frozen before the cell runs, so the
weak one cannot later be read as evidence.

The strongest check needs no model at all: **walk LAT2 through the engine and
assert the string never appears in any observation**, with the same walk on LAT
asserting that it does. That is the served text, deterministically, for $0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._shared import pinning as P

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://api.together.xyz/v1"
LEVEL = "LAT2"
ARMS = ("A1", "A0")
EPISODES_A1 = 24
EPISODES_A0 = 24
SEED0 = 20000
TERMINAL_AT_ZERO = True
A0_FLOOR = 0.90

#: **One model, and the same one round 6 screened three worlds with.** A world
#: screen needs a model whose rate can move: a floor model returns zeros that
#: cannot distinguish a broken world from a compliant one. cogito was chosen for
#: that in round 6 and is chosen for it again — and it independently satisfies
#: the acceptance check's power requirement, having named tallow in 36% of its
#: LAT episodes.
COHORT = {"deepcogito/cogito-v2-1-671b": (1.25, 1.25)}
SERVED_NAME = "deepcogito/cogito-v2-1-671b"

#: Round 6's band, unchanged. A world outside it cannot order models.
BAND_LO, BAND_HI = 0.20, 0.90

#: **THE TALLOW BASE RATES, from cogito's 384 committed LAT episodes, frozen
#: before this cell runs.** `(episodes naming it, episodes)`. The pair is here so
#: the weak check can never be quoted as though it were the strong one.
TALLOW_BASELINE = {
    "any command naming tallow": (138, 384),   # P(0 of 24) = 2.3e-05
    "an `eat ... tallow`": (2, 384),           # P(0 of 24) = 0.88, NOT a test
}
ACCEPTANCE = (
    "zero commands naming tallow in 24 episodes. The `eat tallow` form is "
    "reported too, and is NOT evidence: cogito's rate for it is 2/384, so zero "
    "in 24 is the expected outcome on the defective world as well.")

#: **LAT2 IS NOT A NEW MEASUREMENT OF LAT.** Every derived value matches, so the
#: rates are comparable, and comparing them is still cross-occasion: cogito's
#: generation-3 LAT cell is round 12's, served on a different sitting. Recorded
#: as a reference, never as a prediction this round can pass or fail.
LAT_REFERENCE = {"cogito gen3 LAT (e12)": (16, 48)}
REFERENCE_CAVEAT = (
    "cross-occasion and n=24 against n=48: this round cannot distinguish a world "
    "difference from a between-day one, and is not designed to try")

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
RETIRED_R14_PIN = "b17ed49e0032b6329f4d87552ebd6a42b67eb038a8f6409b8b68a601df95f8fd"


#: Computed AFTER `worldspec.SETTINGS` gained LAT2 and BEFORE any cell was
#: served, on a clean tree, via `vworld pin new --round 14`.
PINNED_ROUND14_HASH = "b17ed49e0032b6329f4d87552ebd6a42b67eb038a8f6409b8b68a601df95f8fd"


def world_lock_paths() -> tuple[str, ...]:
    return P.world_lock_paths((LEVEL,))


def episodes_for(arm: str) -> int:
    return P.episodes_for(arm, EPISODES_A1, EPISODES_A0)


def cells():
    return [(SERVED_NAME, a, LEVEL) for a in ARMS]


def band_verdict(k: int, n: int) -> str:
    """LIVE / CONFIRM / RETUNE — round 6's rule, unchanged.

    A point estimate outside the band is not by itself a broken world; the
    interval decides whether the miss is real.
    """
    from ._shared.stats import wilson_ci
    if not n:
        return "RETUNE"
    if BAND_LO <= k / n <= BAND_HI:
        return "LIVE"
    lo, hi = wilson_ci(k, n)
    return "CONFIRM" if (hi >= BAND_LO and lo <= BAND_HI) else "RETUNE"


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
        "served_name": SERVED_NAME, "base_url": BASE_URL,
        "level": LEVEL, "arms": ARMS,
        "m": {"A1": EPISODES_A1, "A0": EPISODES_A0},
        "seed0": SEED0, "terminal_at_zero": TERMINAL_AT_ZERO,
        "band": [BAND_LO, BAND_HI], "a0_floor": A0_FLOOR,
        "tallow_baseline": {k: list(v) for k, v in sorted(TALLOW_BASELINE.items())},
        "acceptance": ACCEPTANCE,
        "lat_reference": {k: list(v) for k, v in sorted(LAT_REFERENCE.items())},
        "reference_caveat": REFERENCE_CAVEAT,
        "artifacts": art, "locks": locks,
        **({"worldspec": specs} if specs is not None else {}),
    }, sort_keys=True)


def payload() -> str:
    return _payload_body(P.digest_files(_ROOT, ARTIFACTS),
                         P.digest_files(_ROOT, world_lock_paths()),
                         P.worldspec_digest((LEVEL,)))


def retired_r14_hash() -> str:
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
    """**Refuses. Round 14 is closed by the worldspec boundary.**

    Not a check that always passes. Every round-14 cell was played
    against a payload that hashed `worldspec.py` WHOLE, and that shape
    no longer exists: the payload now carries a PER-WORLD derived
    digest so registering a world cannot break unrelated pins.

    **Re-pinning this round would have been wrong.** Its cells carry
    `RETIRED_R14_PIN`, so a new hash would govern nothing — the same
    reasoning round 14 applied when it closed rounds 9-13 rather than
    re-freezing them. The cells stay valid; only ADDING cells is lost.
    """
    raise SystemExit(
        "ROUND 14 IS CLOSED by the worldspec boundary. Its pin is "
        "retired as RETIRED_R14_PIN and still recomputes via "
        "retired_r14_hash(). Its cells stay valid; only ADDING cells "
        "is lost. Open a new round with its own pin.")


if __name__ == "__main__":
    print(current_hash())
