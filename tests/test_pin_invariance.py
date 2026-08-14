"""**THE P0 GUARD.** Every pin hash, captured before the `_shared` refactor.

The refactor deduplicates machinery that is copy-pasted across `round2..round13`
— `wilson`, `_fisher`, `mds`, `payload`, `current_hash`, `world_lock_paths`,
`episodes_for`. That is safe *only* because a round module's own source is not
hashed into its pin: `payload()` covers `ARTIFACTS`, the world locks, and the
*values* of the round's constants, and nothing else.

Safe in principle is not safe in fact, so this file makes it checkable. These
literals were recorded **before a single line moved**. If the refactor changes any
of them, it has changed what a round was served under, and the answer is to revert
— never to re-record the literals here.

**The retired digests are the part most at risk**, and the reason they get equal
billing: they recompute from frozen literals *inside* the round modules, which is
exactly the code P0 rewrites. The retirement pattern is the programme's integrity
story — *restore it, do not re-freeze* — so a refactor that quietly broke a
recomputation would break the guarantee everything else rests on.
"""

from __future__ import annotations

import importlib

import pytest

#: Live `current_hash()` per round, captured pre-refactor.
LIVE = {
    "round2":  "08063dcaef2b6bb6e3509baeb36939c5d2ff84c5eee8f8729665cd5323daeb10",
    "round3":  "eccab6b1e9d78c1540f3547f345cd9a4d314745fbd8f518e0ae983e211c8bd6d",
    "round4":  "b6f9260f9d1fd53dd7d3b62c65ee66b0f5dd780111558c601ff939636960a404",
    "round6":  "eacbcb6c0db7fef00b20634e75c335b272a915f0b3331824b1dd2d9c524982e6",
    "round7":  "6523113f3e154be07f2c6593896c268ce7684bb0b016a5edc821e41c0a504fbc",
    "round8":  "6a9049d32c8d930c952949d02c00de7a1e98f9275e2660246316b6a8e5e1b418",
    "round9":  "f523e8d503869a2f42067daee79cb1968c39e7d172ccafa6ed46d3171fd2fd83",
    "round10": "a30a236fc40b4e1a56f392dd30a81ce459bd2b77534f8889df6b3512c9aa8a78",
    "round11": "eb4b8befbc84dc1263ac66cf187f9d9a190c6ab55e79447b7d48ff5ae54bc048",
    "round12": "f9ac9323ede0632770d13387a00a4cc76d16df3231ff46e9243fb31989a7edd1",
    "round13": "668ea92d0c3bf2d335b48f561c71a5a9ab8a97f7939b198df437c80f77a7cd0e",
}

#: The six retired recompute functions, across five rounds.
RETIRED = {
    ("round3", "retired_bracket_hash"):
        "84f093f3a766b117b895dfe81c28e4990e00bf5ee30661f4a39a993980048bcb",
    ("round3", "retired_lat_hash"):
        "5cda15238a1d5fa377409e7ee014587747927e7669b0c5eaef72b974d8b72888",
    ("round4", "retired_comp_hash"):
        "f39b691c34b9c31a12277f10679bde7c410af98f585be0e017b7bb8ab1d533d4",
    ("round6", "retired_w_hash"):
        "25388e91a7839db6e995fbd5259c5992a5742837666078695f789064b7a73469",
    ("round7", "retired_g2_hash"):
        "6286e0b8d7bfc24e674838412e9480d9c835504116b7e374782bac7872e7a6c7",
    ("round8", "retired_g2_hash"):
        "70bbaa84846e9e07b258c4a7cc81c7a75d7746f931f83274a20a0f46950f0192",
}

#: The files whose BYTES are hashed by at least one live pin, with their
#: pre-refactor digests. The refactor may not touch these; every shared helper is
#: a NEW file instead.
FROZEN_ARTIFACTS = {
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
    "seahaven/eden/intent.py":
        "a1074aec3523d512ed3fce926be8f25fe073f5f013d130804c2ead01990584c1",
    "seahaven/fidelity/worldspec.py":
        "bba9d54e9e13e31e260efa11d01ba1f2c7007653d62c42870bda5ab7a369bb85",
}


def _mod(name):
    return importlib.import_module(f"seahaven.eden.{name}")


@pytest.mark.parametrize("name,want", sorted(LIVE.items()))
def test_live_pin_hash_is_UNCHANGED_by_the_refactor(name, want):
    assert _mod(name).current_hash() == want, (
        f"{name}'s pin moved. The refactor changed what that round was served "
        "under. REVERT — do not re-record the literal in this file.")


@pytest.mark.parametrize("name,fn", sorted(RETIRED))
def test_retired_digest_still_RECOMPUTES(name, fn):
    """A retired digest is the record of a closed round. `restore it, do not
    re-freeze` is the rule; this is the check that notices."""
    assert getattr(_mod(name), fn)() == RETIRED[(name, fn)], (
        f"{name}.{fn}() no longer recomputes. The record of what that round was "
        "served under has been disturbed by the refactor.")


#: Which rounds are still open. A CLOSED round's `assert_pinned` raises
#: unconditionally and its `current_hash()` has DRIFTED from its `PINNED_*`
#: literal — that drift is *why* it was retired, and the retired digest is the
#: preserved record. Conflating the two states is a mistake this file made once.
OPEN = {"round2", "round9", "round10", "round11", "round12", "round13"}
CLOSED = {"round3", "round4", "round6", "round7", "round8"}


@pytest.mark.parametrize("name", sorted(OPEN))
def test_an_OPEN_rounds_hash_equals_its_own_frozen_constant(name):
    """For an open round the live payload and the pinned literal must agree —
    that is what `assert_pinned` enforces, restated so the two cannot drift."""
    m = _mod(name)
    pinned = [getattr(m, a) for a in dir(m)
              if a.startswith("PINNED_") and isinstance(getattr(m, a), str)
              and len(getattr(m, a)) == 64]
    assert pinned, f"{name} is open but carries no PINNED_* literal"
    assert m.current_hash() in pinned
    m.assert_pinned()


@pytest.mark.parametrize("name", sorted(CLOSED))
def test_a_CLOSED_round_REFUSES_and_its_hash_is_expected_to_have_drifted(name):
    """The other half of the invariant, and the one worth stating out loud.

    A closed round's `current_hash()` no longer matches its `PINNED_*` literal,
    because artifacts moved after it was frozen. That is correct and expected —
    the drift is the reason it was retired. What must hold is that it REFUSES to
    be used, and that its retired digest still recomputes (tested above).
    """
    m = _mod(name)
    with pytest.raises(SystemExit):
        m.assert_pinned()
    pinned = [getattr(m, a) for a in dir(m)
              if a.startswith("PINNED_") and isinstance(getattr(m, a), str)
              and len(getattr(m, a)) == 64]
    if pinned:
        assert m.current_hash() not in pinned, (
            f"{name} is closed but its live hash matches its pin again — either "
            "an artifact was reverted or the round should not be closed")


@pytest.mark.parametrize("rel,want", sorted(FROZEN_ARTIFACTS.items()))
def test_no_frozen_artifact_is_EDITED_by_the_refactor(rel, want):
    """**Names the file, so a failure says what to revert.**

    The live-pin tests above already fail if any of these bytes move, but they
    fail with a round name and a hash, not a filename. This is the check that
    would have said "you edited `intent.py`" when `route_to_zero` was appended to
    it — the mistake round 13's pin caught once already, at the cost of a red
    suite and ten minutes working out which artifact had moved.
    """
    import hashlib
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / rel
    assert p.exists(), f"{rel} vanished"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == want, (
        f"{rel} was EDITED. It is hashed into at least one live pin, so this "
        "breaks a freeze. Put the change in a NEW module instead.")
