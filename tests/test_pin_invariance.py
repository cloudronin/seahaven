"""**THE PIN GUARD.** Every pin literal, and what is allowed to move.

Written as the P0 guard: every hash captured before the `_shared` refactor, so
that a refactor claiming to change nothing could be checked rather than trusted.
It caught the VetoWorld rename on the same terms — that too moved only
user-facing strings, and these literals are what proves it.

**A ROUND BOUNDARY IS NOT A REFACTOR, and the file now says which is which.**
Round 14 added `world_eden_LAT2` to `worldspec.SETTINGS` so LAT2 could be served
at all. `worldspec.py` is hashed by rounds 6 through 13, so eight rounds'
`current_hash()` moved — five of them live at the time, all five retired in that
commit. That is the designed cost of a boundary, and it is a different event from
a refactor that must be inert.

So there are three kinds of literal here, with three different rules:

- **`PINNED`** — each round's own frozen constant, the record of what it was
  served under. **These may never change, ever.** A change here rewrites history.
- **`RETIRED`** — the eleven retired digests, each recomputed from frozen
  literals *inside* its round module. **These may never change either**, and they
  are the piece most at risk from a refactor, because a body-delegation change
  rewrites exactly the code that recomputes them. The retirement pattern is the
  programme's integrity story — *restore it, do not re-freeze*.
- **`LIVE`** — what `current_hash()` computes today. This moves **only** at a
  declared boundary, and only for the rounds whose hashed artifacts the boundary
  touched. A `LIVE` change with no boundary in the same commit is a bug, and the
  answer is to revert the artifact — never to re-record the literal here.
"""

from __future__ import annotations

import importlib

import pytest

#: Each round's own frozen constant. **Never changes.** The record of what a
#: round was served under does not move because later code did.
PINNED = {
    "round2":  "08063dcaef2b6bb6e3509baeb36939c5d2ff84c5eee8f8729665cd5323daeb10",
    "round3":  "5cda15238a1d5fa377409e7ee014587747927e7669b0c5eaef72b974d8b72888",
    "round4":  "f39b691c34b9c31a12277f10679bde7c410af98f585be0e017b7bb8ab1d533d4",
    "round6":  "25388e91a7839db6e995fbd5259c5992a5742837666078695f789064b7a73469",
    "round7":  "6286e0b8d7bfc24e674838412e9480d9c835504116b7e374782bac7872e7a6c7",
    "round8":  "70bbaa84846e9e07b258c4a7cc81c7a75d7746f931f83274a20a0f46950f0192",
    "round9":  "f523e8d503869a2f42067daee79cb1968c39e7d172ccafa6ed46d3171fd2fd83",
    "round10": "a30a236fc40b4e1a56f392dd30a81ce459bd2b77534f8889df6b3512c9aa8a78",
    "round11": "eb4b8befbc84dc1263ac66cf187f9d9a190c6ab55e79447b7d48ff5ae54bc048",
    "round12": "f9ac9323ede0632770d13387a00a4cc76d16df3231ff46e9243fb31989a7edd1",
    "round13": "668ea92d0c3bf2d335b48f561c71a5a9ab8a97f7939b198df437c80f77a7cd0e",
    "round14": "b17ed49e0032b6329f4d87552ebd6a42b67eb038a8f6409b8b68a601df95f8fd",
    "round15": "d553a4e590979bee3b067c5338ef86c27bb752e051468a22fd58fe1e6760ee6b",
    "round16": "cf72ca823f26e0c238b4deb2db7ce9f0bc858d31558f249d5b0f0c3277ede501",
}

#: Live `current_hash()` per round. Rounds 2, 3 and 4 do not hash `worldspec.py`
#: and so are still at their pre-refactor values, captured before a line moved.
#: Rounds 6-13 moved at the LAT2 boundary and nowhere else.
LIVE = {
    "round2":  "08063dcaef2b6bb6e3509baeb36939c5d2ff84c5eee8f8729665cd5323daeb10",
    "round3":  "eccab6b1e9d78c1540f3547f345cd9a4d314745fbd8f518e0ae983e211c8bd6d",
    "round4":  "b6f9260f9d1fd53dd7d3b62c65ee66b0f5dd780111558c601ff939636960a404",
    "round6":  "5c620fffa328667df2aef4955e7d18c922703cfb068263dff33b7fe5d7022673",
    "round7":  "a7b8b482e4d8168e020489e2d7c4081d9f132e35018a0ed240f6dc4897216265",
    "round8":  "8cb2f0f65451f5b96d72e591d58c2cf233f31285da13186cdae3adc637dbc0a1",
    "round9":  "2692c3530dfb6ff9e3f96858cc84eefd07ae77654dd05af644492f7f871b4732",
    "round10": "bcd2882985fd23f018b73c0aafb8a80712a1d9c204088425c08c12eaa6220b46",
    "round11": "40f3b4fb9ef8966b6a8b2d7825c533dcfaf763202ee9366da5278ed8d2037bf7",
    "round12": "562d94db8c1db9fc43ee139c5b36ff751c624d71f296cf5f23c61b3bb4072888",
    "round13": "096e4172930a018d6ed08a66362010750a0d8006a500d47e3291f5938cd4b428",
    "round14": "b17ed49e0032b6329f4d87552ebd6a42b67eb038a8f6409b8b68a601df95f8fd",
    "round15": "d553a4e590979bee3b067c5338ef86c27bb752e051468a22fd58fe1e6760ee6b",
    "round16": "cf72ca823f26e0c238b4deb2db7ce9f0bc858d31558f249d5b0f0c3277ede501",
}

#: The eleven retired recompute functions, across ten rounds. Six from earlier
#: boundaries; five from the LAT2 boundary, which closed every live pin that
#: hashed `worldspec.py` in one commit.
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
    ("round9", "retired_r9_hash"):
        "f523e8d503869a2f42067daee79cb1968c39e7d172ccafa6ed46d3171fd2fd83",
    ("round10", "retired_r10_hash"):
        "a30a236fc40b4e1a56f392dd30a81ce459bd2b77534f8889df6b3512c9aa8a78",
    ("round11", "retired_r11_hash"):
        "eb4b8befbc84dc1263ac66cf187f9d9a190c6ab55e79447b7d48ff5ae54bc048",
    ("round12", "retired_r12_hash"):
        "f9ac9323ede0632770d13387a00a4cc76d16df3231ff46e9243fb31989a7edd1",
    ("round13", "retired_r13_hash"):
        "668ea92d0c3bf2d335b48f561c71a5a9ab8a97f7939b198df437c80f77a7cd0e",
}

#: The files whose BYTES are hashed by a LIVE pin — rounds 2 and 14 — with their
#: current digests. Editing one breaks a freeze; put the change in a NEW module.
#: `worldspec.py` carries its POST-boundary digest here; its pre-boundary bytes
#: survive inside every retired snapshot that hashed them.
FROZEN_ARTIFACTS = {
    "docs/eden-round2-hosted.md":
        "d36beb6b507b9a3934c1451fe6c94b2aeb7e612de6e749981393db5f9b4f68d2",
    "docs/edenbench-spec.md":
        "27b79a6666f4d8b902f3fa01a6babacbc6b0173a2cd2a695816128aea6cf7e78",
    "seahaven/eden/conditioning.py":
        "a7a91dcce4da9c220298e35ab565057d3a08de02129fc523253be4fa0b7e16f7",
    "seahaven/eden/crossing.py":
        "73c08d2a93d90f9ef0757559b163e48149692d6618c8752a920c6061807748a7",
    "seahaven/eden/intent.py":
        "a1074aec3523d512ed3fce926be8f25fe073f5f013d130804c2ead01990584c1",
    "seahaven/eden/manifest.py":
        "3ee32edfcc83ac15e9a003bcefa226cd610c458600e0b16e497c0aceb33dd79c",
    "seahaven/eden/outcome.py":
        "a29e10f6fbcaa05e7c8777b4d332d31cd9399d4fdba2ccdab58574048aa439a3",
    "seahaven/eden/simulate.py":
        "8c0d05f23eceba0bbe1c76f9624591ff786424340b1ae3fb1f40c7cd711cbc58",
    "seahaven/fidelity/worldspec.py":
        "9746c2ffb8ec7c9eab8a1748c0995c1d18572c92f44fe3195bf46382b7624c69",
    "worlds/build_eden_worlds.py":
        "7b08d0d7d0dda94a88d907636389037caa8cd322344a0985820b1c6df7f8267d",
}


def _mod(name):
    return importlib.import_module(f"seahaven.eden.{name}")


@pytest.mark.parametrize("name,want", sorted(PINNED.items()))
def test_a_rounds_OWN_PIN_LITERAL_never_moves(name, want):
    """**The strongest invariant in the programme.** A round's pin is the record
    of what it was served under; if that literal changes, history was rewritten.
    Unlike `LIVE`, no boundary licenses a change here."""
    m = _mod(name)
    got = [getattr(m, a) for a in dir(m)
           if a.startswith("PINNED_") and isinstance(getattr(m, a), str)
           and len(getattr(m, a)) == 64]
    assert want in got, (
        f"{name}'s PINNED_* literal changed. That is not a boundary and not a "
        "refactor — it rewrites what a served round claims to have run under.")


@pytest.mark.parametrize("name,want", sorted(LIVE.items()))
def test_live_pin_hash_moves_ONLY_at_a_declared_boundary(name, want):
    assert _mod(name).current_hash() == want, (
        f"{name}'s payload moved. If no round boundary is being declared in "
        "this same commit, REVERT the artifact — do not re-record the literal.")


@pytest.mark.parametrize("name,fn", sorted(RETIRED))
def test_retired_digest_still_RECOMPUTES(name, fn):
    """A retired digest is the record of a closed round. `restore it, do not
    re-freeze` is the rule; this is the check that notices."""
    assert getattr(_mod(name), fn)() == RETIRED[(name, fn)], (
        f"{name}.{fn}() no longer recomputes. The record of what that round was "
        "served under has been disturbed.")


#: Which rounds are still open. A CLOSED round's `assert_pinned` raises
#: unconditionally and its `current_hash()` has DRIFTED from its `PINNED_*`
#: literal — that drift is *why* it was retired, and the retired digest is the
#: preserved record. Conflating the two states is a mistake this file made once.
#:
#: **Ten of the twelve are closed.** The LAT2 boundary closed five in one
#: commit; round 2 survives only because it never hashed `worldspec.py`, and
#: round 14 is the boundary round itself.
OPEN = {"round2", "round14", "round15", "round16"}
CLOSED = {"round3", "round4", "round6", "round7", "round8", "round9",
          "round10", "round11", "round12", "round13"}


def test_every_round_is_classified_exactly_once():
    """A round missing from both sets is a round nobody checks."""
    assert OPEN | CLOSED == set(LIVE), (OPEN | CLOSED) ^ set(LIVE)
    assert not OPEN & CLOSED


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


@pytest.mark.parametrize("name", sorted(CLOSED))
def test_a_CLOSED_rounds_RETIRED_digest_equals_the_pin_it_preserves(name):
    """**What retirement is FOR, asserted rather than described.**

    The retired digest is not a new number: it is the round's own pin, made
    permanently recomputable from a frozen snapshot after the live files moved
    past it. If the two ever disagree, retirement has stopped preserving the
    record and has started inventing one.
    """
    m = _mod(name)
    digests = {getattr(m, a)() for a in dir(m)
               if a.startswith("retired_") and callable(getattr(m, a))}
    assert PINNED[name] in digests, (
        f"{name} is closed but no retired digest reproduces its pin "
        f"{PINNED[name][:16]}...; got {[d[:16] for d in digests]}")


@pytest.mark.parametrize("rel,want", sorted(FROZEN_ARTIFACTS.items()))
def test_no_frozen_artifact_is_EDITED_without_a_boundary(rel, want):
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
        f"{rel} was EDITED. It is hashed into a LIVE pin, so this breaks a "
        "freeze. Put the change in a NEW module — or, if this really is a round "
        "boundary, retire every live pin that hashes it in the same commit.")
