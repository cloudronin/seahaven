"""The seal's invariants — every one of them, as tests rather than prose.

**The seal is the only thing standing between this program and the fifth pretty
death.** A partition that merely *claims* coverage protects nothing; these tests
are what make the claim checkable, and they run on every suite invocation so the
seal cannot rot quietly between sessions.

Two of them are the load-bearing pair. `test_seal_hash_is_pinned` catches the
partition or its justification moving. `test_the_firewall_refuses_held_out_repos`
catches Phase 1 code reaching a sealed model. Everything else is the coverage
guarantee the seal exists to protect.
"""

from __future__ import annotations

import pytest

from seahaven.dimensional import seal as S


def test_seal_hash_is_pinned():
    """The whole point. See the module docstring in `seal.py`.

    On failure the question is never "update the pin" — it is what moved, and
    whether the held-out set had been loaded by then.
    """
    assert S.SEAL_HASH == S.PINNED_SEAL_HASH
    S.assert_sealed()


def test_the_firewall_refuses_held_out_repos():
    """`assert_not_held_out` is the firewall as a callable."""
    S.assert_not_held_out(S.EXPLORATION)          # must not raise
    with pytest.raises(SystemExit, match="SEAL BREACH"):
        S.assert_not_held_out([S.EXPLORATION[0], S.HELD_OUT[0]])


def test_partition_is_a_partition():
    assert len(S.HELD_OUT) == 12
    assert len(S.EXPLORATION) == 18
    assert not set(S.HELD_OUT) & set(S.EXPLORATION)
    assert set(S.HELD_OUT) | set(S.EXPLORATION) == set(S.COHORT)
    assert len(set(S.HELD_OUT)) == len(S.HELD_OUT), "duplicate in held-out"


# --- the coverage guarantee the seal protects -------------------------------

def test_every_held_out_model_is_proxy_covered():
    """A seal precondition, not a Phase 1 detail.

    A model can only be placed in a capability bin if it HAS a proxy number.
    Sealing an uncovered model into held-out would make the bin-diversity
    guarantee unverifiable exactly when Phase 2 needs it — and unfixable, since
    reopening the seal is either breaking it or proxy-shopping the replacement.
    """
    uncovered = [m for m in S.HELD_OUT if S.COHORT[m][3] is None]
    assert not uncovered, f"held-out models without a pinned proxy: {uncovered}"
    assert S.JUSTIFICATION["held_out"]["all_proxy_covered"]


def test_the_nine_uncovered_models_are_all_in_exploration():
    """They cannot be capability-partialled anywhere, so they sit on one side.

    Concentrating them in exploration also removes the temptation to rescue the
    gap with a different proxy later.
    """
    uncovered = {m for m, v in S.COHORT.items() if v[3] is None}
    assert len(uncovered) == 9
    assert uncovered <= set(S.EXPLORATION)


def test_both_sets_carry_family_diversity():
    assert S.JUSTIFICATION["held_out"]["n_families"] >= 4
    assert S.JUSTIFICATION["exploration"]["n_families"] >= 4


def test_held_out_spans_the_capability_window_across_families():
    """Phase 2 cannot confirm capability-comparable structure from one family."""
    f = S.JUSTIFICATION["held_out"]
    assert len(f["window_families"]) >= 2, f["window_families"]
    assert len(f["window_models"]) >= 2


def test_exploration_also_spans_the_window():
    f = S.JUSTIFICATION["exploration"]
    assert len(f["window_families"]) >= 2, f["window_families"]


def test_burned_models_are_exploration_only():
    """Seeing the junk-masking effect again where it was found proves nothing."""
    assert S.JUSTIFICATION["burned_all_in_exploration"]
    assert not (S.BURNED & set(S.HELD_OUT))


def test_no_base_instruct_pair_is_split_across_the_firewall():
    """A pair split across the seal is useless for the base/instruct axis."""
    have = {(f, s, k) for f, s, k, _ in S.COHORT.values()}
    pairs = {(f, s) for f, s, k in have
             if k == "base" and (f, s, "instruct") in have}
    for fam, size in pairs:
        sides = {m in S.HELD_OUT for m, (f, s, _, _) in S.COHORT.items()
                 if (f, s) == (fam, size)}
        assert len(sides) == 1, f"pair {fam}-{size}B is split across the seal"


# --- the limitation the cohort cannot fix, recorded rather than pretended ----

def test_base_location_is_recorded_and_names_the_qwen_concentration():
    """The spec asks for something the cohort cannot provide.

    Every intact pair is Qwen2.5, Qwen3 or Mistral — Falcon3 and OLMo-2 bases
    all failed the loop test, Gemma-2/Granite/Llama-3.1 have none here, and
    Mistral's instruct half is burned. So held-out's base checkpoints are
    necessarily Qwen-only, and the seal records that instead of implying it was
    solved.
    """
    loc = S.JUSTIFICATION["base_location"]
    assert loc["held_out"] == ["Qwen2.5"], loc["held_out"]
    assert "excluded from Phase 2" in loc["note"].replace("EXCLUDED", "excluded")
    assert set(loc["exploration"]) >= {"Mistral"}, \
        "exploration keeps the one non-Qwen pair, which is the only leverage " \
        "this axis has anywhere"


def test_a_non_qwen_held_out_pair_is_genuinely_impossible():
    """Proves the rule is unsatisfiable rather than merely unsatisfied.

    If this ever fails, the cohort gained a usable non-Qwen base checkpoint and
    the base/instruct axis can be reopened for Phase 2.
    """
    have = {(f, s, k) for f, s, k, _ in S.COHORT.values()}
    non_qwen_pairs = {(f, s) for f, s, k in have
                      if k == "base" and (f, s, "instruct") in have
                      and "Qwen" not in f}
    assert non_qwen_pairs == {("Mistral", 7.0)}, non_qwen_pairs
    assert "mistralai/Mistral-7B-Instruct-v0.3" in S.BURNED, \
        "the only non-Qwen pair's instruct half must be burned for this to hold"


def test_window_is_the_widened_one_and_says_so():
    """20-25 left held-out a single matched pair; 18-28 gives it four families.

    The cost is that this is capability-COMPARABLE, not capability-matched.
    """
    assert len(S.JUSTIFICATION["held_out"]["window_families"]) >= 2

    # **Recorded, because it changed the window decision.** Widening to 18-28
    # was justified on a candidate partition in which held-out could hold at
    # most two of the 20-25 bin. Requiring proxy coverage then forced all nine
    # uncovered models into exploration, the covered ones redistributed, and the
    # narrow bin became viable: held-out holds three of it across three
    # families, exploration three across two. Whichever window is sealed, this
    # asserts the narrow one's status so the choice stays visible rather than
    # inherited from a superseded analysis.
    for name, models in (("held_out", S.HELD_OUT), ("exploration", S.EXPLORATION)):
        narrow = [m for m in models
                  if (v := S.COHORT[m][3]) is not None and 20.0 <= v < 25.0]
        fams = {S.COHORT[m][0] for m in narrow}
        assert len(fams) >= 2, (
            f"{name} lost multi-family coverage of the narrow 20-25 bin: {fams}")
