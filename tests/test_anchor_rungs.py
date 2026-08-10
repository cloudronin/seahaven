"""R2 and R3 — the survey's higher-order anchors.

**What can go wrong here is silent.** R1 gets add-one smoothing for free by
appending `sorted(vocab)` to each context's successor list and sampling with
`rng.choice`; uniform choice over a multiset is already count-proportional.
Backoff and interpolation need unequal weights, so R2/R3 build explicit vectors
— and a wrong weight vector still produces plausible commands, still runs, and
still yields an anchor number. It just yields the wrong one, and nothing
downstream would notice.

So each rung gets a **degeneracy witness**: a parameter setting under which it
must reduce to R1 exactly. Comparisons are between *distributions*, never
generated strings — `choices` draws a float where `choice` draws bits, so two
policies can agree exactly on the distribution and still emit different text
from the same seed. The distribution is what the parameters control.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from seahaven.fidelity.policy import (BigramPolicy, InterpolatedNgramPolicy,
                                      TrigramBackoffPolicy)

#: Small but structurally realistic, and deliberately containing higher-order
#: structure: `brass` continues to `key` or `chest` depending on the verb before
#: it. A corpus of one- and two-token commands would not do — with nothing but
#: sentinel padding above the bigram, all three tiers see identical
#: context/successor pairs and every rung collapses onto R1 for reasons that
#: have nothing to do with whether the mixing code is correct.
CORPUS = [
    "look", "inventory", "look",
    "go north", "go south", "go north", "go east", "go west",
    "take brass key", "take brass key",
    "open brass chest", "open brass chest",
    "close brass chest",
    "drop brass key",
    "examine wooden chest",
    "take wooden plank",
]


def r1_distribution(pol: BigramPolicy, context: str) -> dict[str, float]:
    """R1's implied distribution, read out of its multiset without touching it.

    `BigramPolicy` is frozen and already in the burn ledger, so the witness
    reconstructs its distribution rather than adding a method to it.
    """
    bag = pol._model[context]
    n = float(len(bag))
    return {t: c / n for t, c in Counter(bag).items()}


def assert_same_distribution(a: dict[str, float], b: dict[str, float]) -> None:
    keys = {k for k, v in a.items() if v} | {k for k, v in b.items() if v}
    for k in sorted(keys):
        assert a.get(k, 0.0) == pytest.approx(b.get(k, 0.0), abs=1e-12), \
            f"distributions differ on {k!r}: {a.get(k, 0.0)} vs {b.get(k, 0.0)}"


# --- witness 1: R3 collapsed onto its bigram tier is R1 ----------------------

def test_r3_collapsed_to_the_bigram_tier_reproduces_r1():
    """Weights (0, 0, 1) leave only the add-one bigram, which is R1.

    This is the whole mixing path under test: tier construction, add-one, the
    per-order padding, and the weighted sum. If any of them is wrong, the
    collapsed case stops matching.
    """
    r1 = BigramPolicy(CORPUS, seed=5150)
    r3 = InterpolatedNgramPolicy(CORPUS, seed=5150, lambdas=(0.0, 0.0, 1.0))

    for ctx in ("<s>", "go", "take", "brass", "key"):
        assert_same_distribution(r3.distribution((ctx,) * 3), r1_distribution(r1, ctx))


def test_r3_weights_are_a_genuine_mixture():
    """The collapsed case must not pass by accident of the tiers agreeing.

    Probed at `brass`, the one context where the corpus has structure above the
    bigram: `take brass` continues to `key`, `open brass` to `chest`, and the
    bigram alone cannot tell them apart.
    """
    r1 = BigramPolicy(CORPUS, seed=5150)
    r3 = InterpolatedNgramPolicy(CORPUS, seed=5150)          # 0.5 / 0.3 / 0.2

    d = r3.distribution(("<s>", "take", "brass"))
    b = r1_distribution(r1, "brass")
    assert d != pytest.approx(b), \
        "the default weights should differ from the pure bigram, or the " \
        "higher orders are contributing nothing and R3 is R1 in disguise"
    assert sum(d.values()) == pytest.approx(1.0)

    # The disambiguation claim is about the RATIO, not the level: `brass` is
    # followed by `key` and `chest` equally often overall, and only the verb
    # before it breaks the tie.
    assert b["key"] == pytest.approx(b["chest"]), "the bigram cannot tell them apart"
    assert d["key"] > 2 * d["chest"], "but the higher orders can"


def test_add_one_at_high_order_dilutes_rather_than_sharpens():
    """A property of the specced R3, recorded because it shapes the survey.

    Add-one costs a tier `|V|` pseudo-counts regardless of how much evidence it
    has. A 4-gram context with two observations is therefore diluted far harder
    than a bigram context with six, so interpolating toward higher orders can
    move probability mass *down* on the very continuation the higher orders
    agree on — as it does here.

    The consequence for the survey is the part worth stating: the ladder assumed
    higher order means stronger imitator means higher anchor. Under add-one with
    a real vocabulary (|V| ~ 300 on the fit corpus) against sparse 4-gram
    contexts, that assumption is not safe, and **R3 may land below R1**. R2 is
    the opposite case — stupid backoff gives an observed continuation its raw
    MLE with no pseudo-counts, so it concentrates instead of diluting.
    """
    r1 = BigramPolicy(CORPUS, seed=5150)
    r3 = InterpolatedNgramPolicy(CORPUS, seed=5150)
    r2 = TrigramBackoffPolicy(CORPUS, seed=5150)

    bigram_key = r1_distribution(r1, "brass")["key"]
    assert r3.distribution(("<s>", "take", "brass"))["key"] < bigram_key
    assert r2.distribution(("take", "brass"))["key"] > bigram_key


def test_r3_rejects_a_wrong_length_weight_vector():
    with pytest.raises(ValueError, match="one per order"):
        InterpolatedNgramPolicy(CORPUS, lambdas=(0.5, 0.5))


# --- witness 2: R2 with nothing observed at trigram order is R1 --------------

def test_r2_on_an_unobserved_trigram_context_reproduces_r1():
    """Where backoff has nothing to prefer, alpha cancels and R2 *is* R1.

    Note the framing: the degenerate case is an **unseen** trigram context, not
    an unrepeated one. A context observed exactly once is not neutral — its
    single successor takes the whole trigram MLE at weight 1.0, which is the
    opposite of neutral. So the witness probes a context the corpus never
    contains, where the entire distribution backs off and the 0.4 divides out
    under normalisation.
    """
    r1 = BigramPolicy(CORPUS, seed=5150)
    r2 = TrigramBackoffPolicy(CORPUS, seed=5150)

    # ("plank", "brass") never occurs; the bigram context "brass" is populated.
    assert ("plank", "brass") not in r2._tri
    assert_same_distribution(r2.distribution(("plank", "brass")),
                             r1_distribution(r1, "brass"))


def test_r2_alpha_cancels_regardless_of_its_value():
    """Confirms the previous test is about backoff, not about 0.4 specifically."""
    base = TrigramBackoffPolicy(CORPUS, seed=5150).distribution(("plank", "brass"))
    for alpha in (0.1, 0.9):
        other = TrigramBackoffPolicy(CORPUS, seed=5150, alpha=alpha)
        assert_same_distribution(other.distribution(("plank", "brass")), base)


def test_r2_prefers_an_observed_trigram_continuation():
    """The non-degenerate direction: seen trigram context must shift mass.

    Without this, a policy that always backed off would pass every witness
    above and be a bigram wearing a trigram's name.
    """
    r1 = BigramPolicy(CORPUS, seed=5150)
    r2 = TrigramBackoffPolicy(CORPUS, seed=5150)

    seen = ("take", "brass")                   # -> "key" twice, never "chest"
    assert seen in r2._tri
    d, b = r2.distribution(seen), r1_distribution(r1, "brass")
    assert d["key"] > b["key"], \
        "trigram structure should raise the observed continuation above bigram"
    assert d["chest"] < b["chest"], \
        "and should suppress the continuation this verb never takes"


# --- shape invariants shared with R1 ----------------------------------------

@pytest.mark.parametrize("factory", [TrigramBackoffPolicy, InterpolatedNgramPolicy])
def test_rungs_share_r1s_generation_shape(factory):
    """Same 4-token cap, same fallback, same commands-shaped output.

    A rung that differs anywhere but the n-gram order is measuring a pipeline
    change rather than an anchor family.
    """
    pol = factory(CORPUS, seed=5150)
    for step in range(60):
        out = pol.reply([], step=step, seed=7)
        assert out and out == out.strip()
        assert len(out.split()) <= 4


@pytest.mark.parametrize("factory", [TrigramBackoffPolicy, InterpolatedNgramPolicy])
def test_rungs_are_deterministic_given_the_seed(factory):
    a = factory(CORPUS, seed=5150)
    b = factory(CORPUS, seed=5150)
    assert [a.reply([], step=s, seed=3) for s in range(40)] == \
           [b.reply([], step=s, seed=3) for s in range(40)]


@pytest.mark.parametrize("factory", [TrigramBackoffPolicy, InterpolatedNgramPolicy])
def test_cache_eviction_cannot_change_a_draw(factory):
    """The cumulative-weight cache is bounded, so it must be provably inert."""
    pol = factory(CORPUS, seed=5150)
    first = [pol.reply([], step=s, seed=11) for s in range(30)]
    pol._cum.clear()
    assert [pol.reply([], step=s, seed=11) for s in range(30)] == first


def test_r2_and_r3_disagree_with_each_other():
    """Three rungs that all collapse to R1 would make the survey vacuous."""
    r2 = TrigramBackoffPolicy(CORPUS, seed=5150)
    r3 = InterpolatedNgramPolicy(CORPUS, seed=5150)
    ctx2, ctx3 = ("take", "brass"), ("<s>", "take", "brass")
    assert r2.distribution(ctx2) != pytest.approx(r3.distribution(ctx3))
