"""Policies — what decides the next command.

**Why this exists.** The calibration study (`docs/addendum-publication-battery.md`
study 2) needs three scripted policies to traverse the **identical** pipeline as
a served model: same worlds, same schedule, same `classify()`, same analysis
path. Anything less and the baselines answer a different question than the
models do, which defeats the point of having baselines.

`C-RAND` in particular is a pipeline audit wearing a baseline's clothes. It
draws uniformly from the *declared* vocabulary and the objects actually present,
so it must score exactly 100.0 action-level. Anything else means `classify()`
has a bug that every model number has been quietly absorbing.

**The constraint on `EndpointPolicy`.** It must reproduce the existing rollout
byte for byte. The measurement loop is shared with a 542-episode corpus and a
published table; if wrapping the endpoint perturbs generation, every future
number is incomparable to every past one.
"""

from __future__ import annotations

import random
from array import array
from typing import Protocol

from .endpoint import Endpoint
from .worldspec import WorldSpec

#: Shared by every n-gram anchor. Generation stops at `</s>` or four tokens,
#: whichever comes first, and an empty draw falls back to `look`. Frozen at R1
#: and reused verbatim by R2/R3 so a rung difference is an anchor-family
#: difference and not a generation-loop difference.
MAX_TOKENS = 4
BOS, EOS = "<s>", "</s>"


class Policy(Protocol):
    """Decides the next raw reply, given the conversation so far.

    Returns the model-shaped *reply* rather than a cleaned command: the caller
    owns normalisation, so a scripted policy and a served model go through the
    same parsing and the same failure modes.
    """

    name: str

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str: ...


class EndpointPolicy:
    """A served model. Byte-identical to the pre-existing inline call."""

    def __init__(self, ep: Endpoint):
        self.ep = ep
        self.name = f"endpoint:{ep.served_name}"

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str:
        # Arguments and seed derivation are verbatim from the original
        # `_rollout` body. Changing any of them invalidates comparison with the
        # existing corpus.
        return self.ep.chat(messages, max_tokens=16, temperature=0.9,
                            seed=seed * 100_003 + step)


# --------------------------------------------------------------------------
# Scripted calibration policies. Frozen — see the pinning test.
# --------------------------------------------------------------------------

class RandomLegalPolicy:
    """C-RAND — uniform over the DECLARED vocabulary × every entity in the world.

    Scores exactly 100.0 action-level by construction. If it does not,
    `classify()` rejects something the rules permit, and every published
    adherence figure is wrong with it.

    **Draws from all world entities rather than only those in the current room.**
    Adherence is about vocabulary, not success: `take dipper` in a room without
    the dipper is a legal command that fails, exactly as it would from a model,
    and it exercises the classifier over the whole entity set rather than
    whichever few happen to be underfoot. Coverage, which does care about what
    executed, is handled separately by `consumed()`.
    """

    def __init__(self, spec: WorldSpec, seed: int = 5150):
        self.spec = spec
        self.name = "C-RAND"
        self._rng = random.Random(seed)
        names = [n for _, group in spec.kinds for n in group]
        self._objects = sorted(names)

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str:
        rng = random.Random(f"{seed}:{step}")
        verb = rng.choice(["go", "look", "inventory", "examine", "take",
                           "drop", "open", "close"])
        if verb == "go":
            return f"go {rng.choice(['north', 'south', 'east', 'west'])}"
        if verb in ("look", "inventory"):
            return verb
        return f"{verb} {rng.choice(self._objects)}"


class NoiseWordPolicy:
    """C-NOISE — random tokens from a frozen wordlist, length-matched.

    The true floor. Nothing here is a legal command except by accident.
    """

    #: Frozen, committed, and pinned. Ordinary English with no overlap with the
    #: action vocabulary, so a hit is coincidence rather than construction.
    WORDLIST = ("harbour", "candle", "thistle", "marble", "orchard", "ribbon",
                "cinder", "meadow", "pewter", "lantern", "furrow", "bramble",
                "tallow", "quarry", "sable", "willow", "gable", "hearth",
                "pallet", "sedge", "trellis", "vellum", "wicket", "yarrow")

    def __init__(self, seed: int = 5150):
        self.name = "C-NOISE"
        self._seed = seed

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str:
        rng = random.Random(f"{self._seed}:{seed}:{step}")
        n = rng.choice([1, 1, 2])          # matches observed command lengths
        return " ".join(rng.choice(self.WORDLIST) for _ in range(n))


class BigramPolicy:
    """C-MIMIC — a bigram over real commands, with no instruction understanding.

    The informative baseline: where it lands relative to the worst real model
    says whether the leaderboard's spread is adherence or surface statistics.

    **Fitted on P1 cells only**, never on a phrasing-pooled corpus. Pooling
    would make the bar's height depend on the phrasing mix — P5-heavy data
    carries more violations, which weakens the bigram and flatters every real
    model against the gate.
    """

    def __init__(self, commands: list[str], seed: int = 5150):
        self.name = "C-MIMIC"
        self._seed = seed
        self._model: dict[str, list[str]] = {}
        vocab = {t for c in commands for t in c.split()} | {"<s>", "</s>"}
        pairs: dict[str, list[str]] = {}
        for c in commands:
            toks = ["<s>"] + c.split() + ["</s>"]
            for a, b in zip(toks, toks[1:]):
                pairs.setdefault(a, []).append(b)
        # Add-one smoothing over the observed vocabulary, per the frozen spec.
        for k in vocab:
            self._model[k] = pairs.get(k, []) + sorted(vocab)
        self.n_fit = len(commands)

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str:
        rng = random.Random(f"{self._seed}:{seed}:{step}")
        out, cur = [], "<s>"
        for _ in range(4):
            cur = rng.choice(self._model.get(cur) or ["</s>"])
            if cur == "</s>":
                break
            out.append(cur)
        return " ".join(out) or "look"


# --------------------------------------------------------------------------
# R2 and R3 — the higher-order anchors from the survey. See the burn-ledger
# append of 2026-08-09.
#
# **Why these cannot reuse R1's trick.** `BigramPolicy` gets add-one smoothing
# for free by appending `sorted(vocab)` to every context's successor list and
# sampling with `rng.choice`: uniform choice over a multiset is already
# count-proportional, and one extra copy of each vocabulary item is exactly
# add-one. That works only because every term in the mixture has weight 1.
# Backoff (0.4) and interpolation (0.5/0.3/0.2) need genuinely unequal weights,
# so both build explicit weight vectors and sample with `rng.choices`.
#
# Consequence worth stating: R2/R3 consume the RNG differently from R1
# (`choices` draws one float, `choice` draws bits), so a degeneracy check
# compares *distributions*, never generated strings. The distribution is what
# the parameters control; the stream is an artefact of the sampler.
# --------------------------------------------------------------------------

def _ngram_counts(commands: list[str], order: int) -> dict[tuple, dict[str, int]]:
    """context -> {successor: count}, each order padded to its own arity.

    The bigram tier pads with one `<s>` exactly as `BigramPolicy` does, which is
    what makes R3-collapsed-onto-bigram reproduce R1 rather than merely resemble
    it. Higher tiers pad with `order - 1`, so their contexts are well defined
    from the first generated token onward.
    """
    counts: dict[tuple, dict[str, int]] = {}
    for c in commands:
        toks = [BOS] * (order - 1) + c.split() + [EOS]
        for i in range(order - 1, len(toks)):
            d = counts.setdefault(tuple(toks[i - order + 1:i]), {})
            d[toks[i]] = d.get(toks[i], 0) + 1
    return counts


def _vocab(commands: list[str]) -> tuple[str, ...]:
    """Sorted, and including both sentinels — identical to `BigramPolicy`.

    `<s>` is a drawable successor there, an oddity of the frozen R1 that has to
    be reproduced rather than tidied: add-one over a different vocabulary is a
    different distribution, and R1's numbers are already in the ledger.
    """
    return tuple(sorted({t for c in commands for t in c.split()} | {BOS, EOS}))


class _NgramPolicy:
    """Shared generation loop for the weighted anchors.

    Cumulative weight vectors are cached per context, which turns an O(|V|)
    rebuild per token into a bisect. The cache is **bounded and cleared
    wholesale** when it fills: add-one smoothing puts real mass on the tail
    (~40% for a typical context at |V| ~ 350), so contexts genuinely are diverse
    and an unbounded cache would grow with the run rather than with the model.
    Clearing cannot perturb a result — a weight vector is a pure function of its
    context, so the cache is speed only.
    """

    #: ~350 doubles per entry, so this is tens of MB rather than gigabytes.
    CACHE_MAX = 20_000

    def __init__(self, commands: list[str], seed: int):
        self._seed = seed
        self.vocab = _vocab(commands)
        self.n_fit = len(commands)
        self._cum: dict[tuple, array] = {}

    def scores(self, history: tuple[str, ...]) -> list[float]:
        """Unnormalised score over `self.vocab`, in vocab order. Subclass hook."""
        raise NotImplementedError

    def distribution(self, history: tuple[str, ...]) -> dict[str, float]:
        """Normalised, for tests and reporting. Not used in generation.

        Degeneracy checks compare this, never generated strings: `choices` and
        `choice` consume the RNG differently, so two policies can agree exactly
        on the distribution and still emit different text from the same seed.
        """
        s = self.scores(history)
        total = sum(s)
        return {t: x / total for t, x in zip(self.vocab, s)}

    def _draw(self, rng: random.Random, history: tuple[str, ...]) -> str:
        cum = self._cum.get(history)
        if cum is None:
            cum, run = array("d"), 0.0
            for x in self.scores(history):
                run += x
                cum.append(run)
            if len(self._cum) >= self.CACHE_MAX:
                self._cum.clear()
            self._cum[history] = cum
        return rng.choices(self.vocab, cum_weights=cum, k=1)[0]

    def reply(self, messages: list[dict], *, step: int, seed: int) -> str:
        rng = random.Random(f"{self._seed}:{seed}:{step}")
        out: list[str] = []
        for _ in range(MAX_TOKENS):
            nxt = self._draw(rng, self._context(out))
            if nxt == EOS:
                break
            out.append(nxt)
        return " ".join(out) or "look"

    def _context(self, out: list[str]) -> tuple[str, ...]:
        raise NotImplementedError


class TrigramBackoffPolicy(_NgramPolicy):
    """R2 — trigram with stupid backoff to the add-one bigram.

    Stupid backoff is per-word, not per-context: a successor observed in the
    trigram context gets its trigram MLE, and every other word gets `alpha`
    times its bigram score. Unnormalised by construction, so sampling normalises
    what Brants et al. left as a score.

    **The bigram tier is R1's distribution exactly**, so where R2 lands relative
    to R1 is attributable to trigram structure and nothing else.
    """

    ALPHA = 0.4

    def __init__(self, commands: list[str], seed: int = 5150, alpha: float | None = None):
        super().__init__(commands, seed)
        self.name = "C-MIMIC-R2"
        self.alpha = self.ALPHA if alpha is None else alpha
        self._tri = _ngram_counts(commands, 3)
        self._bi = _ngram_counts(commands, 2)

    def _bigram_probs(self, ctx: tuple[str, ...]) -> list[float]:
        row = self._bi.get(ctx, {})
        w = [row.get(t, 0) + 1 for t in self.vocab]      # add-one, as in R1
        total = float(sum(w))
        return [x / total for x in w]

    def scores(self, history: tuple[str, ...]) -> list[float]:
        bi = self._bigram_probs(history[-1:])
        tri = self._tri.get(history, {})
        if not tri:
            # Nothing observed at trigram order: the whole distribution backs
            # off, alpha cancels under normalisation, and R2 *is* R1 here.
            return bi
        n = float(sum(tri.values()))
        return [tri[t] / n if t in tri else self.alpha * p
                for t, p in zip(self.vocab, bi)]

    def _context(self, out: list[str]) -> tuple[str, ...]:
        return tuple(([BOS, BOS] + out)[-2:])


class InterpolatedNgramPolicy(_NgramPolicy):
    """R3 — linear interpolation of add-one 4-, 3- and 2-gram distributions.

    Every tier is a proper distribution before mixing, so the weights mean what
    they say. Collapsing them onto the bigram term must reproduce R1 exactly,
    which is the witness that the mixing code is right.
    """

    WEIGHTS = (0.5, 0.3, 0.2)                            # 4-gram, 3-gram, 2-gram
    ORDERS = (4, 3, 2)

    def __init__(self, commands: list[str], seed: int = 5150,
                 lambdas: tuple[float, ...] | None = None):
        super().__init__(commands, seed)
        self.name = "C-MIMIC-R3"
        self.lambdas = self.WEIGHTS if lambdas is None else lambdas
        if len(self.lambdas) != len(self.ORDERS):
            raise ValueError(f"need {len(self.ORDERS)} weights, one per order")
        self._counts = {o: _ngram_counts(commands, o) for o in self.ORDERS}

    def _tier(self, order: int, ctx: tuple[str, ...]) -> list[float]:
        row = self._counts[order].get(ctx, {})
        w = [row.get(t, 0) + 1 for t in self.vocab]      # add-one, as in R1
        total = float(sum(w))
        return [x / total for x in w]

    def scores(self, history: tuple[str, ...]) -> list[float]:
        tiers = [self._tier(o, history[-(o - 1):]) for o in self.ORDERS]
        return [sum(lam * t[i] for lam, t in zip(self.lambdas, tiers))
                for i in range(len(self.vocab))]

    def _context(self, out: list[str]) -> tuple[str, ...]:
        return tuple(([BOS, BOS, BOS] + out)[-3:])
