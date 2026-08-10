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
from typing import Protocol

from .endpoint import Endpoint
from .worldspec import WorldSpec


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
