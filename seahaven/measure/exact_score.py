"""Exact option scoring for forced-choice probes.

The default way to measure a probe is to sample it K times and count. That makes
every measurement a multinomial draw, and the resulting plug-in divergence
estimator is biased upward at small K — worse, the bias is *larger* for the
within-run comparison than for the across-seed one, so it does not cancel in the
ratio the kill criteria depend on. Correcting it needs either an unbiased
U-statistic or a large K.

For a forced choice there is a better option: read the distribution directly.
Score each option's continuation under the prompt, softmax over the options, and
you have the model's exact preference in one forward pass per option. No
sampling, no K, no bias. Test-retest at an unchanged checkpoint is then exactly
zero rather than merely small, which turns the instrument-error check from a
statistical estimate into an assertion.

This is the "Tier-A" path in the plan. Tier-B — probes needing a real multi-step
action in the world — still has to sample, because there is no closed form for
"what would it have done over five turns."

Length bias, and why it is normalised by default. A summed logprob is
monotonically more negative the more tokens an option has, so a longer phrasing
is penalised for being longer rather than for being less preferred. The bias is
constant across checkpoints and therefore cancels in any before/after or
across-seed comparison — but it does real damage before it cancels: it drives
slots toward degeneracy. Measured on a 0.6B checkpoint, "climb the ladder" vs
"stay where you are" scored 0.994/0.006 with entropy 0.04, mostly on length. A
slot pinned at 0.99 leaves almost no room for a disposition to show up, so it
contributes nothing to a distance no matter how much character exists.

`length_normalise=True` (the default) divides each option's summed logprob by its
token count, giving mean logprob per token. That measures "which does it prefer"
rather than "which is shorter." Raw sums remain available on the result for
diagnosis, and the choice is recorded in the output so it cannot drift silently
between measurements.
"""

from __future__ import annotations

import dataclasses as dc
import math
from typing import Sequence

__all__ = ["OptionScores", "score_options", "softmax"]


@dc.dataclass(frozen=True)
class OptionScores:
    options: tuple[str, ...]
    logprobs: tuple[float, ...]
    """Raw summed token logprobs. Kept for diagnosis; length-biased."""

    token_counts: tuple[int, ...]
    length_normalised: bool

    probs: tuple[float, ...]
    """Softmax over the scores actually used. The per-probe distribution."""

    @property
    def degenerate(self) -> bool:
        """No variance available, so this slot contributes nothing to a distance.

        The culling protocol drops these: a slot the model answers the same way
        every time measures competence, and competence converges.
        """
        return self.max_prob > 0.95

    @property
    def modal(self) -> str:
        return self.options[max(range(len(self.probs)), key=self.probs.__getitem__)]

    @property
    def max_prob(self) -> float:
        return max(self.probs)

    @property
    def entropy(self) -> float:
        """Natural-log entropy. A probe with near-zero entropy has no variance
        available to contribute to a distance and is a culling candidate."""
        return -sum(p * math.log(p) for p in self.probs if p > 0)

    def as_dict(self) -> dict:
        return {
            "options": list(self.options),
            "logprobs_raw": [round(x, 4) for x in self.logprobs],
            "token_counts": list(self.token_counts),
            "length_normalised": self.length_normalised,
            "probs": [round(p, 6) for p in self.probs],
            "modal": self.modal,
            "entropy": round(self.entropy, 4),
            "degenerate": self.degenerate,
        }


def softmax(values: Sequence[float]) -> tuple[float, ...]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return tuple(e / total for e in exps)


def score_options(
    model,
    tokenizer,
    prompt: str,
    options: Sequence[str],
    *,
    length_normalise: bool = True,
) -> OptionScores:
    """Return the model's exact distribution over `options` given `prompt`.

    One forward pass per option. Deterministic: no sampler is involved, so
    repeated calls with identical inputs return identical numbers. That is what
    makes test-retest at an unchanged checkpoint an assertion (== 0) rather than
    an estimate.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prompt_ids = tokenizer.encode(prompt)
    totals: list[float] = []
    counts: list[int] = []

    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)
        if not option_ids:
            totals.append(float("-inf"))
            counts.append(0)
            continue

        ids = mx.array([prompt_ids + option_ids])
        logits = model(ids[:, :-1])
        logprobs = nn.log_softmax(logits.astype(mx.float32), axis=-1)

        # Sum the logprob assigned to each option token by the position before it.
        start = len(prompt_ids) - 1
        total = 0.0
        for offset, token in enumerate(option_ids):
            total += float(logprobs[0, start + offset, token])
        totals.append(total)
        counts.append(len(option_ids))
        mx.clear_cache()

    if length_normalise:
        scored = [
            t / c if c else float("-inf") for t, c in zip(totals, counts)
        ]
    else:
        scored = totals

    return OptionScores(
        options=tuple(options),
        logprobs=tuple(totals),
        token_counts=tuple(counts),
        length_normalised=length_normalise,
        probs=softmax(scored),
    )
