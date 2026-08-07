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

Length bias: options of different token length get different raw logprob sums.
That bias is constant across checkpoints, so it cancels in any before/after or
across-seed comparison. It does NOT cancel when comparing one probe to another,
so per-probe distributions are never compared to each other, only to the same
probe under a different checkpoint.
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
    """Summed token logprobs of each option's continuation. Unnormalised."""

    probs: tuple[float, ...]
    """Softmax over `logprobs`. This is the per-probe distribution."""

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
            "logprobs": [round(x, 4) for x in self.logprobs],
            "probs": [round(p, 6) for p in self.probs],
            "modal": self.modal,
            "entropy": round(self.entropy, 4),
        }


def softmax(values: Sequence[float]) -> tuple[float, ...]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return tuple(e / total for e in exps)


def score_options(
    model, tokenizer, prompt: str, options: Sequence[str]
) -> OptionScores:
    """Return the model's exact distribution over `options` given `prompt`.

    One forward pass per option. Deterministic: no sampler is involved, so
    repeated calls with identical inputs return identical numbers.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prompt_ids = tokenizer.encode(prompt)
    totals: list[float] = []

    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)
        if not option_ids:
            totals.append(float("-inf"))
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
        mx.clear_cache()

    return OptionScores(
        options=tuple(options),
        logprobs=tuple(totals),
        probs=softmax(totals),
    )
