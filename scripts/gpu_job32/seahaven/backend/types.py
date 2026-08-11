"""Data types crossing the model-backend boundary.

The seam is "one complete constrained completion, given a declared-stable prefix
and a variable suffix." Not tokens, not logits, not a cache object.

`PromptParts` is why. The obvious signature is `generate(prompt: str, ...)`, and
it is wrong here: a backend handed a flat string cannot know which part of it is
stable across calls, and the entire cost model of this experiment rests on prefix
cache hits. Putting the boundary in the contract lets each backend exploit it
differently — vLLM's automatic prefix caching keys on it, a stub ignores it —
without any caller ever touching a cache object.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Any, Literal, NewType

__all__ = [
    "AgentText",
    "Message",
    "PromptParts",
    "Constraint",
    "NoConstraint",
    "JsonSchema",
    "Regex",
    "Choice",
    "GenRequest",
    "GenResult",
    "AdapterRef",
    "TrainSpec",
    "BackendCaps",
]

AgentText = NewType("AgentText", str)
"""Text that is safe to show the agent.

Minted only by `world.scrub` and `prompt.blocks`. Prompt assembly accepts nothing
else, which is what keeps `HiddenState.score` from reaching a prompt by accident.
"""

Role = Literal["system", "user", "assistant"]


@dc.dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dc.dataclass(frozen=True)
class PromptParts:
    """A prompt split at the cache boundary.

    `frozen` MUST be byte-identical across every call within a cache scope, or
    append-only relative to it. Violating that costs a cold prefill on every step
    and shows up only as a schedule that quietly doubles — which is why
    `prompt.assemble` asserts the invariant rather than trusting it.

    The classic way to break it is a relative timestamp ("14 steps ago") inside a
    frozen block: correct-looking, and it invalidates the prefix every step.
    """

    frozen: tuple[Message, ...]
    variable: tuple[Message, ...]

    def all_messages(self) -> tuple[Message, ...]:
        return self.frozen + self.variable


# --- constraints ---------------------------------------------------------
# Backend-neutral. vLLM maps these onto its structured-outputs backend; the stub
# uses them to validate its scripted replies.


@dc.dataclass(frozen=True)
class NoConstraint:
    pass


@dc.dataclass(frozen=True)
class JsonSchema:
    schema: dict[str, Any]
    name: str


@dc.dataclass(frozen=True)
class Regex:
    pattern: str


@dc.dataclass(frozen=True)
class Choice:
    """Forced choice over a fixed option set.

    The cheapest constraint there is — a trivial automaton — and the right one
    for probes. It also enables exact option scoring: read the probability mass
    over the options from logprobs in one forward pass instead of sampling K
    times, which removes sampling noise from the measurement entirely.
    """

    options: tuple[str, ...]


Constraint = NoConstraint | JsonSchema | Regex | Choice


@dc.dataclass(frozen=True)
class GenRequest:
    parts: PromptParts
    constraint: Constraint
    sampling_seed: int
    """Per-call, never per-process. This is what makes a step replayable."""

    max_tokens: int
    temperature: float = 0.8
    top_p: float = 0.95
    stop: tuple[str, ...] = ()


@dc.dataclass(frozen=True)
class GenResult:
    text: str
    parse_ok: bool
    value: Any | None
    """The parsed object when the constraint was a JsonSchema."""

    prompt_tokens: int
    prefill_tokens: int
    """Tokens actually prefilled, i.e. the cache miss. Drives the cost model and
    doubles as the empirical check that the prefix design works."""

    completion_tokens: int
    """Charged against the deliberation budget."""

    cache_hit: bool
    finish_reason: Literal["stop", "length", "constraint_dead_end", "error"]
    wall_s: float
    backend_fingerprint: str

    def digest(self) -> dict[str, Any]:
        """Compact form for the step record."""
        return {
            "parse_ok": self.parse_ok,
            "prompt_tokens": self.prompt_tokens,
            "prefill_tokens": self.prefill_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit": self.cache_hit,
            "finish_reason": self.finish_reason,
            "wall_s": round(self.wall_s, 4),
            "fp": self.backend_fingerprint,
        }


@dc.dataclass(frozen=True)
class AdapterRef:
    name: str
    """Serving name. MUST be unique per (run, campaign) and never reused.

    vLLM keys its KV prefix cache on the adapter *name* and does not invalidate
    it when an adapter is reloaded (vllm#42125). Reusing a name after retraining
    makes later requests reuse KV blocks computed from the old weights — silently,
    and biased toward making campaign N behave like campaign N-1. That fabricates
    within-run stability, which is exactly the signal K2 tests for.

    Hence `run07_c1`, `run07_c2`, never `run07`.
    """

    path: str
    sha256: str
    base_model: str
    trained_from: str | None = None
    """sha256 of the adapter this one resumed from, forming the campaign chain."""


@dc.dataclass(frozen=True)
class TrainSpec:
    base_model: str
    resume_from: AdapterRef | None
    train_jsonl: str
    valid_jsonl: str | None
    out_dir: str
    adapter_name: str
    rank: int
    alpha: float
    layers: int
    iters: int
    batch_size: int
    learning_rate: float
    seed: int
    run_id: str
    """Asserted against every line of `train_jsonl` before training starts.

    Cheap, and it catches the single most catastrophic bug available here: one
    run's trajectories leaking into another run's adapter, which would destroy
    the isolation the divergence claim depends on.
    """


@dc.dataclass(frozen=True)
class BackendCaps:
    supports: frozenset[str]
    supports_batch: bool
    adapter_switch_cost: Literal["free", "reload"]
    deterministic_across_cache_states: bool
    """Measured, not assumed. See scripts/bench_determinism.py."""

    max_context: int
