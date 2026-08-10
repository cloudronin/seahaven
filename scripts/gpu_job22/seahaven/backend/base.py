"""Backend protocols.

`Generator` and `Trainer` are separate on purpose. On rented CUDA you generate
with vLLM in one process and train with peft in another — possibly on another
box, possibly hours later. Fusing them into one object bakes in a colocation
assumption that is true only of a toy setup.

What is deliberately NOT abstracted here, and why:

  - **Batching.** The interface is one request at a time. Offering a
    `generate_batch()` that silently loops on a backend which cannot batch
    creates a performance lie and invites code that only works on one backend.
  - **Prompt content.** Templates are experiment identity and must be
    byte-identical across backends. A backend that "helpfully" prepends a system
    preamble is a confound, not a convenience.
  - **Adapter scheduling.** Whether switching adapters is free or forces a
    reload is a scheduling fact the sweep planner needs *globally* to plan a
    long run. It is exposed declaratively as `caps.adapter_switch_cost` rather
    than hidden behind `load_adapter()`.
  - **Cross-backend determinism.** Two backends will not emit identical tokens
    for the same seed. Papering over that would make a cross-backend comparison
    look valid when it is not. Any replication on different hardware is a fresh
    set of runs with its own null distribution.
"""

from __future__ import annotations

import abc
import contextlib
from typing import Iterator

from .types import AdapterRef, BackendCaps, GenRequest, GenResult, Message, TrainSpec

__all__ = ["CacheScope", "Generator", "Trainer"]


class CacheScope(abc.ABC):
    """The lifetime of one stable prefix.

    Every generation inside a scope shares the prefix's prefill. Opened once per
    campaign for the step loop, and once per measurement point for the probe
    battery — where ~40 probes amortise a single cold prefill between them, which
    is most of the reason the battery is affordable at all.
    """

    @abc.abstractmethod
    def generate(self, req: GenRequest) -> GenResult: ...

    @property
    @abc.abstractmethod
    def prefix_tokens(self) -> int: ...


class Generator(abc.ABC):
    caps: BackendCaps

    @abc.abstractmethod
    def fingerprint(self) -> str:
        """Stamped on every GenResult.

        Covers backend name and version, model repo and revision, quantisation,
        active adapter sha256, and sampler parameters — everything that could
        change an output without changing the prompt.
        """

    @abc.abstractmethod
    def load_adapter(self, adapter: AdapterRef | None) -> None:
        """May be expensive; see `caps.adapter_switch_cost`.

        Minimising calls is the scheduler's job, not the caller's.
        """

    @abc.abstractmethod
    @contextlib.contextmanager
    def cache_scope(self, frozen: tuple[Message, ...]) -> Iterator[CacheScope]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train_adapter(self, spec: TrainSpec) -> AdapterRef:
        """Train one adapter on one run's own trajectories.

        Implementations MUST verify that every line of `spec.train_jsonl` carries
        `run_id == spec.run_id` before doing any work. No cross-seed pooling is
        the load-bearing isolation property of the whole experiment, and this is
        the cheapest place to enforce it.
        """
