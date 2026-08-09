"""A Generator that emits scripted output. The workhorse of the test suite.

Everything expensive and stochastic about this system lives behind the backend
seam, so faking the seam makes the rest of the harness testable in under a
second: the step loop, record commits, resume-after-kill, containment scanning,
budget accounting, and probe no-trace all exercise fully without a model.

Two behaviours it supports deliberately:

  - **Scripted replies**, keyed by call index, so "identical trajectory" is a
    meaningful assertion in the resume tests.
  - **Injectable malformed output**, so the parse-failure and repair paths are
    exercised on purpose rather than only when a real model happens to misbehave.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Callable, Iterator, Sequence

from .base import CacheScope, Generator
from .types import (
    AdapterRef,
    BackendCaps,
    Choice,
    GenRequest,
    GenResult,
    JsonSchema,
    Message,
)

__all__ = ["StubBackend", "StubScope", "wander_script"]

CAPS = BackendCaps(
    supports=frozenset({"json_schema", "regex", "choice"}),
    supports_batch=False,
    adapter_switch_cost="free",
    deterministic_across_cache_states=True,
    max_context=1_000_000,
)


class StubScope(CacheScope):
    def __init__(self, backend: "StubBackend", frozen: tuple[Message, ...]) -> None:
        self._backend = backend
        self._frozen = frozen
        self._prefix_tokens = sum(len(m.content) // 4 for m in frozen)
        self._calls_in_scope = 0

    @property
    def prefix_tokens(self) -> int:
        return self._prefix_tokens

    def generate(self, req: GenRequest) -> GenResult:
        started = time.perf_counter()
        text = self._backend._next_text(req)
        self._calls_in_scope += 1

        # First call in a scope pays the prefill; later ones hit the cache. This
        # mirrors real backend behaviour closely enough that cost-model code and
        # cache-hit assertions can be tested without a GPU.
        cache_hit = self._calls_in_scope > 1
        prefill = 0 if cache_hit else self._prefix_tokens

        value, parse_ok = None, True
        if isinstance(req.constraint, JsonSchema):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                parse_ok = False
        elif isinstance(req.constraint, Choice):
            parse_ok = text in req.constraint.options

        variable_tokens = sum(len(m.content) // 4 for m in req.parts.variable)
        return GenResult(
            text=text,
            parse_ok=parse_ok,
            value=value,
            prompt_tokens=self._prefix_tokens + variable_tokens,
            prefill_tokens=prefill,
            completion_tokens=max(1, len(text) // 4),
            cache_hit=cache_hit,
            finish_reason="stop",
            wall_s=time.perf_counter() - started,
            backend_fingerprint=self._backend.fingerprint(),
        )


class StubBackend(Generator):
    """Replays `script` in order; falls back to `default` once exhausted."""

    caps = CAPS

    def __init__(
        self,
        script: Sequence[str] | None = None,
        *,
        default: Callable[[GenRequest], str] | None = None,
        adapter: AdapterRef | None = None,
    ) -> None:
        self._script = list(script or [])
        self._default = default
        self._adapter = adapter
        self.calls = 0
        self.adapter_loads = 0

    def fingerprint(self) -> str:
        name = self._adapter.name if self._adapter else "none"
        return f"stub/1|adapter={name}"

    def load_adapter(self, adapter: AdapterRef | None) -> None:
        self._adapter = adapter
        self.adapter_loads += 1

    @contextlib.contextmanager
    def cache_scope(self, frozen: tuple[Message, ...]) -> Iterator[CacheScope]:
        yield StubScope(self, frozen)

    def close(self) -> None:
        pass

    def _next_text(self, req: GenRequest) -> str:
        index = self.calls
        self.calls += 1
        if index < len(self._script):
            return self._script[index]
        if self._default is not None:
            return self._default(req)
        if isinstance(req.constraint, Choice):
            return req.constraint.options[0]
        return json.dumps({"expect": "nothing in particular", "command": "look"})


def wander_script(seed: int = 0) -> Callable[[GenRequest], str]:
    """A deterministic pseudo-agent that moves and pokes at things.

    Not a good policy — it is not supposed to be. It only needs to produce an
    action distribution with the right *shape*: mostly movement and examination,
    occasional taking, so that trajectory-derived training data resembles what a
    real run would produce.
    """
    import random

    verbs = [
        "go north", "go south", "go east", "go west",
        "look", "inventory", "examine crate", "examine kettle",
        "take kettle", "take coil of rope", "open crate", "open locker",
    ]
    rng = random.Random(seed)

    def choose(_req: GenRequest) -> str:
        return json.dumps(
            {
                "expect": rng.choice(
                    ["the room will be as before", "something will give",
                     "nothing", "it will be where I left it"]
                ),
                "command": rng.choice(verbs),
            }
        )

    return choose
