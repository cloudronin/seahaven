"""MLX Generator/Trainer for local spikes.

This is the *development* backend. Real runs go to vLLM on CUDA; this exists so
the Phase A gates can be answered without provisioning a GPU.

What transfers to CUDA and what does not:

  - **Transfers:** whether a base checkpoint can hold a parseable action loop
    (A4), and whether a LoRA pass on realistically-shaped self-generated data
    moves behavior at all (A1b). Those are questions about the model and the
    data, not about the serving stack.
  - **Does NOT transfer:** exact token sequences, throughput, and any absolute
    distance value. MLX and vLLM will not agree bitwise for the same seed, so a
    CUDA replication is a fresh set of runs with its own null distribution, never
    a continuation of these.

Constraint handling is deliberately minimal. `mlx_lm` has no grammar backend, and
wiring `outlines` in would add a dependency whose behaviour differs from vLLM's
xgrammar anyway. Since one of the questions being asked is precisely "how often
does a base model fail to produce parseable output," constraining the output here
would destroy the measurement. Constraints are therefore advisory: rendered into
the prompt, enforced by the parser, and the failure rate is the result.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Iterator

from .base import CacheScope, Generator
from .types import (
    AdapterRef,
    BackendCaps,
    Choice,
    GenRequest,
    GenResult,
    JsonSchema,
    Message,
    NoConstraint,
    Regex,
)

__all__ = ["MLXBackend"]

CAPS = BackendCaps(
    supports=frozenset(),  # nothing is *enforced* at decode time; see module docstring
    supports_batch=False,
    adapter_switch_cost="reload",
    deterministic_across_cache_states=False,  # measured; see bench
    max_context=32_768,
)


def _render_constraint(constraint: Any) -> str:
    """Describe the required output shape in the prompt.

    Advisory only. A base checkpoint may ignore it entirely, and how often it
    does is the A4 result.
    """
    if isinstance(constraint, JsonSchema):
        keys = list(constraint.schema.get("properties", {}))
        example = {k: "..." for k in keys}
        return (
            "Reply with one line of JSON and nothing else, in exactly this form:\n"
            f"{json.dumps(example)}"
        )
    if isinstance(constraint, Choice):
        return "Reply with exactly one of: " + ", ".join(constraint.options)
    if isinstance(constraint, Regex):
        return f"Reply matching this pattern exactly: {constraint.pattern}"
    return ""


class MLXScope(CacheScope):
    def __init__(self, backend: "MLXBackend", frozen: tuple[Message, ...]) -> None:
        self._backend = backend
        self._frozen = frozen
        self._cache = None
        self._prefix_tokens = 0
        self._primed = False

    @property
    def prefix_tokens(self) -> int:
        return self._prefix_tokens

    def generate(self, req: GenRequest) -> GenResult:
        import mlx.core as mx
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        started = time.perf_counter()
        prompt = self._backend._format(req)

        # Per-call seeding is what makes a step replayable. Set immediately
        # before generating so nothing between here and the sampler can consume
        # entropy and desynchronise a replay.
        mx.random.seed(req.sampling_seed)

        sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
        text = mlx_generate(
            self._backend.model,
            self._backend.tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            sampler=sampler,
            verbose=False,
        )

        for stop in req.stop:
            if stop in text:
                text = text.split(stop)[0]

        n_prompt = len(self._backend.tokenizer.encode(prompt))
        cache_hit = self._primed
        if not self._primed:
            self._prefix_tokens = n_prompt
            self._primed = True

        value, parse_ok = None, True
        if isinstance(req.constraint, JsonSchema):
            try:
                value = json.loads(text.strip())
            except json.JSONDecodeError:
                parse_ok = False
        elif isinstance(req.constraint, Choice):
            parse_ok = text.strip() in req.constraint.options

        return GenResult(
            text=text,
            parse_ok=parse_ok,
            value=value,
            prompt_tokens=n_prompt,
            prefill_tokens=0 if cache_hit else n_prompt,
            completion_tokens=len(self._backend.tokenizer.encode(text)),
            cache_hit=cache_hit,
            finish_reason="stop",
            wall_s=time.perf_counter() - started,
            backend_fingerprint=self._backend.fingerprint(),
        )


class MLXBackend(Generator):
    caps = CAPS

    def __init__(
        self,
        model_path: str,
        *,
        adapter_path: str | None = None,
        enable_thinking: bool = False,
        use_chat_template: bool | None = None,
    ) -> None:
        from mlx_lm import load

        self.model_path = model_path
        self.adapter_path = adapter_path
        self.enable_thinking = enable_thinking
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)

        # Auto-detection by "does a chat template exist" is WRONG for Qwen3.
        # Qwen ships chat_template.jinja with Qwen3-4B-*Base*, so the presence of
        # a template says nothing about whether the checkpoint was trained to
        # follow one. Feeding a base checkpoint chat-formatted prompts makes it
        # echo the scaffolding — observed emitting bare "assistant" and
        # "ʁsystem\nHere is the shape of a reply." instead of an action.
        #
        # That is a measurement artefact, not a model property, and it would
        # have been read as "base checkpoints cannot hold an action loop."
        # Callers comparing base against instruct MUST set this explicitly.
        if use_chat_template is None:
            use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        self._is_chat = use_chat_template

    def fingerprint(self) -> str:
        import mlx_lm

        return f"mlx-lm/{mlx_lm.__version__}|{self.model_path}|adapter={self.adapter_path}"

    def load_adapter(self, adapter: AdapterRef | None) -> None:
        from mlx_lm import load

        path = adapter.path if adapter else None
        if path == self.adapter_path:
            return
        self.adapter_path = path
        self.model, self.tokenizer = load(self.model_path, adapter_path=path)

    @contextlib.contextmanager
    def cache_scope(self, frozen: tuple[Message, ...]) -> Iterator[CacheScope]:
        yield MLXScope(self, frozen)

    def close(self) -> None:
        self.model = None

    def _format(self, req: GenRequest) -> str:
        messages = [
            {"role": m.role, "content": m.content} for m in req.parts.all_messages()
        ]
        hint = _render_constraint(req.constraint)
        if hint:
            messages[-1] = {
                **messages[-1],
                "content": messages[-1]["content"] + "\n\n" + hint,
            }

        if self._is_chat:
            # Qwen3 ships hybrid thinking ON by default. Left enabled, every
            # generation opens `<think>\nOkay, the user is...` and burns the
            # entire token budget before reaching an action — measured at 0/3
            # parseable on Qwen3-4B-Instruct at 120 max_tokens.
            #
            # It is disabled here rather than accommodated because the spec's
            # deliberation budget is the harness's own mechanism: reasoning
            # tokens are metered, priced against acting, and rendered back to
            # the agent as a felt constraint. An uncontrolled provider-side
            # thinking block would compete with that and make the budget
            # unenforceable.
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Template does not accept the kwarg; nothing to disable.
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        # Base checkpoints have no chat template. Flatten to plain text; the
        # few-shot prefix in the caller's prompt is what carries the format.
        return "\n\n".join(m["content"] for m in messages) + "\n"
