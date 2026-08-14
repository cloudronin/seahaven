"""Backends. One OpenAI-compatible client, no translation layer.

**Deliberately no litellm or per-provider translation in the serving path.** The
serving diagnostic ran on byte-level payload diffs, and every byte between the
policy and the wire has to stay auditable. One generic OpenAI-compatible client
covers Together, OpenAI, xAI, Modal, Moonshot and any self-hosted vLLM or Ollama
endpoint — including fine-tuned models.

Conversation assembly is shared and provider-agnostic: it happens upstream in the
rollout, so no backend can change it. A fixture seed must produce byte-identical
messages regardless of endpoint, and `tests/test_expdx_backends.py` asserts it.
"""

from .base import Backend, EndpointSpec, resolve

__all__ = ["Backend", "EndpointSpec", "resolve"]
