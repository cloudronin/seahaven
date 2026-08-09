"""One place that decides how a (system, user) pair becomes a prompt string.

Training data and inference prompts MUST be built the same way. If the model
generates through a chat template but trains on raw concatenated text, the
adapter is fitted to a distribution the model is never shown at inference, and
its effect can fail to appear — producing a null result that is an artefact of
prompt formatting rather than a fact about the data.

That failure is silent and it points the wrong way: it looks like "training did
nothing," which is exactly the conclusion the A1b gate is built to test. So the
formatting decision lives here, once, and every caller goes through it.
"""

from __future__ import annotations

__all__ = ["render_prompt", "wants_chat_template"]


def wants_chat_template(model_path: str) -> bool:
    """Whether this checkpoint should be prompted through a chat template.

    NOT the same question as "does a chat template exist." Qwen ships
    `chat_template.jinja` with Qwen3-4B-**Base**, so template presence is not
    evidence the checkpoint was trained to follow one. Feeding a base checkpoint
    chat-formatted prompts makes it echo the scaffolding — observed emitting bare
    "assistant" instead of an action.
    """
    name = model_path.lower()
    return "base" not in name


def render_prompt(tokenizer, system: str, user: str, *, chat: bool) -> str:
    """Build the prompt string for generation or for a training example."""
    if chat and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{system}\n\n{user}\n"
