"""Turn a raw generation into an action, or into a counted failure.

Constrained decoding guarantees the *shape* of the output, not that a backend
actually applied the constraint, and not that the model produced something
usable. Base (non-instruction-tuned) checkpoints are the reason this is a real
module and not a `json.loads` call: they wander, they continue past the turn
boundary, and they hallucinate the environment's next observation.

Every failure is classified rather than collapsed into a boolean, because the
parse-failure rate per arm IS kill criterion K3, and "differs across arms" is
only interpretable if you know *how* it differs.
"""

from __future__ import annotations

import dataclasses as dc
import json
import re
from enum import Enum

from .schemas import FALLBACK_COMMAND

__all__ = ["Action", "ParseFailure", "FailureKind", "parse_action"]


class FailureKind(str, Enum):
    EMPTY = "empty"
    NOT_JSON = "not_json"
    WRONG_SHAPE = "wrong_shape"
    EMPTY_COMMAND = "empty_command"
    HALLUCINATED_OBSERVATION = "hallucinated_observation"
    """The model wrote the world's reply as well as its own action.

    The classic base-model failure in an agent loop, and worth its own label: it
    means the stop sequence is wrong, not that the model is incapable.
    """


@dc.dataclass(frozen=True)
class Action:
    expect: str
    command: str
    raw: str

    @property
    def kind(self) -> str:
        """Coarse verb class, for action-weighted resource accounting."""
        head = self.command.strip().split()
        return head[0].lower() if head else "none"


@dc.dataclass(frozen=True)
class ParseFailure:
    kind: FailureKind
    detail: str
    raw: str

    def fallback(self) -> Action:
        return Action(expect="", command=FALLBACK_COMMAND, raw=self.raw)


# A model continuing past its turn usually starts drawing the room title or a
# new prompt.
_OBSERVATION_TELLS = (
    re.compile(r"^\s*-=\s*.+?\s*=-", re.MULTILINE),
    re.compile(r"^\s*>\s*\S", re.MULTILINE),
)

# Tolerate a fenced block or leading prose around the object. This is repair, not
# permissiveness: the alternative is discarding a well-formed action because the
# model wrapped it in ```json.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(raw: str) -> Action | ParseFailure:
    if not raw or not raw.strip():
        return ParseFailure(FailureKind.EMPTY, "no output", raw)

    for tell in _OBSERVATION_TELLS:
        if tell.search(raw):
            return ParseFailure(
                FailureKind.HALLUCINATED_OBSERVATION,
                "output contains a room title or prompt; check the stop sequence",
                raw,
            )

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(raw)
        if not match:
            return ParseFailure(FailureKind.NOT_JSON, "no JSON object found", raw)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return ParseFailure(FailureKind.NOT_JSON, str(exc), raw)

    if not isinstance(obj, dict):
        return ParseFailure(
            FailureKind.WRONG_SHAPE, f"expected object, got {type(obj).__name__}", raw
        )

    missing = {"expect", "command"} - obj.keys()
    if missing:
        return ParseFailure(
            FailureKind.WRONG_SHAPE, f"missing {sorted(missing)}", raw
        )

    command = obj["command"]
    expect = obj["expect"]
    if not isinstance(command, str) or not isinstance(expect, str):
        return ParseFailure(FailureKind.WRONG_SHAPE, "fields must be strings", raw)

    command = command.strip().lstrip(">").strip()
    if not command:
        return ParseFailure(FailureKind.EMPTY_COMMAND, "command is blank", raw)

    return Action(expect=expect.strip(), command=command, raw=raw)
