"""The action the agent emits each step.

Two fields, and the first one is not decoration.

`expect` implements the spec's pleasure mechanic: prediction confirmation rather
than a task scalar. The agent states what it thinks will happen; the orchestrator
logs match or mismatch and never shows the result. The signal decays as the world
becomes predictable, which supplies exploration pressure without specifying a
goal.

Note the containment consequence: because `expect` is a ritual the harness
imposes, it needs an in-world justification or it reads as an experimenter
artefact. In the real world build that justification is the station's standing
orders. Here in world_v0 it is unmotivated, which is acceptable only because
world_v0 is throwaway.

`command` is free text, not a choice over `admissible_commands`. Constraining it
to the admissible set would leak the world model — the agent would learn exactly
which verbs the engine understands, which is information the parser is not
supposed to give away, and which would also destroy the glitch log's value by
making parser rejections impossible.
"""

from __future__ import annotations

ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "expect": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "What you think will happen.",
        },
        "command": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "What you do.",
        },
    },
    "required": ["expect", "command"],
    "additionalProperties": False,
}

ACTION_SCHEMA_NAME = "action"

# Emitted after a parse failure so the agent still acts. Counted in the glitch
# log and reported as the K3 metric — a differential parse-failure rate across
# arms is a headline result, not an implementation detail.
FALLBACK_COMMAND = "look"
