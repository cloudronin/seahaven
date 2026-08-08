"""Open a compiled world and split what the agent may see from what it may not.

The split is the whole point. `textworld.gym`'s `step()` returns
`(obs, score, done, infos)`; `obs` is strictly the z-machine feedback text, and
everything else — ground-truth facts, entities, score, win/loss — arrives in a
structurally separate dict. This module makes that separation a type-level fact
rather than a convention, so a caller cannot reach the score by accident.

Two failure modes this guards against, both verified on real output:

  - `JerichoEnv._gather_infos()` populates admissible_commands/score/won/lost/
    moves but NOT facts or entities. Those come from the `.json` sidecar that
    `compile_game()` writes beside the `.z8`. Ship the `.z8` without its sidecar
    and the ledger is silently empty for the entire run. `open_world` refuses to
    start rather than let that happen.
  - The raw feedback text carries engine artifacts (the TextWorld banner, the
    status line's score/turn digits). Nothing here returns raw text; it is
    scrubbed on the way out.
"""

from __future__ import annotations

import dataclasses as dc
import re
from pathlib import Path
from typing import Any

import textworld.gym
from textworld import EnvInfos

from .scrub import ScrubReport, scrub_with_report

__all__ = ["WorldHandle", "Observation", "HiddenState", "open_world"]


@dc.dataclass(frozen=True)
class Observation:
    """Everything the agent is permitted to see from one engine step."""

    text: str
    """Scrubbed feedback. Safe to place in a prompt."""

    room: str | None
    """Room name, for orchestrator bookkeeping. Already present in `text`."""

    description: str
    """The current room as `look` would report it.

    Agent-facing and safe: it is exactly what the engine prints for `look`, so
    showing it grants nothing the agent could not fetch itself for the cost of a
    turn. It is NOT `admissible_commands`, which stays hidden.

    Carrying it every step matters. Action results are terse — `examine kettle`
    returns "Tin, dented on one side." — so an agent that saw the room once on
    entry has no standing view of its surroundings, including where the exits
    are. Without this the loop degenerates: 60 steps, 3 unique commands, never
    leaving the first room.
    """


@dc.dataclass(frozen=True)
class HiddenState:
    """Everything the agent must never see. Operator-only sink.

    Kept as a separate frozen dataclass with no path into prompt assembly. The
    reward the spec requires discarding is `score`, and it is dropped by never
    being read out of here.
    """

    score: int
    done: bool
    won: bool | None
    lost: bool | None
    moves: int | None
    """Jericho's turn counter. UNRELIABLE — do not use it as a step index.

    Jericho warns `UnsupportedGameWarning: ... Score, move, change detection will
    be disabled` for TextWorld-compiled games, because it has no game-specific
    handler for them. The consequence is measurable: after three accepted
    commands the z-machine status line read turn 3 while this field reported 1.
    The orchestrator's own step counter is the authority; this is logged for
    forensics only.
    """

    max_score: int | None
    facts: tuple[str, ...]
    entities: tuple[str, ...]
    admissible_commands: tuple[str, ...]
    scrub: ScrubReport


class WorldHandle:
    """A live world. Wraps `textworld.gym` and enforces the split."""

    def __init__(self, env: Any, world_path: Path) -> None:
        self._env = env
        self.world_path = world_path
        self._last_room: str | None = None
        self._last_description: str = ""

    def reset(self) -> tuple[Observation, HiddenState]:
        raw, infos = self._env.reset()
        return self._split(raw, score=0, done=False, infos=infos)

    def step(self, command: str) -> tuple[Observation, HiddenState]:
        # The 4-tuple is unpacked here and nowhere else. `score` goes straight
        # into HiddenState; no caller of this method ever sees it positionally.
        raw, score, done, infos = self._env.step(command)
        return self._split(raw, score=score, done=done, infos=infos)

    def close(self) -> None:
        self._env.close()

    def _split(
        self, raw: str, *, score: int, done: bool, infos: dict[str, Any]
    ) -> tuple[Observation, HiddenState]:
        text, report = scrub_with_report(raw)
        room = _room_of(infos, fallback=self._last_room)
        self._last_room = room

        # `description` is engine text and gets the same scrubbing as feedback.
        description = scrub_with_report(infos.get("description") or "")[0]
        if description:
            self._last_description = description
        else:
            description = getattr(self, "_last_description", "")

        return (
            Observation(text=text, room=room, description=description),
            HiddenState(
                score=score,
                done=done,
                won=infos.get("won"),
                lost=infos.get("lost"),
                moves=infos.get("moves"),
                max_score=infos.get("max_score"),
                facts=tuple(str(f) for f in (infos.get("facts") or ())),
                entities=tuple(infos.get("entities") or ()),
                admissible_commands=tuple(infos.get("admissible_commands") or ()),
                scrub=report,
            ),
        )


# Requested wholesale. `facts` and `entities` are the ones that need the sidecar;
# `score`/`max_score`/`won`/`lost` are requested so they can be logged to the
# hidden sink, which is also how we prove they are not reaching the agent.
REQUESTED_INFOS = EnvInfos(
    facts=True,
    entities=True,
    admissible_commands=True,
    description=True,
    inventory=True,
    location=True,
    last_action=True,
    won=True,
    lost=True,
    score=True,
    max_score=True,
    moves=True,
)


def open_world(z8_path: str | Path, *, max_episode_steps: int = 100_000) -> WorldHandle:
    """Open `z8_path`, refusing to start if the sidecar is missing.

    `max_episode_steps` defaults high because campaign length is the
    orchestrator's decision, not the engine's — the spec is explicit that a
    campaign ends on step count and never on depletion or goal achievement.
    """
    path = Path(z8_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"world not found: {path}")

    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"missing sidecar {sidecar.name} beside {path.name}.\n"
            "JerichoEnv does not populate facts/entities on its own; without the "
            "sidecar they arrive empty and the ledger records nothing, silently, "
            "for the whole run."
        )

    env_id = textworld.gym.register_game(
        str(path), request_infos=REQUESTED_INFOS, max_episode_steps=max_episode_steps
    )
    return WorldHandle(textworld.gym.make(env_id), path)


# The player's room, as it appears in the sidecar-backed fact set:
#   at(P, Store: r)
_PLAYER_AT = re.compile(r"^at\(P,\s*(?P<room>.+?):\s*r\)$")


def _room_of(infos: dict[str, Any], *, fallback: str | None) -> str | None:
    """Derive the current room from ground-truth facts.

    `infos["location"]` is requested but comes back `None` — Jericho does not
    populate it, and unlike facts/entities it is not recovered from the sidecar
    either. The fact set is the reliable source, so parse it from there.
    """
    for fact in infos.get("facts") or ():
        match = _PLAYER_AT.match(str(fact))
        if match:
            return match.group("room")
    return fallback
