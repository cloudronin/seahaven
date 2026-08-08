"""Vendored world loader for the GPU job — scrub.py + loader.py inlined.

Shipped flat rather than as a package so the job needs no install step and no
PYTHONPATH juggling. Behaviour is identical to the originals; only the module
boundary is removed.
"""

from __future__ import annotations

"""Strip engine artifacts from observation text before it reaches the agent.

This is the ONLY module permitted to turn raw z-machine output into agent-facing
text. Everything it removes was verified against real TextWorld 1.7.0 / Jericho
3.3.1 output on this machine, not inferred from documentation.

Four classes of leak, in descending order of how badly they break containment:

1. The TextWorld banner. `env.reset()` prepends a large ASCII-art logo spelling
   TEXT WORLD in '$' characters. It tells the agent the name of the engine it is
   running inside, on step zero. Critically, a word-based lexicon matcher cannot
   see it: `"textworld" in obs.lower()` is False because the letters are drawn,
   not written. Banner suppression therefore cannot be delegated to the lexicon
   lint and has to happen here.

2. The status line. Jericho appends the z-machine status bar to the feedback
   text, e.g. `><spaces>-= Galley =-0/1`. The trailing digits are a score/turn
   readout. This appears even when the game is compiled with no quest and
   `max_score == 0`, so "compile without a quest" is necessary but not sufficient
   to keep numbers away from the agent.

3. Inform 7 score chatter, e.g. `[Your score has just gone up by one point.]`.
   Not reachable in a no-quest build, handled anyway because the cost is a regex
   and the failure mode is silent contamination of every downstream result.

4. Terminal banners, e.g. `*** You have won ***`. Same reasoning as (3).

`scrub_observation` is deliberately conservative about (1): it anchors on the
room-title marker rather than on a hardcoded picture of the logo, so a future
TextWorld release that redraws the art still gets stripped.
"""


import re

__all__ = ["scrub_observation", "ScrubReport", "scrub_with_report"]

# `-= Room Name =-` is Inform 7's room-title marker under TextWorld's rendering.
# It is the first thing in the observation that legitimately belongs to the agent.
_ROOM_TITLE = re.compile(r"^[ \t]*-=\s*(?P<name>.+?)\s*=-[ \t]*$", re.MULTILINE)

# The status line as Jericho appends it: an optional prompt, a run of whitespace,
# a room title, then an unlabelled `score/turns` pair. The digits are the reason
# this exists.
_STATUS_LINE = re.compile(
    r"^[ \t]*>?[ \t]*-=\s*.+?\s*=-[ \t]*\d+\s*/\s*\d+[ \t]*$",
    re.MULTILINE,
)

# Same thing when it is welded onto the end of a line rather than standing alone,
# which is what actually happens on `reset()`.
_STATUS_TAIL = re.compile(r">[ \t]{2,}-=\s*.+?\s*=-[ \t]*\d+\s*/\s*\d+[ \t]*$")

# Inform 7 score notifications. Point counts may be words or digits.
_SCORE_CHATTER = re.compile(
    r"\[Your score has (?:just )?gone (?:up|down) by [^\]]*\.?\]\s*",
    re.IGNORECASE,
)

# Terminal banners: *** You have won ***, *** You have died ***, etc.
_BANNER = re.compile(r"^\s*\*\*\*.*?\*\*\*\s*$", re.MULTILINE)

# A bare prompt left on its own line once the status tail is removed.
_DANGLING_PROMPT = re.compile(r"^[ \t]*>[ \t]*$", re.MULTILINE)

# Three or more blank lines collapse to one blank line.
_EXCESS_BLANKS = re.compile(r"\n{3,}")


class ScrubReport:
    """What `scrub_with_report` removed. Logged, never shown to the agent."""

    __slots__ = ("banner_stripped", "status_lines", "score_chatter", "banners")

    def __init__(self) -> None:
        self.banner_stripped: bool = False
        self.status_lines: int = 0
        self.score_chatter: int = 0
        self.banners: int = 0

    @property
    def anything_removed(self) -> bool:
        return bool(
            self.banner_stripped
            or self.status_lines
            or self.score_chatter
            or self.banners
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "banner_stripped": self.banner_stripped,
            "status_lines": self.status_lines,
            "score_chatter": self.score_chatter,
            "terminal_banners": self.banners,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScrubReport({self.as_dict()})"


def scrub_with_report(raw: str) -> tuple[str, ScrubReport]:
    """Scrub `raw` and report what was removed.

    Returns the cleaned text and a `ScrubReport`. The report is diagnostic: a
    nonzero `score_chatter` count in a no-quest build means the world was built
    wrong, and that should surface as a glitch rather than be silently absorbed.
    """
    report = ScrubReport()
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # (2) status line, tail form first — it sits on the same line as the prompt.
    text, n = _STATUS_TAIL.subn("", text)
    report.status_lines += n
    text, n = _STATUS_LINE.subn("", text)
    report.status_lines += n

    # (1) banner: drop everything before the first room title. Anchoring on the
    # title rather than on the art means a redrawn logo is still removed. If no
    # room title is present (mid-episode observations usually have none), leave
    # the text alone — there is no banner to strip.
    match = _ROOM_TITLE.search(text)
    if match and text[: match.start()].strip():
        preamble = text[: match.start()]
        # Only treat it as a banner if it looks like art rather than prose. Real
        # prose before a room title is rare but possible; box-drawing characters
        # and long runs of '$' or '\' are not.
        if _looks_like_art(preamble):
            text = text[match.start():]
            report.banner_stripped = True

    # (3) score chatter.
    text, n = _SCORE_CHATTER.subn("", text)
    report.score_chatter += n

    # (4) terminal banners.
    text, n = _BANNER.subn("", text)
    report.banners += n

    text = _DANGLING_PROMPT.sub("", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)
    return text.strip(), report


def scrub_observation(raw: str) -> str:
    """Scrub `raw` and return only the cleaned text."""
    return scrub_with_report(raw)[0]


def _looks_like_art(chunk: str) -> bool:
    """True if `chunk` is ASCII art rather than prose.

    The TextWorld logo is drawn with '$', '\\', '/', '|' and '_'. Prose is not.
    Using a character-class ratio rather than matching the logo itself keeps this
    working if the art changes.
    """
    stripped = "".join(chunk.split())
    if not stripped:
        return False
    art_chars = sum(ch in "$\\/|_-" for ch in stripped)
    return art_chars / len(stripped) > 0.4


# ----------------------------------------------------------------------

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


import dataclasses as dc
import re
from pathlib import Path
from typing import Any

import textworld.gym
from textworld import EnvInfos


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
