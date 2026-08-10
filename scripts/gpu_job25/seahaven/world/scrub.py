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

from __future__ import annotations

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
