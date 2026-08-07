"""Scrubber tests.

The banner and status-line fixtures are verbatim TextWorld 1.7.0 / Jericho 3.3.1
output captured on this machine, not invented. That matters: the status line
`-= Galley =-0/1` was not anticipated by the design, and the ASCII banner is
invisible to any word-based lexicon check because the letters are drawn in '$'.
"""

from __future__ import annotations

import pytest

from seahaven.world.scrub import scrub_observation, scrub_with_report

# Captured verbatim from env.reset() on a no-quest world.
REAL_RESET = """



                    ________  ________  __    __  ________
                   |        \\|        \\|  \\  |  \\|        \\
                    \\$$$$$$$$| $$$$$$$$| $$  | $$ \\$$$$$$$$
                      | $$   | $$__     \\$$\\/  $$   | $$
                      | $$   | $$  \\     >$$  $$    | $$
                       \\$$    \\$$$$$$$$ \\$$   \\$$    \\$$
              __       __   ______   _______   __        _______
             |  \\  _  |  \\ /      \\ |       \\ |  \\      |       \\
             | $$ / \\ | $$|  $$$$$$\\| $$$$$$$\\| $$      | $$$$$$$\\
              \\$$      \\$$  \\$$$$$$  \\$$   \\$$ \\$$$$$$$$ \\$$$$$$$



-= Galley =-
A cramped galley. Salt has got into everything.

There is a kettle on the floor.

>                                                                                                                                -= Galley =-0/1"""

REAL_STEP = """-= Store =-
Shelves, mostly empty. It smells of paraffin.

>                                                                                                                                -= Store =-0/2"""


class TestBanner:
    def test_ascii_logo_is_removed(self):
        out = scrub_observation(REAL_RESET)
        assert "$$" not in out
        assert out.startswith("-= Galley =-")

    def test_word_search_would_not_have_caught_it(self):
        """The reason banner stripping cannot be delegated to the lexicon lint."""
        assert "textworld" not in REAL_RESET.lower()
        assert "text world" not in REAL_RESET.lower()

    def test_room_text_survives(self):
        out = scrub_observation(REAL_RESET)
        assert "A cramped galley. Salt has got into everything." in out
        assert "There is a kettle on the floor." in out

    def test_reports_the_strip(self):
        _, rep = scrub_with_report(REAL_RESET)
        assert rep.banner_stripped is True

    def test_prose_before_a_room_title_is_not_mistaken_for_art(self):
        text = "The wind drops.\n\n-= Galley =-\nA cramped galley."
        out, rep = scrub_with_report(text)
        assert rep.banner_stripped is False
        assert "The wind drops." in out


class TestStatusLine:
    @pytest.mark.parametrize("raw", [REAL_RESET, REAL_STEP])
    def test_score_turn_digits_are_removed(self, raw):
        out = scrub_observation(raw)
        assert "0/1" not in out
        assert "0/2" not in out

    def test_no_trailing_prompt_survives(self):
        out = scrub_observation(REAL_STEP)
        assert not out.rstrip().endswith(">")

    def test_step_text_survives(self):
        out = scrub_observation(REAL_STEP)
        assert out == "-= Store =-\nShelves, mostly empty. It smells of paraffin."

    def test_standalone_status_line(self):
        out = scrub_observation("You wait.\n-= Cistern =-12/340")
        assert "12/340" not in out
        assert "You wait." in out


class TestScoreChatter:
    @pytest.mark.parametrize(
        "chatter",
        [
            "[Your score has just gone up by one point.]",
            "[Your score has just gone up by 3 points.]",
            "[Your score has gone down by two points.]",
        ],
    )
    def test_removed(self, chatter):
        out = scrub_observation(f"You take the key.\n{chatter}\nThe lock turns.")
        assert "score" not in out.lower()
        assert "You take the key." in out
        assert "The lock turns." in out

    def test_counted(self):
        _, rep = scrub_with_report("a\n[Your score has just gone up by one point.]\nb")
        assert rep.score_chatter == 1

    def test_legitimate_bracketed_text_survives(self):
        out = scrub_observation("You read the label. [It is water-damaged.]")
        assert "[It is water-damaged.]" in out


class TestTerminalBanners:
    @pytest.mark.parametrize(
        "banner", ["*** You have won ***", "*** You have died ***"]
    )
    def test_removed(self, banner):
        out = scrub_observation(f"The floor gives way.\n\n{banner}\n")
        assert "***" not in out
        assert "The floor gives way." in out


class TestInvariants:
    """Properties that must hold for anything the agent is ever shown."""

    FIXTURES = [REAL_RESET, REAL_STEP, "", "   ", "You wait.", "-= Galley =-"]

    @pytest.mark.parametrize("raw", FIXTURES)
    def test_output_is_a_stripped_string(self, raw):
        out = scrub_observation(raw)
        assert isinstance(out, str)
        assert out == out.strip()

    @pytest.mark.parametrize("raw", FIXTURES)
    def test_idempotent(self, raw):
        once = scrub_observation(raw)
        assert scrub_observation(once) == once

    @pytest.mark.parametrize("raw", FIXTURES)
    def test_never_emits_a_score_readout(self, raw):
        import re

        out = scrub_observation(raw)
        assert not re.search(r"=-\s*\d+\s*/\s*\d+", out)
