"""Action parser tests.

Failure classification matters as much as success here: the parse-failure rate
per arm is kill criterion K3, and a rate is only interpretable if the failures
are distinguishable from each other.
"""

from __future__ import annotations

import json

import pytest

from seahaven.agent.parse import Action, FailureKind, ParseFailure, parse_action


def good(expect="the door gives", command="open the door"):
    return json.dumps({"expect": expect, "command": command})


class TestSuccess:
    def test_plain_object(self):
        act = parse_action(good())
        assert isinstance(act, Action)
        assert act.command == "open the door"
        assert act.expect == "the door gives"

    def test_fenced_block_is_repaired(self):
        act = parse_action(f"```json\n{good()}\n```")
        assert isinstance(act, Action)
        assert act.command == "open the door"

    def test_leading_prose_is_repaired(self):
        act = parse_action(f"Here is my action:\n{good()}")
        assert isinstance(act, Action)

    def test_leading_angle_bracket_stripped(self):
        act = parse_action(good(command="> go north"))
        assert isinstance(act, Action)
        assert act.command == "go north"

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("go north", "go"),
            ("take the brass key", "take"),
            ("LOOK", "look"),
            ("  examine crate  ", "examine"),
        ],
    )
    def test_kind_is_the_verb(self, command, expected):
        act = parse_action(good(command=command))
        assert act.kind == expected


class TestFailures:
    def test_empty(self):
        f = parse_action("")
        assert isinstance(f, ParseFailure) and f.kind is FailureKind.EMPTY

    def test_whitespace_only(self):
        f = parse_action("   \n  ")
        assert f.kind is FailureKind.EMPTY

    def test_not_json(self):
        f = parse_action("I think I will go north.")
        assert f.kind is FailureKind.NOT_JSON

    def test_missing_field(self):
        f = parse_action(json.dumps({"command": "look"}))
        assert f.kind is FailureKind.WRONG_SHAPE
        assert "expect" in f.detail

    def test_non_string_field(self):
        f = parse_action(json.dumps({"expect": 3, "command": "look"}))
        assert f.kind is FailureKind.WRONG_SHAPE

    def test_blank_command(self):
        f = parse_action(good(command="   "))
        assert f.kind is FailureKind.EMPTY_COMMAND

    def test_array_instead_of_object(self):
        f = parse_action("[1, 2, 3]")
        assert f.kind in (FailureKind.WRONG_SHAPE, FailureKind.NOT_JSON)


class TestHallucinatedObservation:
    """The classic base-model failure: writing the world's reply too."""

    def test_room_title_detected(self):
        raw = good() + "\n\n-= Store =-\nShelves, mostly empty."
        f = parse_action(raw)
        assert isinstance(f, ParseFailure)
        assert f.kind is FailureKind.HALLUCINATED_OBSERVATION

    def test_next_prompt_detected(self):
        raw = good() + "\n> go north"
        f = parse_action(raw)
        assert f.kind is FailureKind.HALLUCINATED_OBSERVATION

    def test_detail_points_at_the_stop_sequence(self):
        f = parse_action(good() + "\n-= Store =-")
        assert "stop sequence" in f.detail


class TestFallback:
    def test_fallback_is_a_valid_action(self):
        f = parse_action("garbage")
        act = f.fallback()
        assert isinstance(act, Action)
        assert act.command == "look"

    def test_fallback_preserves_raw_for_forensics(self):
        f = parse_action("garbage")
        assert f.fallback().raw == "garbage"
