"""A0 loop tests, run against the stub backend.

These check loop *mechanics*, not agent quality. The stub's policy is bad on
purpose; what matters is that the loop closes, records what it should, and never
lets hidden state reach the prompt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The one-off scripts moved to archive/ when the CLI absorbed the live
# family. Both roots are on the path so a test need not know which side
# of that move its subject landed on.
for _d in ("scripts", "archive"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _d))

from a0_bare_loop import expectation_met, run  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("a0")
    summary = run(steps=40, out_dir=out, seed=7)
    rows = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    return summary, rows


class TestLoopCloses:
    def test_produces_one_row_per_step(self, result):
        summary, rows = result
        assert len(rows) == summary["steps"] == 40

    def test_steps_are_contiguous(self, result):
        _, rows = result
        assert [r["step"] for r in rows] == list(range(40))

    def test_every_step_has_a_command(self, result):
        _, rows = result
        assert all(r["command"].strip() for r in rows)

    def test_every_step_has_an_observation(self, result):
        _, rows = result
        assert all(r["observation"].strip() for r in rows)

    def test_summary_is_written(self, result):
        summary, _ = result
        assert summary["rooms_visited"]


class TestDeterminism:
    # `gen.wall_s` is measured, not derived, so it differs run to run. Comparing
    # raw file bytes made this test flaky in a way that looked like a real
    # determinism failure. Determinism here means identical actions and
    # observations, not identical timings.
    VOLATILE = {"wall_s"}

    @staticmethod
    def _stable(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for key in TestDeterminism.VOLATILE:
                row.get("gen", {}).pop(key, None)
        return rows

    def test_same_seed_gives_same_trajectory(self, tmp_path):
        a = run(steps=25, out_dir=tmp_path / "a", seed=3)
        b = run(steps=25, out_dir=tmp_path / "b", seed=3)
        assert a["stats"] == b["stats"]
        assert self._stable(tmp_path / "a" / "trajectory.jsonl") == self._stable(
            tmp_path / "b" / "trajectory.jsonl"
        )

    def test_timing_really_is_the_only_difference(self, tmp_path):
        """Guards the exclusion above from hiding a real divergence."""
        run(steps=10, out_dir=tmp_path / "a", seed=5)
        run(steps=10, out_dir=tmp_path / "b", seed=5)
        rows_a = self._stable(tmp_path / "a" / "trajectory.jsonl")
        rows_b = self._stable(tmp_path / "b" / "trajectory.jsonl")
        assert [r["command"] for r in rows_a] == [r["command"] for r in rows_b]
        assert [r["observation"] for r in rows_a] == [r["observation"] for r in rows_b]

    def test_different_seed_gives_different_trajectory(self, tmp_path):
        a = run(steps=25, out_dir=tmp_path / "a", seed=3)
        b = run(steps=25, out_dir=tmp_path / "b", seed=4)
        assert a["stats"] != b["stats"]


class TestNothingHiddenLeaks:
    """The prompt is where a leak would actually hurt. Check it directly."""

    def test_no_score_in_any_prompt(self, result):
        _, rows = result
        for row in rows:
            for message in row["prompt"]:
                assert "score" not in message["content"].lower()

    def test_no_status_digits_in_any_prompt(self, result):
        import re

        _, rows = result
        for row in rows:
            for message in row["prompt"]:
                assert not re.search(r"=-\s*\d+\s*/\s*\d+", message["content"])

    def test_no_meta_vocabulary_in_the_frozen_prefix(self, result):
        """world_v0's framing must not name the apparatus."""
        _, rows = result
        system = rows[0]["prompt"][0]["content"].lower()
        for word in ("experiment", "episode", "task", "reward", "score",
                     "campaign", "trial", "agent", "model"):
            assert word not in system, f"{word!r} leaked into the system prompt"

    def test_expectation_result_is_recorded_but_not_shown(self, result):
        """The pleasure signal is logged; the agent must never see the verdict."""
        _, rows = result
        assert all("expectation_met" in r for r in rows)
        for row in rows:
            for message in row["prompt"]:
                assert "expectation_met" not in message["content"]


class TestCachePrefix:
    def test_frozen_prefix_is_identical_every_step(self, result):
        _, rows = result
        prefixes = {r["prompt"][0]["content"] for r in rows}
        assert len(prefixes) == 1, "frozen prefix drifted; the cache would miss"

    def test_cache_hits_after_the_first_call(self, result):
        _, rows = result
        assert rows[0]["gen"]["cache_hit"] is False
        assert all(r["gen"]["cache_hit"] for r in rows[1:])

    def test_only_the_first_call_prefills(self, result):
        _, rows = result
        assert rows[0]["gen"]["prefill_tokens"] > 0
        assert all(r["gen"]["prefill_tokens"] == 0 for r in rows[1:])


class TestExpectationMatching:
    def test_overlap_counts_as_met(self):
        assert expectation_met("the door will open", "The door opens.") is True

    def test_no_overlap_is_a_miss(self):
        assert expectation_met("the door will open", "Nothing happens.") is False

    def test_empty_expectation_is_unscored(self):
        assert expectation_met("", "anything") is None

    def test_stopword_only_expectation_is_unscored(self):
        assert expectation_met("it is a", "anything") is None
