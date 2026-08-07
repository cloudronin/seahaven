"""A5 — the committed world artifacts work, and the agent/hidden split holds.

This is the spike the plan calls A5, kept as a permanent regression test. It runs
against the real committed `.z8`, not a fixture, because the things it checks are
exactly the things that differ between "the file we built" and "the file in git".

The failure it exists to prevent: a `.z8` committed without its `.json` sidecar
produces empty `facts`/`entities` at runtime, silently, and the ledger records
nothing for the whole run. That is a 40-hour loss discovered at analysis time.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import pytest

from seahaven.world.loader import open_world

WORLD_DIR = Path(__file__).resolve().parents[1] / "worlds" / "world_v0"
Z8 = WORLD_DIR / "world_v0.z8"
SIDECAR = WORLD_DIR / "world_v0.json"
LOCK = WORLD_DIR / "BUILD.lock.json"

# A tour that exercises movement, the loop, the spurs, taking, containers, and
# supporters — enough surface that missing ground truth shows up.
TRAJECTORY = [
    "look", "take kettle", "inventory", "north", "look",
    "open crate", "take coil of rope", "east", "north", "look",
    "take logbook", "west", "look", "take oil can", "open locker",
    "south", "south", "look", "south", "look",
]


@pytest.fixture(scope="module")
def world():
    handle = open_world(Z8)
    yield handle
    handle.close()


class TestArtifactsCommitted:
    def test_z8_and_sidecar_both_present(self):
        assert Z8.exists(), "world .z8 missing"
        assert SIDECAR.exists(), (
            "sidecar missing — facts/entities would be empty at runtime"
        )

    def test_hashes_match_the_lock(self):
        lock = json.loads(LOCK.read_text())
        for name, expected in lock["artifacts"].items():
            actual = hashlib.sha256((WORLD_DIR / name).read_bytes()).hexdigest()
            assert actual == expected, f"{name} does not match BUILD.lock.json"

    def test_world_was_built_without_a_quest(self):
        assert json.loads(LOCK.read_text())["quest"] is None

    def test_open_world_refuses_a_missing_sidecar(self, tmp_path):
        orphan = tmp_path / "orphan.z8"
        orphan.write_bytes(Z8.read_bytes())
        with pytest.raises(FileNotFoundError, match="missing sidecar"):
            open_world(orphan)


class TestGroundTruthReachesInfos:
    """The sidecar gap. These fail loudly if the .json is absent or stale."""

    def test_facts_non_empty_at_reset(self, world):
        _, hidden = world.reset()
        assert len(hidden.facts) > 0, "facts empty — sidecar not being read"

    def test_entities_non_empty_at_reset(self, world):
        _, hidden = world.reset()
        assert len(hidden.entities) > 0

    def test_facts_track_the_world(self, world):
        _, before = world.reset()
        _, after = world.step("take kettle")
        assert set(before.facts) != set(after.facts), (
            "facts did not change after taking an object — ground truth is static"
        )

    def test_admissible_commands_present(self, world):
        _, hidden = world.reset()
        assert any(c.startswith("go ") for c in hidden.admissible_commands)


class TestAgentNeverSeesScore:
    """Every observation across a real trajectory, checked for leaks."""

    @pytest.fixture(scope="class")
    def transcript(self):
        handle = open_world(Z8)
        obs, hidden = handle.reset()
        rows = [(obs, hidden)]
        for command in TRAJECTORY:
            rows.append(handle.step(command))
        handle.close()
        return rows

    def test_no_status_line_digits(self, transcript):
        for obs, _ in transcript:
            assert not re.search(r"=-\s*\d+\s*/\s*\d+", obs.text), obs.text

    def test_no_score_word(self, transcript):
        for obs, _ in transcript:
            assert "score" not in obs.text.lower(), obs.text

    def test_no_terminal_banner(self, transcript):
        for obs, _ in transcript:
            assert "***" not in obs.text

    def test_no_engine_ascii_art(self, transcript):
        for obs, _ in transcript:
            assert "$$" not in obs.text

    def test_no_fact_string_leaks_verbatim(self, transcript):
        """Ground truth is structurally separate; assert it stays that way."""
        for obs, hidden in transcript:
            for fact in hidden.facts:
                assert fact not in obs.text, f"fact {fact!r} leaked into observation"

    def test_max_score_is_zero(self, transcript):
        """No quest means no score for the engine to leak in the first place."""
        for _, hidden in transcript:
            assert hidden.max_score in (0, None), hidden.max_score

    def test_observations_are_non_empty(self, transcript):
        for obs, _ in transcript:
            assert obs.text.strip(), "empty observation would starve the prompt"


class TestRoomTracking:
    """`infos["location"]` is never populated; room comes from the fact set."""

    def test_location_field_is_useless(self, world):
        """Documents why room is derived rather than read."""
        import textworld.gym

        from seahaven.world.loader import REQUESTED_INFOS

        env_id = textworld.gym.register_game(
            str(Z8), request_infos=REQUESTED_INFOS, max_episode_steps=100
        )
        env = textworld.gym.make(env_id)
        _, infos = env.reset()
        assert infos.get("location") is None, (
            "location is populated now — the fact-parsing fallback may be removable"
        )
        env.close()

    def test_room_is_tracked_across_a_tour(self):
        handle = open_world(Z8)
        obs, _ = handle.reset()
        seen = [obs.room]
        for command in ["north", "east", "north", "west", "south"]:
            obs, _ = handle.step(command)
            seen.append(obs.room)
        handle.close()
        assert seen == ["Galley", "Store", "Landing", "Lamp Room", "Workshop", "Store"]

    def test_room_never_none_after_reset(self, world):
        obs, _ = world.reset()
        assert obs.room is not None


class TestDeterminism:
    """Replay is the restore mechanism, so the engine must be deterministic."""

    def _run(self):
        handle = open_world(Z8)
        obs, _ = handle.reset()
        texts = [obs.text]
        for command in TRAJECTORY:
            obs, _ = handle.step(command)
            texts.append(obs.text)
        handle.close()
        return texts

    def test_same_commands_give_same_observations(self):
        assert self._run() == self._run()
