"""The `expdx` front door: the $0 verbs, and the guarantees they carry."""

from __future__ import annotations

import os

import pytest

from expedientbench import cli
from expedientbench.commands import doctor, emit, read, seeds, worlds


class _A:
    """Minimal args stand-in."""
    def __init__(self, **kw):
        self.results = "results"
        self.level = None
        self.model = None
        self.generation = None
        self.check = None
        self.artifact = None
        self.__dict__.update(kw)


def test_every_verb_has_help_which_no_script_ever_did():
    """Fourteen scripts used raw sys.argv, so a mistyped flag silently ran the
    other path and spent real money. argparse refuses instead."""
    p = cli.build_parser()
    with pytest.raises(SystemExit) as e:
        p.parse_args(["read", "--nonsense"])
    assert e.value.code == 2


def test_the_FREE_verbs_run_with_NO_KEY_AT_ALL(monkeypatch, capsys):
    """A command that costs nothing must not require credentials. Asserted by
    removing every provider key from the environment first."""
    for env in ("TOGETHER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(env, raising=False)
    assert doctor.main(_A()) == 0
    assert seeds.main(_A(check=[99000, 24])) == 0
    assert read.main(_A(level=["W2"], generation="gen3")) == 0
    assert emit.main(_A(artifact="spend")) == 0
    out = capsys.readouterr().out
    assert "Together  no" in out and "OpenAI    no" in out


def test_doctor_reports_open_and_closed_pins_CORRECTLY(capsys):
    assert doctor.main(_A()) == 0
    out = capsys.readouterr().out
    assert "round13  OPEN, verifies" in out
    assert "round6   CLOSED (refuses, as designed)" in out
    assert "retired_w_hash:ok" in out
    assert "DOES NOT RECOMPUTE" not in out


def test_seeds_check_is_PER_MODEL(capsys):
    """15000 is round 10's block. It collides for Qwen3.5-9B and not for a model
    that never used it — because seed space is per model."""
    assert seeds.main(_A(check=[15000, 8], model="Qwen/Qwen3.5-9B")) == 1
    assert "COLLISION" in capsys.readouterr().out
    assert seeds.main(_A(check=[15000, 8], model="gpt-5.6-terra")) == 0
    assert "FREE" in capsys.readouterr().out


def test_read_reports_the_OCCASION_SOURCE_not_just_a_timestamp(capsys):
    """**The bug this caught in its first version.** `read` rebuilt the path and
    passed an EMPTY meta to `occasion_of`, which forced the mtime fallback for
    every row — including the eight cells that carry a real serving timestamp.
    The provenance design is worthless if the first consumer discards it."""
    read.main(_A(level=["LAT"], generation="gen3", model=["gpt-5.6-terra"]))
    out = capsys.readouterr().out
    assert "(wall_start_epoch)" in out, "a real serving timestamp was shown as mtime"
    read.main(_A(level=["LAT"], generation="gen3", model=["Qwen/Qwen3.5-9B"]))
    assert "(mtime)" in capsys.readouterr().out


def test_read_prints_BOTH_metrics_and_the_route(capsys):
    read.main(_A(level=["LAT"], generation="gen3", model=["gpt-5.6-terra"]))
    out = capsys.readouterr().out
    assert "rate_any" in out and "intent" in out and "gap" in out
    assert "BINDS_AT_EAT" in out, "the route must sit beside the label"


def test_read_NEVER_pools_across_generations(capsys):
    """The prompt and the death semantics differ at each boundary."""
    read.main(_A(level=["LAT"], model=["deepcogito/cogito-v2-1-671b"]))
    out = capsys.readouterr().out
    gens = {ln.split()[2] for ln in out.splitlines()
            if "cogito" in ln and len(ln.split()) > 3}
    assert len(gens) > 1, "cogito has cells in several generations; keep them apart"


def test_emit_occasions_states_the_source_per_row(capsys):
    assert emit.main(_A(artifact="occasions")) == 0
    out = capsys.readouterr().out
    assert "wall_start_epoch" in out and "mtime" in out
    assert "mtime is NOT a serving date" in out


def test_worlds_validates_and_names_failures(capsys):
    assert worlds.main(_A(level=["LAT", "W2", "W3", "LAT2"])) == 0
    out = capsys.readouterr().out
    assert "lock ok" in out and "necessity ok" in out
    assert "0 failing check(s)" in out


def test_no_user_facing_string_names_the_internal_package(capsys):
    """The internal library keeps its name because pins hash its paths. It must
    not appear in any command's output.

    Matched on WORD BOUNDARIES, not substrings: the first version of this test
    failed on the word "credentials", which contains "eden". A check that fires
    on an unrelated English word teaches you to ignore it.
    """
    import re
    doctor.main(_A())
    seeds.main(_A())
    worlds.main(_A(level=["LAT"]))
    read.main(_A(level=["W2"], generation="gen3"))
    # The echoed working directory is the USER'S path, not our vocabulary, and
    # it happens to be named after the repo. Everything else is ours.
    out = "\n".join(ln for ln in capsys.readouterr().out.lower().splitlines()
                    if not ln.startswith("working dir"))
    leaks = re.findall(r"\bseahaven\b|\beden\b|eden_|world_eden", out)
    assert not leaks, f"internal vocabulary leaked into user-facing output: {leaks}"
