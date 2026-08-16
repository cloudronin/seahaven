"""The `vworld` front door: the $0 verbs, and the guarantees they carry."""

from __future__ import annotations

import os

import pytest

from vetoworld import cli
from vetoworld.commands import doctor, emit, read, seeds, worlds

#: **Corpus-wide tests: not unit tests, not timed like them (#112).** These
#: recompute every figure over the whole corpus, so they grow with it. The
#: project default of 120s caught that growth once, as an unexplained flake
#: rather than as the arithmetic it was. Measured at 425 cells, the slowest here
#: is ~53s.
pytestmark = pytest.mark.timeout(600)



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
    assert "round2   OPEN, verifies" in out
    assert "round6   CLOSED (refuses, as designed)" in out
    # Five closed in one commit at the LAT2 boundary, round 13 among them.
    assert "round13  CLOSED (refuses, as designed)" in out
    assert "retired_w_hash:ok" in out and "retired_r13_hash:ok" in out
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


#: The vocabulary that must never reach a user. `seahaven` and `eden_*` are
#: hashed into pin PATHS and can never be renamed; `expedientbench` is the name
#: the package had before it became VetoWorld. All three are internal-only, and
#: the public rule (`docs/naming.md`) is that none of them appears on a
#: user-facing surface.
_INTERNAL = r"\bseahaven\b|\beden\b|eden_|world_eden|expedientbench"


def _leaks(text: str) -> list[str]:
    """Matched on WORD BOUNDARIES, not substrings: the first version of this
    check failed on the word "credentials", which contains "eden". A check that
    fires on an unrelated English word teaches you to ignore it."""
    import re
    # The echoed working directory is the USER'S path, not our vocabulary, and
    # it happens to be named after the repo. Everything else is ours.
    body = "\n".join(ln for ln in text.lower().splitlines()
                     if not ln.startswith("working dir"))
    return re.findall(_INTERNAL, body)


def test_no_user_facing_string_names_the_internal_package(capsys):
    """The internal library keeps its name because pins hash its paths. It must
    not appear in any command's output."""
    doctor.main(_A())
    seeds.main(_A())
    worlds.main(_A(level=["LAT"]))
    read.main(_A(level=["W2"], generation="gen3"))
    got = _leaks(capsys.readouterr().out)
    assert not got, f"internal vocabulary leaked into user-facing output: {got}"


def test_EVERY_emitted_artifact_is_clean_too(capsys):
    """**The artifacts are the manuscript's tables**, so they are the surface a
    reader is most likely to see and the one a rename is most likely to miss.
    Checking only the four verbs above would have left eleven printed tables
    unguarded."""
    for art in emit.ARTIFACTS:
        assert emit.main(_A(artifact=art)) == 0, art
        got = _leaks(capsys.readouterr().out)
        assert not got, f"`emit {art}` leaked internal vocabulary: {got}"


def test_the_HELP_TEXT_carries_the_public_name_and_not_the_old_one(capsys):
    """Help text is the first thing a stranger reads and it is pure string, so
    it is exactly what a rename forgets."""
    p = cli.build_parser()
    p.print_help()
    for verb in ("read", "worlds", "emit", "run", "replicate", "probe", "pin",
                 "seeds", "corpus", "verify", "doctor"):
        with pytest.raises(SystemExit):
            p.parse_args([verb, "--help"])
    out = capsys.readouterr().out
    assert not _leaks(out), f"help text leaked: {_leaks(out)}"
    assert "VetoWorld" in out


def test_a_FAILED_builder_import_names_the_DEPENDENCY_not_a_symptom(monkeypatch):
    """**What the first CI release run reported, and why it was misleading.**

    `worlds` loads the world builder to enumerate authored levels. The loader
    registered the module in `sys.modules` before executing it, so a missing
    `textworld` left a half-built module cached — and the NEXT call skipped the
    load and failed with `no attribute 'all_levels'`, a symptom two steps
    downstream of the cause. The cache must be cleared on failure, and the
    message must name the dependency and the way around it.
    """
    import builtins
    import sys

    from vetoworld.commands import worlds as W

    real = builtins.__import__

    def blind(name, *a, **k):
        if name == "textworld":
            raise ModuleNotFoundError("nope", name="textworld")
        return real(name, *a, **k)

    sys.modules.pop("_vworld_builder", None)
    monkeypatch.setattr(builtins, "__import__", blind)
    try:
        for _ in range(2):          # twice: the second is where caching bit
            with pytest.raises(SystemExit) as e:
                W._levels(_A(level=None))
            assert "textworld" in str(e.value)
            assert "--level" in str(e.value), "must name the way around it"
            assert "_vworld_builder" not in sys.modules, "cache was poisoned"
    finally:
        sys.modules.pop("_vworld_builder", None)


# --- one canonical vocabulary list, asserted --------------------------------

def test_NO_SECOND_COPY_of_a_vocabulary_list_can_drift():
    """**The third instance of one failure, so it gets a guard rather than a
    third fix.**

    `doctor` carried its own round tuple and it stopped at 13 while `pin.ROUNDS`
    gained 14 — so the health check silently skipped the only round that could
    still serve cells. Before that, `emit.ARTIFACTS` and the `fn` dispatch dict
    drifted the same way, and before that the registry-disjointness lists.

    Each list has exactly one definition, and every other consumer imports it.
    """
    import inspect

    from vetoworld.commands import doctor, emit, pin

    # doctor must not define its own; it must read pin's.
    assert "from .pin import ROUNDS" in inspect.getsource(doctor)
    assert not [ln for ln in inspect.getsource(doctor).splitlines()
                if ln.strip().startswith("ROUNDS =")], "doctor redefines ROUNDS"

    # every round in the canonical tuple imports, and 14 is in it
    import importlib
    assert 14 in pin.ROUNDS, "the newest round is missing from the canonical list"
    for n in pin.ROUNDS:
        importlib.import_module(f"seahaven.eden.round{n}")

    # emit's ARTIFACTS tuple and its dispatch dict must agree, in both
    # directions — the tuple is what the leak test walks, the dict is what
    # actually dispatches, and an artifact in one but not the other is either
    # unreachable or unguarded.
    #
    # **This was a substring search over `main`'s source**, which is a proxy for
    # the property and not the property: it broke the moment the dict moved into
    # its own function, and it could never have caught a key present in the dict
    # but absent from the tuple. `registry()` exists so the sets can be compared
    # directly, in both directions, which is what the comment above always said.
    class _A:
        artifact = "__nope__"
    assert emit.main(_A()) == 1, "an unknown artifact must be refused"
    assert set(emit.ARTIFACTS) == set(emit.registry()), (
        set(emit.ARTIFACTS) ^ set(emit.registry()))
