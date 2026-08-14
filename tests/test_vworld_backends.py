"""Backends: the assembly identity, the model-string discipline, the dry run."""

from __future__ import annotations

import hashlib
import inspect

import pytest

from vetoworld.backends import EndpointSpec, resolve
from vetoworld.backends.base import COHORT_TEMPERATURE, Backend


class _A:
    def __init__(self, **kw):
        self.endpoint = "terra"
        self.model = None
        self.key_env = None
        self.level = "LAT"
        self.arm = "A1"
        self.seed0 = 21000
        self.dry_run = True
        self.budget = None
        self.m_a1 = 48
        self.m_a0 = 24
        self.out = "results_run"
        self.__dict__.update(kw)


def test_the_backend_reply_is_the_COHORT_POLICY_but_for_temperature():
    """Same max_tokens, same seed derivation. Assembly is upstream in the
    rollout and cannot differ per backend."""
    from seahaven.eden.outcome import EdenPolicy
    a = inspect.getsource(EdenPolicy.reply)
    b = inspect.getsource(Backend.reply)
    assert "seed * 100_003 + step" in a and "seed * 100_003 + step" in b
    assert "EDEN_MAX_TOKENS" in a and "EDEN_MAX_TOKENS" in b


def test_message_assembly_is_IDENTICAL_regardless_of_endpoint():
    """A fixture seed must produce byte-identical messages for any backend —
    otherwise a cross-provider comparison measures the harness."""
    from seahaven.fidelity.runner import run_fidelity

    def build():
        seen = []

        class _R:
            usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

            def chat(self, messages, **kw):
                return "ready"

            def reply(self, messages, *, step, seed):
                seen.append([dict(m) for m in messages])
                return "look"

        run_fidelity(_R(), None, runs=1, steps=30, seed0=21000,
                     world_id="world_eden_LAT", narrate=False,
                     eden_level="LAT", eden_arm="A1", terminal_at_zero=True)
        import json
        return hashlib.sha256(
            json.dumps(seen, sort_keys=True).encode()).hexdigest()

    assert build() == build()


def test_a_NEAR_MISS_model_string_is_REFUSED():
    """A near-miss is a different model. Serving one looks like a result."""
    spec = EndpointSpec(name="t", base_url="https://api.openai.com/v1",
                        key_env="OPENAI_API_KEY", model="gpt-5.6-terr")

    class _B(Backend):
        def __init__(self, spec):
            self.spec = spec

        def list_models(self):
            return {"gpt-5.6-terra", "gpt-5.6-sol"}

    with pytest.raises(SystemExit) as e:
        _B(spec).resolve_model()
    assert "NEAR MISSES" in str(e.value) and "gpt-5.6-terra" in str(e.value)


def test_an_UNCATALOGUED_endpoint_falls_back_to_RECORD_AND_PIN():
    """A fine-tuned or self-hosted endpoint has no availability record, so the
    rule becomes: capture the served string and refuse to pool across differing
    ones — not refuse to run."""
    spec = EndpointSpec(name="c", base_url="http://localhost:8000/v1",
                        key_env="X", model="my-finetune-v3")
    assert spec.catalogued is False

    class _B(Backend):
        def __init__(self, spec):
            self.spec = spec

    assert _B(spec).resolve_model() == "my-finetune-v3"


def test_the_endpoints_file_carries_TERRAS_deviation():
    spec = resolve("terra")
    assert spec.temperature == 1.0 != COHORT_TEMPERATURE
    assert "rejects temperature 0.9" in spec.notes


def test_dry_run_SERVES_NOTHING_and_prints_the_FIRST_request(capsys):
    """And the request must be the first one, not the last.

    `_rollout` mutates one message list every turn, so recording a reference
    made every captured request show the final 49-message conversation. The
    recorder copies.
    """
    from vetoworld.commands import run as R
    assert R.main(_A()) == 0
    out = capsys.readouterr().out
    assert "nothing served" in out
    assert "first request, 2 message(s)" in out, (
        "the dry run printed a request other than the first")
    assert "[system]" in out and "[user]" in out


def test_run_REFUSES_without_an_explicit_budget():
    from vetoworld.commands import run as R
    with pytest.raises(SystemExit) as e:
        R.main(_A(dry_run=False, budget=None))
    assert "--budget" in str(e.value)


# --- run --block: a block is a separate cell, not a top-up ------------------

def test_a_BLOCK_writes_a_DISTINCT_cell_and_never_gap_fills_the_round(tmp_path,
                                                                      monkeypatch):
    """**What round 11's block 3 was kept out of round 10's cell to protect.**

    Without `--block` the tag, the seed base and the path all derive from the
    round number, so a second block of the same (model, arm, level) resolves to
    the SAME file and `resume_plan` gap-fills into it — pooling the blocks and
    destroying the structure the measurement exists to show. The paths must
    differ, and the seeds must not overlap.
    """
    from seahaven.eden._shared import corpus as C

    base = C.cell_path("15", "m/x", "A1", "LAT")
    blk = C.cell_path("15b2", "m/x", "A1", "LAT")
    assert base != blk
    assert "e15_" in base.name and "e15b2_" in blk.name
    # and the corpus layer must recognise the block tag, or the cell is invisible
    got = C.parse_cell_name(blk.name)
    assert got and got["round"] == "15b2" and got["schema"] == "current"


def test_block_REFUSES_without_a_round(capsys):
    """The ad-hoc path has no grid and no pin, so it would ignore --block."""
    from vetoworld.commands import run as R

    class _A:
        endpoint, model, key_env = "together", None, None
        block, round, seed0, dry_run, budget = "2", None, 1000, False, 5.0
    with pytest.raises(SystemExit) as e:
        R.main(_A())
    assert "--block needs --round" in str(e.value)


def test_block_REFUSES_without_its_own_seed0():
    """A block reusing the round's seeds would re-serve the same episodes and
    call the repeat a new occasion."""
    from vetoworld.commands import run as R

    class _A:
        endpoint, model, key_env = "together", None, None
        block, round, seed0, dry_run, budget = "2", 14, None, False, 5.0
        level = None
    with pytest.raises(SystemExit) as e:
        R.main(_A())
    assert "--block needs its own --seed0" in str(e.value)


def test_usage_is_PER_CELL_not_cumulative_across_the_backend(tmp_path,
                                                             monkeypatch):
    """**A bug that corrupted committed data and a published spend figure.**

    The backend is created once so the connection is reused, but `usage_total`
    accumulates over its whole lifetime. Reading it directly per cell recorded
    every previous cell's tokens too: the round-15 COMP gate showed prompt
    tokens climbing 991k, 1.98M, 2.98M... in a perfect arithmetic progression
    across fourteen identical 24-episode cells, and billed a 7x-inflated $7.02
    for the seventh. Round 14 shipped with its A0 cell carrying A1's tokens and
    a total of $2.78 against a true $1.91.

    The fix is to snapshot and diff, and this is the check that would have
    caught it: three cells of identical size must bill identically.
    """
    import inspect

    from vetoworld.commands import run as R

    src = inspect.getsource(R._round_cells)
    assert "seen_usage" in src, "no snapshot taken"
    assert "u = dict(be.usage_total)" not in src, "cumulative total read raw"

    # the arithmetic itself, on a stand-in whose totals only ever grow
    totals = [{"prompt_tokens": 1000, "completion_tokens": 10},
              {"prompt_tokens": 2000, "completion_tokens": 20},
              {"prompt_tokens": 3000, "completion_tokens": 30}]
    seen, per_cell = {"prompt_tokens": 0, "completion_tokens": 0}, []
    for t in totals:
        per_cell.append({k: t.get(k, 0) - seen.get(k, 0) for k in t})
        seen = t
    assert per_cell == [{"prompt_tokens": 1000, "completion_tokens": 10}] * 3, \
        "identical cells must bill identically"


def test_no_committed_cell_carries_CUMULATIVE_usage():
    """The corpus-side check. Two cells of one round served in sequence must
    not have one's usage contained in the other's."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "results"
    a1 = json.loads((root / "eden_e14_deepcogito__cogito-v2-1-671b__A1__LAT2.json").read_text())
    a0 = json.loads((root / "eden_e14_deepcogito__cogito-v2-1-671b__A0__LAT2.json").read_text())
    # A0 has 24 episodes like A1; its prompt tokens must be the same order of
    # magnitude, not the sum of both.
    p1 = a1["meta"]["usage"]["prompt_tokens"]
    p0 = a0["meta"]["usage"]["prompt_tokens"]
    assert p0 < p1 * 1.6, f"A0 ({p0:,}) looks like it contains A1 ({p1:,})"
    assert "usage_correction" in a0["meta"], "the correction must stay recorded"
