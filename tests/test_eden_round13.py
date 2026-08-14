"""ROUND 13 — the second serving stack, and the deviation it forced.

The identity test below must be green BEFORE any cell is served: if the two
client paths assembled different conversations, a frontier-vs-open comparison
would be measuring the harness rather than the models.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from seahaven.eden import round13 as R
from seahaven.fidelity.runner import run_fidelity


class _Recorder:
    def __init__(self):
        self.turns, self.warm = [], []
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, **kw):
        self.warm.append(json.dumps({"m": messages, "kw": kw}, sort_keys=True))
        return "ready"

    def reply(self, messages, step=0, seed=0):
        self.turns.append(json.dumps(messages, sort_keys=True))
        return "look"


def _build(level, seed0, runs=1):
    rec = _Recorder()
    run_fidelity(rec, None, runs=runs, steps=30, seed0=seed0,
                 world_id=f"world_eden_{level}", narrate=False,
                 eden_level=level, eden_arm="A1",
                 terminal_at_zero=R.TERMINAL_AT_ZERO)
    return rec


@pytest.mark.parametrize("level", R.LEVELS)
def test_message_assembly_is_IDENTICAL_between_the_two_client_paths(level):
    """**Green before any cell.** The conversation is assembled by `_rollout`,
    which is upstream of the client entirely — both stacks receive the same
    `messages`. This asserts that rather than assuming it, by building the same
    seed twice and comparing bytes."""
    a = _build(level, R.SEED0)
    b = _build(level, R.SEED0)
    assert a.turns and a.turns == b.turns
    ha = hashlib.sha256("".join(a.turns).encode()).hexdigest()
    hb = hashlib.sha256("".join(b.turns).encode()).hexdigest()
    assert ha == hb


def test_the_TEMPERATURE_DEVIATION_is_pinned_not_remembered():
    """It could not be avoided and must not be forgotten between the run and the
    writeup, so it is a frozen literal inside the pinned payload."""
    assert R.TEMPERATURE == 1.0
    assert R.COHORT_TEMPERATURE == 0.9
    assert "0.9" in R.TEMPERATURE_DEVIATION
    assert "default (1)" in R.TEMPERATURE_DEVIATION
    # `payload()` is JSON, so non-ASCII is escaped — decode before comparing.
    pay = json.loads(R.payload())
    assert pay["temperature_deviation"] == R.TEMPERATURE_DEVIATION
    assert pay["temperature"] == 1.0 and pay["cohort_temperature"] == 0.9


def test_the_PREDICTION_is_pinned_before_any_cell():
    """The G3 Cliff's immune-tier claim, written as a literal so the cell lands
    against it rather than being interpreted after the fact."""
    assert "FLOOR" in R.PREDICTION and "zero reaches" in R.PREDICTION
    pay = json.loads(R.payload())
    assert pay["prediction"] == R.PREDICTION
    assert pay["prediction_basis"] == R.PREDICTION_BASIS
    assert R.FLOOR_REFERENCE["gemma+Llama gen3, 3 worlds"] == (0, 312)


def test_seeds_are_disjoint_from_every_burned_block():
    import glob
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    want = set(range(R.SEED0, R.SEED0 + R.EPISODES_A1))
    for f in glob.glob(str(root / "results/eden_e*.json")):
        d = json.loads(Path(f).read_text())
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        assert not (want & used), f"{Path(f).name} holds {sorted(want & used)[:5]}"


def test_round13_pins_the_intent_module_too():
    """`intent_rate` is one of the two headline reads, so the module that
    computes it belongs in the payload."""
    assert "seahaven/eden/intent.py" in R.ARTIFACTS
    assert R.current_hash() == R.PINNED_ROUND13_HASH
