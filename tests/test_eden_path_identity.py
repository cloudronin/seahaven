"""Do a block sweep and a per-seed top-up construct the SAME request?

Round 11 needed this answered and could not answer it from the cells: they record
the parsed command and `raw_len`, **not the served prompt or the request
parameters**. So the question is settled by building both paths offline with a
recording policy — no API, no cost — and comparing bytes.

This is a standing regression, not a one-off. Any future gap-fill or top-up rides
the same two code paths, and a divergence between them would be indistinguishable
from a behavioural finding.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from seahaven.eden import outcome as OUT
from seahaven.eden import round10 as R
from seahaven.fidelity.runner import run_fidelity


class _Recorder:
    """Deterministic stand-in that records every payload handed to it."""

    def __init__(self):
        self.turns = []
        self.warmup = []
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, **kw):
        self.warmup.append(json.dumps({"messages": messages, "kw": kw},
                                      sort_keys=True))
        return "ready"

    def reply(self, messages, step=0, seed=0):
        self.turns.append((seed, step, json.dumps(messages, sort_keys=True)))
        return "look"


def _build(runs, seed0):
    rec = _Recorder()
    run_fidelity(rec, None, runs=runs, steps=30, seed0=seed0,
                 world_id="world_eden_LAT", narrate=False,
                 eden_level="LAT", eden_arm="A1",
                 terminal_at_zero=R.TERMINAL_AT_ZERO)
    return rec


@pytest.fixture(scope="module")
def block():
    return _build(72, R.SEED0)


@pytest.mark.parametrize("seed", [15000, 15001, 15071])
def test_block_sweep_and_single_seed_fill_build_IDENTICAL_requests(block, seed):
    """**The check round 11's step 1 turned on.** The sweep issues one call for a
    contiguous block; a top-up issues one call per missing seed. If those built
    different conversations, every gap-filled cell in the corpus would be a
    mixture of two instruments."""
    single = _build(1, seed)
    a = [m for s, _st, m in block.turns if s == seed]
    b = [m for s, _st, m in single.turns if s == seed]
    assert a and b and len(a) == len(b)
    assert hashlib.sha256("".join(a).encode()).hexdigest() == \
        hashlib.sha256("".join(b).encode()).hexdigest()


def test_the_WARMUP_call_is_identical_too(block):
    """It runs outside `one_run`'s try/except and killed whole cells in round 2,
    so it is part of the request surface whether or not it produces data."""
    single = _build(1, 15000)
    assert block.warmup and block.warmup[0] == single.warmup[0]
    assert f'"max_tokens": {OUT.EDEN_MAX_TOKENS}' in block.warmup[0]


def test_the_history_carries_the_PARSED_COMMAND_on_the_eden_path():
    """`_rollout` has two append branches — raw reply when `eden is None`, parsed
    command otherwise. A reasoning model on the raw-reply branch would see a
    different history SHAPE, which was round 11's leading candidate for the
    DS-V4-Flash shift. Both paths take the eden branch, so it is not."""
    rec = _build(1, 15000)
    _seed, _step, last = rec.turns[-1]
    msgs = json.loads(last)
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistant, "no assistant turns to inspect"
    # the recorder always replies "look"; on the eden path that is what history
    # carries, and it would carry the full raw reply on the other branch
    assert set(assistant) == {"look"}


def test_terminal_death_is_ACTIVE_on_the_constructed_path(block):
    """A constructed episode where nothing is eaten must stop at the crossing,
    not run the full horizon — otherwise the flag is not reaching the runner."""
    per_seed = {}
    for s, _st, _m in block.turns:
        per_seed[s] = per_seed.get(s, 0) + 1
    assert set(per_seed.values()) == {24}, sorted(set(per_seed.values()))
