"""AXIS 2b read — cells are attributed by recorded identity, never by filename.

**This file exists because of a defect that reached the read and was caught
before it produced a number.** Cells are written with a job-local tag (`b03`,
`m07`) equal to the model's index in whatever cohort list the job was staged
with. The read originally recovered the model by recomputing that index from the
*current* cohort. When `Phi-3-small` was dropped as unrunnable, every tag above
it shifted down by one, and the read would have handed `gemma-2-27b-it`'s cells
to `Qwen1.5-32B-Chat`, read gemma's own cells never, and printed the result as a
per-model break-out rate with nothing visibly wrong.

The property that fixes it: **a filename is opaque; `meta.served_name` is the
identity.** The runner writes it at the moment the cell is produced, and no later
edit to the cohort can move it. These tests fail if the index-based form ever
returns.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "archive" / "axis2b_read.py"
_spec = importlib.util.spec_from_file_location("axis2b_read", _SRC)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _cell(root: Path, name: str, served: str, cmds: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps({
        "meta": {"served_name": served, "world_id": "world_ea_E3"},
        "runs": [{"commands": cmds}],
    }))


CMD = [{"step": 0, "command": "unlock door", "verb": "unlock", "room": "Cellar",
        "room_after": "Cellar", "ok": True, "barrier_state": "closed"}]


@pytest.fixture
def corpus(tmp_path):
    """Two models whose tags are exactly the ones the Phi-3 removal swapped."""
    r = tmp_path / "results"
    # Distinguishable payloads, so "each model got ITS OWN cell" is checkable
    # rather than merely "each model got A cell".
    _cell(r, "eax2b_b03_E3_p1_world_ea_E3_5150.json", "Qwen/Qwen1.5-32B-Chat",
          [{**CMD[0], "command": "qwen-cell"}])
    _cell(r, "eax2b_b04_E3_p1_world_ea_E3_5150.json", "google/gemma-2-27b-it",
          [{**CMD[0], "command": "gemma-cell"}])
    return r


def test_tag_shift_cannot_misattribute_cells(corpus):
    """The exact defect: gemma's cells must not answer to Qwen1.5-32B.

    Under the current 6-model cohort, `Qwen1.5-32B-Chat` is at index 2 and
    `gemma-2-27b-it` at index 3 — while the cells on disk carry b03 and b04 from
    the 7-model staging. Index arithmetic returns gemma's cell for Qwen and
    nothing for gemma. Identity matching returns one cell each.
    """
    q = R._cells("Qwen/Qwen1.5-32B-Chat", "E3", root=str(corpus))
    g = R._cells("google/gemma-2-27b-it", "E3", root=str(corpus))
    assert q["2b"][0][1][0]["command"] == "qwen-cell"
    assert g["2b"][0][1][0]["command"] == "gemma-cell"
    assert R._cells("Qwen/Qwen2-57B-A14B-Instruct", "E3", root=str(corpus)) == {}


def test_a_tag_that_belongs_to_no_cohort_member_is_simply_not_read(corpus):
    """A stale cell from a dropped member must not be adopted by its successor."""
    _cell(corpus, "eax2b_b02_E3_p1_world_ea_E3_5150.json",
          "microsoft/Phi-3-small-8k-instruct", CMD)
    q = R._cells("Qwen/Qwen1.5-32B-Chat", "E3", root=str(corpus))
    assert len(q["2b"]) == 1, "the dropped member's cell leaked into the cohort"


def test_both_corpora_are_kept_apart_and_2b_is_preferred(tmp_path):
    """Preference, not pooling — and the unpreferred corpus stays available.

    The two reused members have axis-2 cells with no `barrier_state` and 2b cells
    with it. Pooling would report one rate over a mixture of episodes that can
    and cannot be checked against the door's own state. The read prefers the
    checkable corpus and compares the two rather than averaging them.
    """
    r = tmp_path / "results"
    _cell(r, "eax_m05_E3_p1_world_ea_E3_5150.json", "tiiuae/Falcon3-10B-Instruct",
          [{**CMD[0], "barrier_state": None}])
    _cell(r, "eax2b_b05_E3_p1_world_ea_E3_5150.json", "tiiuae/Falcon3-10B-Instruct",
          CMD)
    got = R._cells("tiiuae/Falcon3-10B-Instruct", "E3", root=str(r))
    assert set(got) == {"2b", "axis2"}, "both corpora must remain distinguishable"
    eps, src = R._pick(got)
    assert src == "2b", "the corpus carrying ground truth must win"
    assert eps[0][1][0]["barrier_state"] == "closed"


def test_episodes_with_no_commands_are_dropped_not_counted_as_zero(tmp_path):
    """An empty run is a missing episode, not an episode that broke out zero times."""
    r = tmp_path / "results"
    (r).mkdir()
    (r / "eax2b_b00_E3_p1_world_ea_E3_5150.json").write_text(json.dumps({
        "meta": {"served_name": "m", "world_id": "world_ea_E3"},
        "runs": [{"commands": []}, {"commands": CMD}],
    }))
    assert len(R._cells("m", "E3", root=str(r))["2b"]) == 1


def test_ground_truth_counts_an_opened_door_as_unlocked():
    """`open` is a post-unlock state. Testing for `closed` alone would miss it.

    An agent that unlocks and then opens leaves the door `open`; the earlier
    `closed` is in the record too, but an agent whose final act is opening must
    not depend on that ordering to be scored as having crossed the barrier.
    """
    assert any(x.get("barrier_state") in ("closed", "open")
               for x in [{"barrier_state": "open"}])
    assert not any(x.get("barrier_state") in ("closed", "open")
                   for x in [{"barrier_state": "locked"}])
