"""Cross-run leakage guard.

"LoRA trains only on that run's trajectories, no cross-seed pooling" is the
isolation property the whole divergence claim rests on. If one run's data
reaches another run's adapter, every distance downstream is meaningless and
nothing in the analysis would reveal it.

The guard is a few lines and runs before any training time is spent. These tests
exist because the guard is the kind of code that gets quietly weakened when it
becomes inconvenient.
"""

from __future__ import annotations

import json

import pytest

from seahaven.backend.mlx_trainer import MLXTrainer
from seahaven.backend.types import TrainSpec


def spec_for(path, run_id="run_a"):
    return TrainSpec(
        base_model="unused", resume_from=None,
        train_jsonl=str(path), valid_jsonl=None,
        out_dir="unused", adapter_name="run_a_c1",
        rank=8, alpha=20.0, layers=8,
        iters=10, batch_size=1, learning_rate=1e-5,
        seed=0, run_id=run_id,
    )


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestSingleRunAssertion:
    def test_matching_run_id_passes(self, tmp_path):
        f = tmp_path / "train.jsonl"
        write(f, [{"run_id": "run_a", "prompt": "p", "completion": "c"}] * 3)
        MLXTrainer._assert_single_run(spec_for(f))

    def test_foreign_run_id_is_rejected(self, tmp_path):
        f = tmp_path / "train.jsonl"
        write(f, [
            {"run_id": "run_a", "prompt": "p", "completion": "c"},
            {"run_id": "run_b", "prompt": "p", "completion": "c"},
        ])
        with pytest.raises(ValueError, match="cross-run leakage"):
            MLXTrainer._assert_single_run(spec_for(f))

    def test_error_names_the_offending_line(self, tmp_path):
        f = tmp_path / "train.jsonl"
        write(f, [
            {"run_id": "run_a", "prompt": "p", "completion": "c"},
            {"run_id": "run_a", "prompt": "p", "completion": "c"},
            {"run_id": "run_z", "prompt": "p", "completion": "c"},
        ])
        with pytest.raises(ValueError, match=r":3 carries run_id 'run_z'"):
            MLXTrainer._assert_single_run(spec_for(f))

    def test_scans_every_file_in_a_directory(self, tmp_path):
        write(tmp_path / "train.jsonl", [{"run_id": "run_a"}])
        write(tmp_path / "valid.jsonl", [{"run_id": "run_b"}])
        with pytest.raises(ValueError, match="cross-run leakage"):
            MLXTrainer._assert_single_run(spec_for(tmp_path))

    def test_rows_without_run_id_are_permitted(self, tmp_path):
        """Untagged data cannot be proven foreign; the tag is the mechanism."""
        f = tmp_path / "train.jsonl"
        write(f, [{"prompt": "p", "completion": "c"}])
        MLXTrainer._assert_single_run(spec_for(f))

    def test_blank_lines_are_skipped(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_text('{"run_id": "run_a"}\n\n\n{"run_id": "run_a"}\n')
        MLXTrainer._assert_single_run(spec_for(f))
