"""MLX LoRA trainer.

Two mlx-lm defaults are wrong for this experiment and are overridden explicitly.

**`mask_prompt` defaults to False.** Training data here is
`{prompt: <system + observation>, completion: <the action>}`. Without prompt
masking the loss also covers the prompt, so the model is trained to predict the
*world's* text as well as its own action — it learns to be the environment as
much as to act in it. For a self-imitation loop whose whole premise is that the
agent is distilling its own dispositions, that is a confound, not a detail.

**`lora_parameters.rank` defaults to 8 and is not a CLI flag.** Rank is a
hyperparameter that has to land in the config hash, so it is written into an
explicit config file rather than inherited silently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from .base import Trainer
from .types import AdapterRef, TrainSpec

__all__ = ["MLXTrainer"]

DEFAULT_PY = Path.home() / "miniconda3/envs/seahaven-mlx/bin/python"


class MLXTrainer(Trainer):
    def __init__(self, python: Path = DEFAULT_PY) -> None:
        self.python = python

    def train_adapter(self, spec: TrainSpec) -> AdapterRef:
        self._assert_single_run(spec)

        out = Path(spec.out_dir)
        partial = out.with_name(out.name + ".partial")
        if partial.exists():
            import shutil

            shutil.rmtree(partial)
        partial.mkdir(parents=True)

        config = partial / "lora_config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "lora_parameters": {
                        "rank": spec.rank,
                        "dropout": 0.0,
                        "scale": float(spec.alpha),
                    }
                },
                sort_keys=True,
            )
        )

        cmd = [
            str(self.python), "-m", "mlx_lm", "lora",
            "--model", spec.base_model,
            "--train",
            "--data", spec.train_jsonl,
            "--adapter-path", str(partial),
            "--config", str(config),
            "--iters", str(spec.iters),
            "--batch-size", str(spec.batch_size),
            "--num-layers", str(spec.layers),
            "--learning-rate", str(spec.learning_rate),
            "--seed", str(spec.seed),
            "--mask-prompt",
            "--max-seq-length", "1024",
            "--steps-per-report", "25",
            "--steps-per-eval", str(max(50, spec.iters // 4)),
        ]
        if spec.resume_from is not None:
            cmd += ["--resume-adapter-file",
                    str(Path(spec.resume_from.path) / "adapters.safetensors")]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        (partial / "train.log").write_text(
            proc.stdout + "\n--- stderr ---\n" + proc.stderr
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "mlx_lm lora failed:\n"
                + proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]
            )

        weights = partial / "adapters.safetensors"
        if not weights.exists():
            raise RuntimeError(f"training produced no adapter at {weights}")

        # Promote only on success, so a crashed run never leaves a directory
        # that looks like a finished adapter.
        if out.exists():
            import shutil

            shutil.rmtree(out)
        partial.rename(out)

        digest = hashlib.sha256((out / "adapters.safetensors").read_bytes()).hexdigest()
        return AdapterRef(
            name=spec.adapter_name,
            path=str(out),
            sha256=digest,
            base_model=spec.base_model,
            trained_from=spec.resume_from.sha256 if spec.resume_from else None,
        )

    @staticmethod
    def _assert_single_run(spec: TrainSpec) -> None:
        """No cross-seed pooling. Cheapest possible guard on the most
        catastrophic possible bug: one run's trajectories reaching another run's
        adapter would destroy the isolation the divergence claim rests on."""
        path = Path(spec.train_jsonl)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            for lineno, line in enumerate(file.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "run_id" in row and row["run_id"] != spec.run_id:
                    raise ValueError(
                        f"{file}:{lineno} carries run_id {row['run_id']!r}, "
                        f"expected {spec.run_id!r} — cross-run leakage"
                    )
