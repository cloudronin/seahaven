#!/usr/bin/env bash
# Dump raw per-run trajectories AND probe fingerprints for the same runs, so the
# battery-vs-behaviour question can be analysed locally and iterated for free.
#
# Assigned characters under identity framing: the one configuration where
# behavioural separation is known real (2.44 vs a 1.13 baseline).
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3
NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
[ -n "${NVCC_DIR:-}" ] && export CUDA_HOME="$NVCC_DIR" PATH="$NVCC_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app VLLM_BATCH_INVARIANT=1

run_phase 25m "dump 12 runs x 50 steps + probe fingerprints" \
    python /app/dump_data.py --model "$MODEL" --seeds-per-character 3 --steps 50 \
        --out "$R/raw_dump.json" || exit 1
push "$R/raw_dump.json"
echo "size: $(du -h "$R/raw_dump.json" | cut -f1)"
