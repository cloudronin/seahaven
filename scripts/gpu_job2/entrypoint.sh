#!/usr/bin/env bash
# Story-diversity gate. Generation only -- no training, no campaigns.
#
# Two runs in the story arm independently wrote the same character in different
# words ("a wanderer, a tinkerer, a seeker of forgotten things" / "a quiet
# observer, drawn to places that hold secrets"). If every run authors the same
# self, the story layer cannot be a source of divergence and nothing downstream
# of it can diverge either.
#
# The control is what makes it readable: each seed also writes a story having
# done nothing at all. Grounded vs ungrounded separates "experience shaped this"
# from "the model has a default self it returns to".
set -uo pipefail

R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"

push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== install ==="
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3

NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
if [ -n "${NVCC_DIR:-}" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/app
export VLLM_BATCH_INVARIANT=1

run_phase 20m "story diversity: grounded vs ungrounded, 12 seeds" \
    python /app/story_diversity.py --model "$MODEL" --seeds 12 --steps 16 \
        --out "$R/story_diversity.json" || exit 1
push "$R/story_diversity.json"

echo
echo "########## REPORT ##########"
cat "$R/story_diversity.json" 2>/dev/null || echo "(no report)"
