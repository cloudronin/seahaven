#!/usr/bin/env bash
# Divergence smoke test: do two differently-seeded runs move APART, or to the
# same place? The spec lists "convergent attractor" as an explicit null and
# nothing measured so far can rule it out, because every result to date concerns
# a single run.
#
# Two engine loads rather than four: both corpora come from one process, and
# base + both adapters are scored in another.
set -uo pipefail

R=/tmp/results
W=/tmp/a1b
mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"

push() { python /app/push.py "$1"; }
source /app/lib.sh

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

echo "=== install ==="
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -4

NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
if [ -n "${NVCC_DIR:-}" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/app
export VLLM_BATCH_INVARIANT=1

python -c "import vllm, torch, jericho; print('vllm', vllm.__version__)"

run_phase 14m "1/4 collect both corpora (seeds 101 / 202)" \
    python /app/divergence.py collect2 --model "$MODEL" \
        --episodes 12 --steps 24 --out "$R/div_corpora.json" || exit 1
push "$R/div_corpora.json"

run_phase 10m "2/4 train adapter A" \
    python /app/a1b_stage.py train --model "$MODEL" --tag A --out "$R/trainA.json" || exit 1

run_phase 10m "3/4 train adapter B" \
    python /app/a1b_stage.py train --model "$MODEL" --tag B --out "$R/trainB.json" || exit 1

run_phase 12m "4/4 score base + A + B, compute the three distances" \
    python /app/divergence.py score_all --model "$MODEL" --out "$R/divergence.json" || exit 1
push "$R/divergence.json"

echo
echo "########## REPORT ##########"
cat "$R/divergence.json" 2>/dev/null || echo "(no divergence report)"
