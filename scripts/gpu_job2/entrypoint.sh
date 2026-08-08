#!/usr/bin/env bash
# The story arm. The no-story floor is already measured: two seeds, no story
# artifact, d(A,B) = 0.015352. `world_v0` never had a story, so that run WAS the
# spec's baseline — "same as across-seed, story layer removed" — not an
# unanchored numerator.
#
# This runs the other side of the ratio: campaign 1 → the agent writes about
# itself → campaign 2 carries that story → train → measure with the story
# present. The result is decomposed into the part explained by the story sitting
# in context and the part carried in the weights, because only the second is the
# mechanism the spec claims.
set -uo pipefail

R=/tmp/results
W=/tmp/a1b
mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
NO_STORY_FLOOR=0.015352

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

run_phase 20m "1/4 two campaigns per run, story written between them" \
    python /app/story_arm.py collect --model "$MODEL" \
        --episodes 10 --steps 20 --out "$R/story_corpora.json" || exit 1
push "$R/story_corpora.json"

run_phase 10m "2/4 train adapter A (story arm)" \
    python /app/a1b_stage.py train --model "$MODEL" --tag A --out "$R/s_trainA.json" || exit 1

run_phase 10m "3/4 train adapter B (story arm)" \
    python /app/a1b_stage.py train --model "$MODEL" --tag B --out "$R/s_trainB.json" || exit 1

run_phase 12m "4/4 score: story-in-prompt at base vs trained, against the floor" \
    python /app/story_arm.py score --model "$MODEL" \
        --floor "$NO_STORY_FLOOR" --out "$R/story_arm.json" || exit 1
push "$R/story_arm.json"

echo
echo "########## REPORT ##########"
cat "$R/story_arm.json" 2>/dev/null || echo "(no report)"
