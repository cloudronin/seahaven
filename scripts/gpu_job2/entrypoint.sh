#!/usr/bin/env bash
# Identical amnesiac seed story: do narratives fan out from a common blank self,
# or fall back to a default the model returns to regardless of history?
#
# Generation only. The seeded arm starts at narrative distance exactly zero by
# construction, so the result is the shape of the curve away from it.
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

run_phase 25m "amnesia: seeded vs emergent, 8 runs x 2 campaigns" \
    python /app/amnesia.py --model "$MODEL" --runs 8 --steps 16 --campaigns 2 \
        --out "$R/amnesia.json" || exit 1
push "$R/amnesia.json"

echo
echo "########## REPORT ##########"
python -c "
import json
d=json.load(open('$R/amnesia.json'))
print(json.dumps(d['summary'], indent=2))
print()
for arm in ('seeded','emergent'):
    print('---', arm, 'final stories ---')
    for i,s in enumerate(d['arms'][arm]['final_stories'][:4]):
        print(f'  [{i}] {s[:200]}')
" 2>/dev/null || cat "$R/amnesia.json"
