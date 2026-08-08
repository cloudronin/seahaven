#!/usr/bin/env bash
# Manipulation check: can an assigned story move behaviour at all?
#
# Everything else asks whether stories diverge. This asks the prior question --
# whether the story channel carries any influence over behaviour whatsoever. If
# deliberately opposed characters do not separate behaviour, narrative
# divergence cannot matter however large it gets.
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

run_phase 25m "assigned characters: explicit + implicit, stated + enacted" \
    python /app/assigned.py --model "$MODEL" --seeds-per-character 4 --steps 14 \
        --out "$R/assigned.json" || exit 1
push "$R/assigned.json"

echo
echo "########## REPORT ##########"
python -c "
import json; d=json.load(open('$R/assigned.json'))
print(json.dumps(d['summary'], indent=2))
for tier in ('explicit','implicit'):
    t=d['tiers'][tier]
    print(f\"\n--- {tier} ---\")
    print('  stated :', t['stated'])
    print('  enacted:', t['enacted'])
    for c,v in t['per_character'].items():
        print(f\"    {c:9s} {list(v['top_commands'])[:3]} rooms={len(v['rooms'])}\")
" 2>/dev/null || cat "$R/assigned.json"
