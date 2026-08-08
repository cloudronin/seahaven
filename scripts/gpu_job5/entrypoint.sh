#!/usr/bin/env bash
# Does distillation preserve character separation, or erase it?
#
# The distilled component has been negative twice. Masking halved it without
# flipping the sign, and the leading explanation is that the corpora were
# near-identical (12-19 examples, same handful of commands). Identity framing now
# makes assigned characters behave measurably differently (separation 2.44 vs a
# 1.13 baseline), so corpora that are distinct BY CONSTRUCTION are available.
#
# If distillation erases separation even here, it is a real convergence pressure
# and not a data artefact.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
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

run_phase 30m "1/4 collect both arms under identity framing" \
    python /app/distill_framed.py collect --model "$MODEL" \
        --seeds-per-character 2 --emergent-runs 8 --steps 30 --campaigns 2 \
        --out "$R/df_collect.json" || exit 1
push "$R/df_collect.json"

run_phase 25m "2/4 train assigned adapters (masked)" \
    python /app/train_many.py "$MODEL" 8 --mask --prefix assigned_ || exit 1

run_phase 25m "3/4 train emergent adapters (masked)" \
    python /app/train_many.py "$MODEL" 8 --mask --prefix emergent_ || exit 1

run_phase 15m "4/4 score before vs after, both arms" \
    python /app/distill_framed.py score --model "$MODEL" \
        --collect "$R/df_collect.json" --out "$R/df_result.json" || exit 1
push "$R/df_result.json"

echo
echo "########## REPORT ##########"
python -c "
import json; d=json.load(open('$R/df_result.json'))
s=d['summary']
print(json.dumps(s, indent=2))
for arm,v in d['arms'].items():
    if 'before' in v:
        print(f\"\n{arm}: b/w {v['before']['ratio']} -> {v['after']['ratio']} (x{v['separation_amplification']})\")
        print(f\"  spread {v['all_pairs_before']} -> {v['all_pairs_after']}  distilled {v['distilled_component']:+}\")
" 2>/dev/null || cat "$R/df_result.json"
