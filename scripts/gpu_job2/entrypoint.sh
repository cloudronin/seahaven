#!/usr/bin/env bash
# Four campaigns, n=8. Answers the three follow-ups from the amnesia run:
#   1. does the fixed rewrite prompt stop verbatim copying (measured, not assumed)
#   2. does divergence accumulate over four campaigns
#   3. does narrative divergence reach behaviour, at matched n
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=8

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

run_phase 25m "1/3 collect: 2 arms x 8 runs x 4 campaigns" \
    python /app/accumulate.py collect --model "$MODEL" \
        --runs $RUNS --steps 14 --campaigns 4 --out "$R/acc_collect.json" || exit 1
push "$R/acc_collect.json"

run_phase 30m "2/3 train 8 adapters, one per run" \
    python /app/train_many.py "$MODEL" $RUNS || exit 1

run_phase 15m "3/3 score behaviour at n=8: story-only vs trained" \
    python /app/accumulate.py score --model "$MODEL" \
        --collect "$R/acc_collect.json" --out "$R/acc_behaviour.json" || exit 1
push "$R/acc_behaviour.json"

echo
echo "########## REPORT ##########"
python - <<'PY'
import json
c=json.load(open("/tmp/results/acc_collect.json"))
print("narrative spread by campaign:")
for arm,d in c["arms"].items():
    print(f"  {arm:9s}", [s["narrative_spread"] for s in d["stages"]])
    print(f"  {'':9s} verbatim", [s["verbatim_overlap"] for s in d["stages"]])
try:
    b=json.load(open("/tmp/results/acc_behaviour.json"))
    print("\nbehaviour:", json.dumps(b, indent=2))
except Exception as e:
    print("no behaviour report:", e)
PY
