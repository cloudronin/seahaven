#!/usr/bin/env bash
# Does a KL term to the run's OWN previous checkpoint stop distillation erasing
# character? Second attempt, with three fixes over the first.
#
#  1. Campaign 1 is plain SFT for BOTH variants and trained ONCE, then shared.
#     The first attempt applied KL from campaign 1, where the only reference is
#     the shared base -- so half the training did exactly the thing the design
#     exists to avoid. KL now applies only where the reference is run-specific.
#  2. A run that selects nothing carries its previous adapter forward instead of
#     dropping out. Last time that cut the sample from 6 runs to 3.
#  3. More runs and more campaigns, so run-specific anchoring has room to act.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=8
CAMPAIGNS=3

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

run_phase 30m "1/5 collect $CAMPAIGNS campaigns x $RUNS runs (40 steps each)" \
    python /app/kl_stages.py collect --model "$MODEL" \
        --runs $RUNS --steps 40 --campaigns $CAMPAIGNS --out "$R/kl_collect.json" || exit 1
push "$R/kl_collect.json"

run_phase 25m "2/5 shared campaign 1 (plain SFT, both variants start here)" \
    python /app/kl_train.py "$MODEL" --runs $RUNS --campaigns 1 \
        --beta 0.0 --tag shared || exit 1

run_phase 30m "3/5 chain c2..c$CAMPAIGNS — plain SFT" \
    python /app/kl_train.py "$MODEL" --runs $RUNS --campaigns $CAMPAIGNS \
        --start 2 --init-tag shared --beta 0.0 --tag sft || exit 1

run_phase 35m "4/5 chain c2..c$CAMPAIGNS — KL to own previous campaign" \
    python /app/kl_train.py "$MODEL" --runs $RUNS --campaigns $CAMPAIGNS \
        --start 2 --init-tag shared --beta 0.5 --tag kl || exit 1

run_phase 15m "5/5 score both chains on the common subset" \
    python /app/kl_stages.py score --model "$MODEL" --campaigns $CAMPAIGNS \
        --collect "$R/kl_collect.json" --out "$R/kl_result.json" || exit 1
push "$R/kl_result.json"

echo
echo "########## REPORT ##########"
cat "$R/kl_result.json"
