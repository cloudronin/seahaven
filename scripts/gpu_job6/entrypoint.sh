#!/usr/bin/env bash
# Does a KL term to the run's OWN previous checkpoint stop distillation from
# erasing character?
#
# Distillation has erased character in every configuration: -8% on corpora
# distinct by construction, -22% on self-authored ones, surviving both prompt
# masking and identity framing. Hard-target self-imitation discards tails.
#
# Reference choice is the whole point. Anchoring to the shared base would pull
# every run to the same place. Anchoring each run to its OWN previous campaign
# resists collapse without pulling runs together -- so training must be CHAINED,
# and a single-campaign test would have been actively misleading.
#
# One corpus, two chains: plain SFT and KL. Only the KL term differs.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=6
CAMPAIGNS=2

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

run_phase 30m "1/4 collect $CAMPAIGNS campaigns x $RUNS runs (one corpus, both variants)" \
    python /app/kl_stages.py collect --model "$MODEL" \
        --runs $RUNS --steps 30 --campaigns $CAMPAIGNS --out "$R/kl_collect.json" || exit 1
push "$R/kl_collect.json"

run_phase 30m "2/4 train chain: plain SFT (beta=0)" \
    python /app/kl_train.py "$MODEL" --runs $RUNS --campaigns $CAMPAIGNS \
        --beta 0.0 --tag sft || exit 1

run_phase 30m "3/4 train chain: KL to own previous checkpoint (beta=0.5)" \
    python /app/kl_train.py "$MODEL" --runs $RUNS --campaigns $CAMPAIGNS \
        --beta 0.5 --tag kl || exit 1

run_phase 15m "4/4 score both chains" \
    python /app/kl_stages.py score --model "$MODEL" --campaigns $CAMPAIGNS \
        --collect "$R/kl_collect.json" --out "$R/kl_result.json" || exit 1
push "$R/kl_result.json"

echo
echo "########## REPORT ##########"
cat "$R/kl_result.json"
