#!/usr/bin/env bash
# Eight generations. Each plays on the adapter trained from the previous
# generation's own selected episodes.
#
# Play injects variance; distillation contracts. Two generations cannot say
# which dominates -- the closed-loop run managed only two and was flat enough to
# be noise. Eight separates: decay to zero (contraction mapping, claim dead),
# a non-zero plateau (equilibrium, bounded character), or growth (drift wins,
# which is path dependence).
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=8
GENS=8

push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3
NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
[ -n "${NVCC_DIR:-}" ] && export CUDA_HOME="$NVCC_DIR" PATH="$NVCC_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app VLLM_BATCH_INVARIANT=1

for G in $(seq 1 $GENS); do
    run_phase 15m "gen $G — PLAY" \
        python /app/multigen.py --model "$MODEL" --gen "$G" --runs $RUNS --steps 30 \
            --out "$R/multigen.json" || exit 1
    push "$R/multigen.json"

    if [ "$G" -lt "$GENS" ]; then
        run_phase 20m "gen $G — TRAIN" \
            python /app/train_step.py "$MODEL" --runs $RUNS --campaign "$G" \
                --beta 0.0 --tag mg || exit 1
    fi
done

echo
echo "########## REPORT ##########"
python -c "
import json; h=json.load(open('$R/multigen.json'))
print(f\"{'gen':>3s} {'own wts':>8s} {'narrative':>10s} {'behav':>9s} {'traj':>9s} {'distinct':>9s}\")
for r in h:
    print(f\"{r['generation']:>3d} {r['runs_on_own_weights']:>8d} {r['narrative_spread']:>10.4f} \"
          f\"{r['behavioural_spread']:>9.4f} {r['trajectory_spread']:>9.4f} {r['distinct_sequences']:>9s}\")
print()
print('narrative :', [r['narrative_spread'] for r in h])
print('behaviour :', [r['behavioural_spread'] for r in h])
print('trajectory:', [r['trajectory_spread'] for r in h])
"
