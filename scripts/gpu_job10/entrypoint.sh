#!/usr/bin/env bash
# Eight generations again, with selection forced to keep at least one episode.
#
# Three of eight runs froze permanently last time: a run that keeps nothing does
# not update, so it produces the same thing, so it keeps nothing again. An
# absorbing state that swallowed half the population within six generations.
#
# Two-pass selection. Pass 1 is unconstrained and diagnostic -- it preserves the
# spec's `selected_nothing` signal AND finally distinguishes a genuine refusal
# from a parse failure or out-of-range indices, which the previous code silently
# conflated. Pass 2 forces >= 1 and is what training uses: only the "whether" is
# constrained, the "which" is still the agent's.
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
    run_phase 15m "gen $G — PLAY + forced selection" \
        python /app/multigen.py --model "$MODEL" --gen "$G" --runs $RUNS --steps 30 \
            --out "$R/multigen_forced.json" || exit 1
    push "$R/multigen_forced.json"

    if [ "$G" -lt "$GENS" ]; then
        run_phase 20m "gen $G — TRAIN" \
            python /app/train_step.py "$MODEL" --runs $RUNS --campaign "$G" \
                --beta 0.0 --tag mg || exit 1
    fi
done

echo
echo "########## REPORT ##########"
python -c "
import json; h=json.load(open('$R/multigen_forced.json'))
print(f\"{'gen':>3s} {'narr':>8s} {'behav':>8s} {'traj':>8s} {'distinct':>8s}  forced-kept | unconstrained\")
for r in h:
    print(f\"{r['generation']:>3d} {r['narrative_spread']:>8.4f} {r['behavioural_spread']:>8.4f} \"
          f\"{r['trajectory_spread']:>8.4f} {r['distinct_sequences']:>8s}  {r['kept_per_run']} | {r['unconstrained_kept']}\")
print()
print('unconstrained reason codes per gen:')
for r in h: print(' ', r['generation'], r['unconstrained_reasons'])
print()
print('narrative :', [r['narrative_spread'] for r in h])
print('behaviour :', [r['behavioural_spread'] for r in h])
print('trajectory:', [r['trajectory_spread'] for r in h])
"
