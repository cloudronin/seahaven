#!/usr/bin/env bash
# Closed loop: each run plays campaign N+1 with campaign N's OWN weights loaded.
#
# Until now, rollouts used base + story and adapters affected only the final
# scoring -- so the spec's silent weight update never influenced subsequent
# behaviour. That removes the only path by which an update can compound:
#   updated weights -> different actions -> different corpus -> further update
#
# Play and train cannot share the GPU (vLLM's EngineCore child does not release
# it), so the job alternates one process per phase per campaign.
#
# Open-loop reference, measured: narrative spread 0 -> 0.071 -> 0.101 -> 0.102
# -> 0.080. It plateaued. If the feedback path matters, this should not.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=8
CAMPAIGNS=4

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

for C in $(seq 1 $CAMPAIGNS); do
    run_phase 20m "campaign $C — PLAY (own c$((C-1)) weights)" \
        python /app/closed_loop.py play --model "$MODEL" --campaign "$C" \
            --runs $RUNS --steps 40 --out "$R/closed_loop.json" || exit 1
    push "$R/closed_loop.json"

    if [ "$C" -lt "$CAMPAIGNS" ]; then
        run_phase 25m "campaign $C — TRAIN adapters" \
            python /app/train_step.py "$MODEL" --runs $RUNS --campaign "$C" \
                --beta 0.0 --tag cl || exit 1
    fi
done

run_phase 5m "report" \
    python /app/closed_loop.py report --model "$MODEL" --out "$R/closed_loop_report.json" || exit 1
push "$R/closed_loop_report.json"

echo
echo "########## REPORT ##########"
python -c "
import json; d=json.load(open('$R/closed_loop_report.json'))
print('closed loop narrative:', d['closed_loop_narrative'])
print('open loop reference  :', d['open_loop_narrative_reference'])
print()
for h in d['closed_loop']:
    print(f\"  c{h['campaign']} played_with={h['played_with']:8s} narr={h['narrative_spread']:.4f} \"
          f\"behav={h['behavioural_spread']:.4f} traj={h['trajectory_spread']:.4f} \"
          f\"distinct={h['distinct_command_sequences']} kept={h['kept_per_run']}\")
" 2>/dev/null || cat "$R/closed_loop_report.json"
