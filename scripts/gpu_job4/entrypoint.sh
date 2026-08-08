#!/usr/bin/env bash
# Can framing give the story authority over action without collapsing it into a
# memory device?
#
# generic  = the measured baseline (enacted between/within 1.07)
# efficacy = "it tells you what you did before, and what helped"  -> retrieval
# identity = "you act like that person"                            -> disposition
#
# Wanted outcome: separation WITHOUT convergence. Separation that decays over the
# trajectory is the memory device revealing itself, and competence converges.
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

run_phase 30m "framing: generic vs efficacy vs identity, 4 characters x 4 seeds" \
    python /app/framing.py --model "$MODEL" --seeds-per-character 4 --steps 20 \
        --out "$R/framing.json" || exit 1
push "$R/framing.json"

echo
echo "########## REPORT ##########"
python -c "
import json; d=json.load(open('$R/framing.json'))
s=d['summary']
print(f\"{'framing':10s} {'separation':>11s} {'convergence':>12s} {'reject':>8s}\")
for f,v in s['by_framing'].items():
    r=d['framings'][f]['parser_rejection_rate']
    print(f\"{f:10s} {str(v['separation']):>11s} {str(v['convergence']):>12s} {r:>8}\")
print()
print('best:', s['best_framing'], '| verdict:', s['verdict'])
print(s['interpretation'])
for f in d['framings']:
    print(f\"\n--- {f} ---\")
    for c,v in d['framings'][f]['per_character'].items():
        print(f'  {c:9s} {list(v[\"top\"])[:3]} rooms={v[\"rooms\"]}')
" 2>/dev/null || cat "$R/framing.json"
