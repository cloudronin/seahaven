#!/usr/bin/env bash
# Test-retest for the fidelity score: 3 repeats per checkpoint, 7 checkpoints.
#
# A per-model benchmark number is only meaningful if it moves less between
# repeats of the SAME model than between different models. TRAP 13 is the
# precedent: within-model noise 0.147 against between-model signal 0.128, which
# could not rank anything, discovered only after eight generations of work.
#
# Generation only. Judging and scoring run locally afterwards so the judge can be
# swapped or re-run without paying for GPU time again.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
OUT="$R/fidelity_repeats.json"

push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm transformers accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3
NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
[ -n "${NVCC_DIR:-}" ] && export CUDA_HOME="$NVCC_DIR" PATH="$NVCC_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app VLLM_BATCH_INVARIANT=1

MODELS=(
  "Alibaba|Qwen/Qwen3-8B"
  "MistralAI|mistralai/Mistral-7B-Instruct-v0.3"
  "AI2|allenai/OLMo-2-1124-13B-Instruct"
  "IBM|ibm-granite/granite-3.1-8b-instruct"
  "TII|tiiuae/Falcon3-10B-Instruct"
  "Meta|meta-llama/Llama-3.1-8B-Instruct"
  "Google|google/gemma-2-9b-it"
)

for ENTRY in "${MODELS[@]}"; do
    LAB="${ENTRY%%|*}"; MODEL="${ENTRY##*|}"
    run_phase 30m "$LAB — $MODEL" \
        python /app/fidelity_repeats.py --model "$MODEL" --lab "$LAB" \
            --runs 8 --steps 30 --repeats 3 --out "$OUT" || true
    push "$OUT"
done

echo
echo "########## REPORT ##########"
python -c "
import json
h=json.load(open('$OUT'))
print(f\"{'lab':<12}{'status':<12}{'repeats':>9}{'runs that acted':>18}\")
for r in h:
    if r['status']!='ok':
        print(f\"{r['lab']:<12}{r['status']:<12}\"); continue
    acted=[sum(1 for x in rep['runs'] if any(x['performed'].values())) for rep in r['repeats']]
    print(f\"{r['lab']:<12}{'ok':<12}{len(r['repeats']):>9}{str(acted):>18}\")
"
