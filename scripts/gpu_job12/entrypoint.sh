#!/usr/bin/env bash
# Donor-narrative control on the lagged coupling.
#
# The +0.412 partial cannot distinguish "my own self-account steers me" from "any
# movement-heavy prompt text raises my go-rate". Both predict the same
# correlation, because the narrative enters the next campaign's system prompt.
#
# Warm up 3 campaigns so narratives develop naturally, then play ONE more
# campaign twice from the same state: once with each run's own narrative, once
# with another run's narrative matched on movement-vocabulary density. Matching
# on density is what makes this a test of authorship rather than of content.
#
# own >> donor  -> self-authorship matters, the result holds
# own ~= donor  -> prompt content, not authorship; PLAN.md kill criterion 1 fires
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
OUT="$R/donor.json"

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
        python /app/donor.py --model "$MODEL" --lab "$LAB" \
            --runs 8 --steps 30 --warmup 3 --out "$OUT" || true
    push "$OUT"
done

echo
echo "########## REPORT ##########"
python -c "
import json
h=json.load(open('$OUT'))
print(f\"{'lab':<12}{'own go':>9}{'donor go':>10}{'own dens':>10}{'donor dens':>12}\")
for r in h:
    if r['status']!='ok':
        print(f\"{r['lab']:<12} {r['status']}\"); continue
    og=sum(r['own_go'])/len(r['own_go']); dg=sum(r['donor_go'])/len(r['donor_go'])
    od=sum(r['own_density'])/len(r['own_density']); dd=sum(r['donor_density'])/len(r['donor_density'])
    print(f\"{r['lab']:<12}{og:>9.3f}{dg:>10.3f}{od:>10.2f}{dd:>12.2f}\")
"
