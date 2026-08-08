#!/usr/bin/env bash
# Do models from different labs converge on DIFFERENT characters?
#
# Self-authored narratives converge (induced convergence 0.583 vs 0.050 for four
# contrasting personas). Every Qwen3-8B run in this world writes a version of the
# same patient, quiet observer. Is that character a property of the model, or of
# any LLM dropped into this world?
#
# Hold world, seed story, framing and step budget constant; vary only the
# checkpoint. High within-model convergence + LOW pooled convergence means
# per-lab attractors. High both means the world picks the character.
#
# One model per phase, each with a smoke check and its own timeout: Olmo-3 once
# produced no output for 25 minutes and took the whole cross-family arm with it.
# A checkpoint that fails here costs its own slot and nothing else.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
OUT="$R/crosslab.json"

push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm transformers accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3
NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
[ -n "${NVCC_DIR:-}" ] && export CUDA_HOME="$NVCC_DIR" PATH="$NVCC_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app VLLM_BATCH_INVARIANT=1

# lab|model. Seven labs. Llama-3.1 and Gemma-2 are gated=manual and included
# because the account already holds their licenses -- verified by fetching
# config.json with the token, not by assuming.
#
# All seven are INSTRUCT checkpoints, deliberately. Base variants of Llama and
# Gemma are also accessible, but mixing them in would confound "different lab"
# with "different post-training stage" -- and TRAP 4.2 measured what happens when
# a base checkpoint meets a chat template: clean rate 0.06 against 0.88.
#
# Every template was pre-flighted locally before launch. Six accept a system
# role; Gemma-2 raises TemplateError on it and takes the merge fallback in
# crosslab.chat(), which preserves the identity framing rather than dropping it.
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
    # `|| true`: one dead checkpoint must not end the sweep. crosslab.py records
    # its own failure row, so the report distinguishes "failed" from "not asked".
    run_phase 25m "$LAB — $MODEL" \
        python /app/crosslab.py --model "$MODEL" --lab "$LAB" \
            --runs 8 --steps 30 --campaigns 2 --out "$OUT" || true
    push "$OUT"
done

echo
echo "########## REPORT ##########"
python -c "
import json
h=json.load(open('$OUT'))
print(f\"{'lab':<12}{'model':<40}{'status':<14}{'distinct':>9}{'parse_ok':>10}\")
for r in h:
    if r['status']!='ok':
        print(f\"{r['lab']:<12}{r['model']:<40}{r['status']:<14}{'-':>9}{'-':>10}  {r.get('error','')[:50]}\")
    else:
        c=r['campaigns'][-1]
        print(f\"{r['lab']:<12}{r['model']:<40}{'ok':<14}{str(c['distinct_sequences'])+'/8':>9}{c['parse_ok_rate']:>10.3f}\")
print()
for r in h:
    if r['status']!='ok': continue
    print(f\"===== {r['lab']} ({r['model']}) — final narratives\")
    for i,s in enumerate(r['campaigns'][-1]['narratives'][:3]):
        print(f'  [{i}] {s[:300]}')
    print()
"
