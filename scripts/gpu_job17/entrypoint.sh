#!/usr/bin/env bash
# V1 rebuild: adjudicate the corrected label set with judges from other labs.
#
# **The question this job exists to answer.** An LLM used as the detector scored
# kappa 0.864 against GPT-5.2's labels, far above any regex. But the detector was
# gpt-4.1-mini and the labels were GPT-5.2 — same lab, so shared blind spots
# inflate agreement, and the number is agreement-with-OpenAI rather than
# accuracy. `adjudicate_v1.py` warns about exactly this in its own docstring.
#
# So the same 806 items are judged by two open models from two other labs. What
# matters is not each judge's agreement with the detector but **the agreement
# between judges**: if three labs converge, LLM judging is measuring something
# real; if OpenAI-to-OpenAI is much higher than cross-lab, the 0.864 was
# self-agreement and V1 is still unvalidated.
#
# Each judge carries the same eight planted controls and the same dual-ask
# self-consistency check, and refuses to emit labels below 87.5% control
# accuracy — a judge that fails those decides nothing regardless of its lab.
set -uo pipefail

R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
python -c "import vllm; print('  vllm', vllm.__version__)" || {
    echo "DEPENDENCY INSTALL FAILED"; exit 1; }

wc -l /app/results/v1b_labelset.csv

# Two labs, neither of them OpenAI. Both are instruction-tuned and large enough
# to be plausible judges of a one-sentence semantic question.
# Chosen for evidence rather than novelty. gemma-2-9b-it already ran this
# pipeline end to end in the V2 sweep, so the 27B sibling exercises a known-good
# template path (it rejects a system role; `endpoint.py` already folds the system
# turn into the first user turn for exactly this family). Qwen2.5-32B-Instruct is
# deliberately the non-reasoning Qwen: Qwen3 would spend a 6-token budget on
# thinking and return empty content, which `endpoint.py` refuses rather than
# scores — a known failure from TRAP 4.1.
# Phase B: three MORE labs, so that six judges split into two disjoint triples
# and the split-half reliability of a 3-judge majority becomes computable. That
# aggregate — not pairwise agreement between individuals — is the ceiling the
# gate must sit below (prereg-v3 sections 1-3).
#
# Already held: OpenAI (gpt-5.2), Alibaba (Qwen2.5-32B), Google (gemma-2-27b).
# Sixth judge. OLMo-2-32B failed its controls (0.75) by denying plainly stated
# room visits, so the panel is one short of the two disjoint triples the
# split-half needs. Order declared in the log BEFORE this run; the first to pass
# the controls is the sixth judge.
JUDGES=(
  "llama70|meta-llama/Llama-3.3-70B-Instruct"
  "granite8|ibm-granite/granite-3.1-8b-instruct"
)

for ENTRY in "${JUDGES[@]}"; do
    TAG="${ENTRY%%|*}"; MODEL="${ENTRY##*|}"
    # 70B in bf16 is ~140 GB against 143 GB of device — fp8 halves the weights
    # and leaves room for the KV cache.
    QUANT=""; [ "$TAG" = "llama70" ] && QUANT="--quantization fp8"
    echo; echo "################ judge $TAG — $MODEL ################"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port $PORT --host 127.0.0.1 \
        --gpu-memory-utilization 0.90 --max-model-len 4096 $QUANT \
        > /tmp/vllm_$TAG.log 2>&1 &
    SERVER_PID=$!

    READY=0
    for i in $(seq 1 120); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  server ready after ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm_$TAG.log; break; }
        [ $((i % 6)) -eq 0 ] && echo "  ...waiting ${i}0s: $(tail -1 /tmp/vllm_$TAG.log | cut -c1-90)"
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        python /app/scripts/adjudicate_v1.py \
            --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
            --labelset /app/results/v1b_labelset.csv \
            --out "$R/v1b_judge_${TAG}.json" --workers 16 2>&1 | tail -12
        push "$R/v1b_judge_${TAG}.json" || true
    else
        echo "  SKIPPED $TAG — server never became ready"
    fi

    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.spawn" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN"; exit 1; }
done

echo; echo "########## DONE ##########"
ls -la "$R"
