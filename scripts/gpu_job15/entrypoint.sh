#!/usr/bin/env bash
# seahaven-fidelity across seven labs, three repeats each.
#
# The pipeline is validated: smoke test 4 passed preflight end to end against a
# real vLLM OpenAI endpoint (permutation 81.07 vs 61.38 shuffled, p=0.002).
# Three repeats per model so `reliability()` can be computed -- a single score
# per model says nothing about whether the ranking is stable.
#
# One server per model, started and reaped cleanly. Bounded startup wait that
# dumps the server log and moves on rather than hanging: a model that never comes
# up must cost its own slot and nothing else.
set -uo pipefail

R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000

push() { python /app/push.py "$1"; }
# reap_gpu() knows that vLLM's EngineCore children survive a kill of the parent.
# This job originally reaped only the api_server process and the first model left
# 124 GB held, so every subsequent server died on startup. The fix already
# existed in lib.sh and was not copied in.
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app

# Only the two the template-kwargs bug lost, plus TII which managed 2/3.
# The other four already have three clean repeats and re-running them would
# spend four model loads to reproduce numbers we hold.
MODELS=(
  "MistralAI|mistralai/Mistral-7B-Instruct-v0.3"
  "Google|google/gemma-2-9b-it"
  "TII|tiiuae/Falcon3-10B-Instruct"
)

for ENTRY in "${MODELS[@]}"; do
    LAB="${ENTRY%%|*}"; MODEL="${ENTRY##*|}"
    echo; echo "################ $LAB — $MODEL ################"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port $PORT --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm_$LAB.log 2>&1 &
    SERVER_PID=$!

    READY=0
    for i in $(seq 1 90); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            READY=1; echo "  server ready after ${i}0s"; break
        fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -25 /tmp/vllm_$LAB.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        for SEED in 5150 7301 9412; do
            python -m seahaven.fidelity.cli eval \
                --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
                --allow-regex-judge --runs 8 --steps 30 --seed $SEED \
                --output "$R/fid_${LAB}_${SEED}.json" 2>&1 | tail -12
            push "$R/fid_${LAB}_${SEED}.json" || true
        done
    else
        echo "  SKIPPED $LAB — server never became ready"
    fi

    # Reap. Killing the api_server is not enough: vLLM's EngineCore children
    # survive it and keep the device. reap_gpu verifies the memory actually came
    # back and aborts if it did not, rather than letting the next model fail in a
    # way that looks like the model's fault.
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.spawn" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN — aborting rather than blaming the next model"; exit 1; }
done

echo; echo "########## SUMMARY ##########"
python -c "
import json,glob,statistics as st
rows={}
for f in sorted(glob.glob('$R/fid_*.json')):
    d=json.load(open(f)); lab=f.split('fid_')[1].split('_')[0]
    ok=d.get('preflight',{}).get('ok'); s=d.get('score',{}).get('fidelity')
    rows.setdefault(lab,[]).append((ok,s))
print(f\"{'lab':<12}{'preflight':>12}{'fidelity per repeat':>34}\")
for lab,v in rows.items():
    oks=sum(1 for o,_ in v if o); vals=[s for o,s in v if o and s is not None]
    cell=' '.join(f'{x:.1f}' for x in vals) if vals else '-'
    print(f'{lab:<12}{oks}/{len(v):>11}{cell:>34}')
"
