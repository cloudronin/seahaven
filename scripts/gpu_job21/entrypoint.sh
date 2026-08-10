#!/usr/bin/env bash
# STUDY 1 / Phase C -- V-P constraint-phrasing invariance. The existential gate.
#
# Adherence has never been tested on the axis that killed fidelity: varying the
# STIMULUS rather than the measurement. Fidelity's narration-register sweep put
# worst-pair Spearman at 0.607 and V3 failed. If G-P fails here, world_v3 is
# never authored and Stage 1 and Stage 2 never run.
#
# **VLLM_BATCH_INVARIANT=1 is set.** B2 measured this stack as nondeterministic
# without it -- 8/12 runs reproducible across 4 repeats, 34% raw command
# divergence -- and fully deterministic with it, 12/12 and 0.000%. The flag
# binds on every run from here forward.
#
# **P1 is re-run, not reused.** The existing corpus was generated bare, so
# reusing its cells would put one fifth of the design on a different serving
# configuration than the other four fifths -- a second factor inside a study
# whose entire purpose is single-factor attribution. It also lacks per-step
# command records, without which coverage and C-MIMIC cannot be computed.
# 210 evals, not 168.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
export VLLM_BATCH_INVARIANT=1            # per B2; see header
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }
python - <<'PY' || exit 1
import sys; sys.path.insert(0,"/app")
from seahaven.fidelity.runner import PHRASINGS
from seahaven.fidelity.worldspec import load
assert sorted(PHRASINGS) == ["p1","p2","p3","p4","p5"], PHRASINGS
for w in ("world_v0","world_v2"):
    print(f"  {w}: {len(load(w).entity_keys)} keys", flush=True)
print(f"  phrasings: {sorted(PHRASINGS)}", flush=True)
PY

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
    echo; echo "################ $LAB — $MODEL ################"
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port $PORT --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm_$LAB.log 2>&1 &
    SERVER_PID=$!
    READY=0
    for i in $(seq 1 90); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  server ready after ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm_$LAB.log; break; }
        [ $((i % 6)) -eq 0 ] && echo "  ...waiting ${i}0s: $(tail -1 /tmp/vllm_$LAB.log | cut -c1-80)"
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        # 30 evals per model, in batches of 10. Each eval is already 12-way
        # concurrent inside, so 10 evals is 120 sequences -- well inside the KV
        # budget at 4096 for these model sizes.
        PIDS=(); N=0
        for PH in p1 p2 p3 p4 p5; do
          for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
              (
                python -m seahaven.fidelity.cli eval \
                  --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
                  --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
                  --seed $SEED --world "$WORLD" --phrasing "$PH" \
                  --output "$R/vp_${LAB}_${PH}_${WORLD}_${SEED}.json" \
                  > /tmp/vp_${LAB}_${PH}_${WORLD}_${SEED}.log 2>&1
              ) &
              PIDS+=($!); N=$((N+1))
              if [ $((N % 10)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
            done
          done
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        echo "  30 evals done; pushing"
        for PH in p1 p2 p3 p4 p5; do
          for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
              push "$R/vp_${LAB}_${PH}_${WORLD}_${SEED}.json" || true
            done
          done
        done
    else
        echo "  SKIPPED $LAB — server never became ready"
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN"; exit 1; }
done
echo; echo "########## DONE ##########"; ls "$R" | wc -l
