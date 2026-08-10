#!/usr/bin/env bash
# B2 — the determinism control. The highest-leverage GPU dollars in the round:
# the P1 reuse check, the caching decision, and whether byte-identity means
# anything on this stack all take their shape from this output.
#
# **n=4 per config, not 2.** TRAP 32's lesson: at n=1 the wrapper looked like
# code drift, at n=2 the control agreed by drawing the same sample twice, and
# only at n=4 did both paths turn out nondeterministic and modally identical.
# Two draws manufacture false confidence in either direction.
#
# **Repeats run as whole evals, not lone rollouts.** The mechanism at issue is
# batch composition, and the sweep runs 12 episodes concurrently inside an eval.
# Four sequential single rollouts would never vary the batch and would report a
# determinism the sweep does not have.
#
# Protocol matches V-P exactly -- world_v0, p1, v1 schedule, --max-model-len
# 4096 -- so the answer applies to the study it gates rather than to a
# condition nobody runs.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export PYTHONPATH=/app PYTHONUNBUFFERED=1
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }

MODEL="meta-llama/Llama-3.1-8B-Instruct"   # a real subject, mid-size, not a toy

for CONFIG in default invariant; do
    echo; echo "################ config: $CONFIG ################"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    if [ "$CONFIG" = "invariant" ]; then
        export VLLM_BATCH_INVARIANT=1
        echo "  VLLM_BATCH_INVARIANT=1"
    else
        unset VLLM_BATCH_INVARIANT || true
        echo "  default flags only (what every sweep so far used)"
    fi

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port $PORT --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm_$CONFIG.log 2>&1 &
    SERVER_PID=$!
    READY=0
    for i in $(seq 1 90); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  server ready after ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm_$CONFIG.log; break; }
        [ $((i % 6)) -eq 0 ] && echo "  ...waiting ${i}0s"
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        # Four repeats of the SAME eval at the SAME seed. Deterministic serving
        # means all four are identical run-by-run.
        for REP in 1 2 3 4; do
            python -m seahaven.fidelity.cli eval \
                --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
                --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
                --seed 5150 --world world_v0 --phrasing p1 \
                --output "$R/b2_${CONFIG}_rep${REP}.json" > /tmp/b2_${CONFIG}_${REP}.log 2>&1
            echo "  rep $REP done"
        done
        for REP in 1 2 3 4; do push "$R/b2_${CONFIG}_rep${REP}.json" || true; done
    else
        echo "  SKIPPED $CONFIG"
    fi

    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN"; exit 1; }
done
echo; echo "########## DONE ##########"; ls -la "$R"
