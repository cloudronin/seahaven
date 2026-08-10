#!/usr/bin/env bash
# PHASE 1a — sweep the EXPLORATION set. Held-out models are never loaded.
#
# **The firewall is structural, not documentary.** The model list is generated
# from `seahaven.dimensional.seal.EXPLORATION` at run time and routed through
# `assert_not_held_out` before a single server starts. There is no hardcoded
# copy of the list in this file that could drift from the seal, and no path by
# which a held-out repo reaches vLLM.
#
# The seal hash is asserted first. If the partition or its coverage
# justification has moved since `de730b7`, this job refuses to run — the
# question then is never "update the pin", it is what moved and whether the
# held-out set was loaded before it moved.
#
# 18 models x 30 cells (5 phrasings x 2 worlds x 3 seeds), narration OFF.
# 30 cells rather than fewer because the bend needs failure-bucket depth: the
# smoke test's thinnest after_fail bucket was 292 at 30 cells, and 6 cells would
# put low-failure models near 60.
#
# VLLM_BATCH_INVARIANT=1 throughout. Per TRAP 35 it binds above ~7B and not
# below, so the <=3B models here carry the measured floors in
# results/determinism_map.json; the flag is mandatory but no longer a claim that
# a given model reproduces.
#
# Output `dim_m<NN>_<phrasing>_world_<v>_<seed>.json`, where NN indexes the
# seal's EXPLORATION order. Reading the results therefore requires the seal,
# which enforces the firewall at analysis time as well as at sweep time.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
source /app/lib.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
export VLLM_BATCH_INVARIANT=1
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }

# ---- preflight: the seal, then the argument shapes, before any model loads
python - <<'PY' || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import seal as S
from seahaven.fidelity.runner import STEP_SCHEDULES, _steps_for, run_fidelity
import inspect

S.assert_sealed()
print(f"  SEAL VERIFIED {S.SEAL_HASH[:16]}  "
      f"exploration={len(S.EXPLORATION)} held-out={len(S.HELD_OUT)} (untouched)",
      flush=True)
S.assert_not_held_out(S.EXPLORATION)
print("  firewall: no held-out repo in the sweep list", flush=True)

assert inspect.signature(run_fidelity).parameters["narrate"].default is True
sched = STEP_SCHEDULES["v1"]
assert 12 == len(sched), f"runs=12 against a {len(sched)}-entry schedule"
lens = [_steps_for(i, 30, sched) for i in range(12)]
print(f"  sweep: 12 runs, steps {min(lens)}-{max(lens)}, {sum(lens)} commands max",
      flush=True)
PY

mapfile -t MODELS < <(python - <<'PY'
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import seal as S
S.assert_sealed()
for i, m in enumerate(S.EXPLORATION):
    print(f"m{i:02d}|{m}")
PY
)
echo "  ${#MODELS[@]} exploration models to sweep"

for ROW in "${MODELS[@]}"; do
    TAG="${ROW%%|*}"; REPO="${ROW##*|}"
    echo; echo "######## $TAG  $REPO ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    SERVER_PID=$!
    READY=0
    for i in $(seq 1 60); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -8 /tmp/vllm.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        PIDS=(); N=0
        for PH in p1 p2 p3 p4 p5; do
          for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
              (
                timeout 900 python -m seahaven.fidelity.cli eval \
                  --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                  --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                  --step-schedule v1 --seed $SEED --world "$WORLD" --phrasing "$PH" \
                  --output "$R/dim_${TAG}_${PH}_${WORLD}_${SEED}.json" \
                  > /tmp/e_${TAG}_${PH}_${WORLD}_${SEED}.log 2>&1
              ) &
              PIDS+=($!); N=$((N+1))
              if [ $((N % 10)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
            done
          done
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        NC=$(python - <<PY
import json, glob
n=0
for f in glob.glob("$R/dim_${TAG}_*.json"):
    try: n += sum(len(r.get("commands",[])) for r in json.load(open(f))["runs"])
    except Exception: pass
print(n)
PY
)
        echo "  30 evals done, $NC commands; pushing"
        if [ "$NC" -lt 1000 ]; then
            echo "  WARNING $TAG produced only $NC commands — recorded, not skipped"
        fi
        # Serially: concurrent pushes raced the dataset branch once and lost
        # four of Google's six uploads to HTTP 500.
        for f in "$R"/dim_${TAG}_*.json; do python /app/push.py "$f" || true; done
    else
        echo "  SKIPPED $TAG $REPO — server never ready"
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
done

echo; echo "########## PHASE 1a DONE ##########"
ls "$R"/dim_*.json 2>/dev/null | wc -l
