#!/usr/bin/env bash
# PHASE 1a BACKFILL — re-run only the cells whose push failed.
#
# Phase 1a swept all 18 exploration models cleanly, but `push.py` uploads ONE
# FILE PER COMMIT, so the sweep made ~540 commits against a single dataset
# branch and the commit endpoint began returning 500s. Twenty cells were lost.
# The evals themselves were fine — a cell is either a complete eval or absent,
# never corrupted — so this re-runs exactly the missing (model, phrasing, world,
# seed) combinations and nothing else.
#
# **The root cause is fixed here, not retried.** `push_batch.py` uses
# `upload_folder`, one commit per model instead of thirty. Phase 2 must inherit
# this: losing held-out cells would be far worse than losing exploration cells,
# because re-running a sealed set raises questions about what was seen and when.
#
# Same seal assertion, same vLLM pin, same firewall as Phase 1a. `missing.json`
# was computed by diffing the Hub against the seal's expected cell set.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
source /app/lib.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
VLLM_PIN=0.26.0
uv pip install --system --no-cache -q "vllm==${VLLM_PIN}" transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
export VLLM_BATCH_INVARIANT=1
python - <<PY || { echo "INSTALL FAILED or VERSION DRIFT"; exit 1; }
import vllm
print("  vllm", vllm.__version__)
assert vllm.__version__ == "${VLLM_PIN}", (
    f"vLLM {vllm.__version__} but the corpus was collected on ${VLLM_PIN}")
PY

python - <<'PY' || exit 1
import sys, json; sys.path.insert(0, "/app")
from seahaven.dimensional import seal as S
S.assert_sealed()
miss = json.load(open("/app/missing.json"))
repos = {m["repo"] for m in miss}
S.assert_not_held_out(repos)
print(f"  SEAL VERIFIED {S.SEAL_HASH[:16]}  firewall clean", flush=True)
print(f"  backfilling {len(miss)} cells across {len(repos)} models", flush=True)
PY

# **Read from a file baked in locally, not generated in-container.** The first
# attempt built this list with `mapfile < <(python - <<PY ...)`; the heredoc
# inside process substitution died with BrokenPipeError, mapfile got "0" and
# "39" instead of TAG|REPO|cells rows, and vLLM was handed an empty repo id.
# Shell input that a job depends on is generated on the host and shipped, the
# same way `missing.json` is.
mapfile -t GROUPS < /app/groups.txt


for ROW in "${GROUPS[@]}"; do
    IFS='|' read -r TAG REPO CELLS <<< "$ROW"
    N=$(echo "$CELLS" | tr ',' '\n' | wc -l | tr -d ' ')
    echo; echo "######## $TAG  $REPO  ($N cells) ########"
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
        PIDS=()
        for CELL in $(echo "$CELLS" | tr ',' ' '); do
            IFS=':' read -r PH WORLD SEED <<< "$CELL"
            (
              timeout 900 python -m seahaven.fidelity.cli eval \
                --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                --step-schedule v1 --seed "$SEED" --world "$WORLD" --phrasing "$PH" \
                --output "$R/dim_${TAG}_${PH}_${WORLD}_${SEED}.json" \
                > /tmp/b_${TAG}_${PH}_${WORLD}_${SEED}.log 2>&1
            ) &
            PIDS+=($!)
        done
        wait "${PIDS[@]}"
        echo "  $N cells done; pushing as ONE commit"
        python /app/push_batch.py "$R" "dim_${TAG}_*.json" || true
    else
        echo "  SKIPPED $TAG $REPO — server never ready"
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
done
echo; echo "########## BACKFILL DONE ##########"; ls "$R"/dim_*.json 2>/dev/null | wc -l
