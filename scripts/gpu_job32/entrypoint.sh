#!/usr/bin/env bash
# PHASE 1a BACKFILL, attempt 6 — and the first one aimed at the real defect.
#
# **The previous five attempts fixed the wrong thing, twice over.** Attempts 1-4
# died in shell list-building. Attempt 5 fixed the commit-batching defect and
# WORKED: `m15`'s twenty cells were recovered and pushed in a single commit. But
# it reported `BACKFILL DONE` while `m06`, `m08` and `m10` produced **zero output
# files between them**, and the job exited clean.
#
# Two separate failures were being conflated under "the push is broken":
#
#   1. **Push batching** — real, fixed in attempt 5, and PROVEN by m15. Not this
#      job's problem.
#   2. **Eval timeouts on the Qwen3 base checkpoints** — the actual reason the
#      other 19 cells are missing, and the reason they were missing from the
#      original sweep too. The backfill re-ran exactly the cells that had already
#      failed deterministically, and they failed again. Of course they did.
#
# The evidence is in the surviving cells. A healthy model yields ~197 commands
# per cell at ~10 characters per command. The three failures ramble:
#
#     Qwen3-1.7B-Base    107 cmds/cell   46.9 chars/cmd
#     Qwen3-4B-Base      125 cmds/cell   31.0 chars/cmd
#     Llama-3.1-8B-I     198 cmds/cell   10.1 chars/cmd
#
# Three to five times the tokens per command, so generation is far slower, and a
# cell that needs more than `timeout 900` produces NOTHING — not a short file, no
# file. The push then correctly reports "nothing matching" and `|| true` swallows
# it.
#
# **This matters well beyond 540/540.** The E-axis sweeps run these same three
# models. An unfixed timeout loses E-cells exactly the same way, silently, and
# an E-level with missing cells is far more expensive than a backfill.
#
# So this job changes three things and nothing else:
#
#   - `CELL_TIMEOUT` 900 -> 1200, because these models are known-verbose.
#   - **Zero output for a model is a hard failure**, not a printed note. The
#     per-cell logs are tailed into the job log so the cause is visible from
#     outside the container, which is the thing that made attempt 5 undiagnosable.
#   - Concurrency throttled in batches, restoring the behaviour of the ORIGINAL
#     sweep (`gpu_job27`, which did get cells from these models). Attempt 5
#     launched every cell at once. That is not the cause -- m08 and m10 failed
#     with only three concurrent cells -- but 13 slow generations sharing one
#     server makes each one slower, which feeds straight into the timeout.
#
# Same seal assertion, same vLLM pin, same firewall.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
CELL_TIMEOUT=1200      # was 900. Probe scope: the job is capped at 50m,
                       # and `exit=124` in the per-cell line distinguishes
                       # a timeout from a crash -- which is the answer this
                       # probe exists to get, whichever way it lands.
BATCH=6                # concurrent cells per wave
FAILED_MODELS=()
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
PY

# The list IS the code -- see gpu_job31's header. Nothing is parsed at run time.
[ -f /app/backfill_cmds.sh ] || { echo "STALE MOUNT: backfill_cmds.sh missing"; exit 1; }
N_CALLS=$(grep -c '^run_model ' /app/backfill_cmds.sh || echo 0)
[ "$N_CALLS" = "2" ] || { echo "STALE MOUNT: $N_CALLS run_model calls, expected 2";
                          cat /app/backfill_cmds.sh; exit 1; }
echo "  mount fresh: $N_CALLS run_model calls  (PROBE: m08+m10; m15 done, m06 excluded)"

run_model() {
    local TAG="$1" REPO="$2" CELLS="$3"
    local N; N=$(echo "$CELLS" | wc -w | tr -d ' ')
    echo; echo "######## $TAG  $REPO  ($N cells, timeout ${CELL_TIMEOUT}s) ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    local SERVER_PID=$! READY=0 i
    for i in $(seq 1 60); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /tmp/vllm.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        local PIDS=() K=0 CELL PH WORLD SEED
        for CELL in $CELLS; do
            PH="${CELL%%:*}"; WORLD="${CELL#*:}"; WORLD="${WORLD%%:*}"; SEED="${CELL##*:}"
            (
              timeout $CELL_TIMEOUT python -m seahaven.fidelity.cli eval \
                --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                --step-schedule v1 --seed "$SEED" --world "$WORLD" --phrasing "$PH" \
                --output "$R/dim_${TAG}_${PH}_${WORLD}_${SEED}.json" \
                > /tmp/b_${TAG}_${PH}_${WORLD}_${SEED}.log 2>&1
              echo "    cell ${PH}/${WORLD}/${SEED} exit=$?"
            ) &
            PIDS+=($!); K=$((K+1))
            # Throttle, as the ORIGINAL sweep did. 13 slow generations sharing one
            # server make each slower, which feeds straight into the timeout.
            if [ $((K % BATCH)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"

        local GOT; GOT=$(ls "$R"/dim_${TAG}_*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $TAG produced $GOT/$N cells"

        # **Zero output is a failure, not a note.** Attempt 5 printed "nothing
        # matching" and exited clean for three models. The per-cell logs are the
        # only place the cause lives, and they die with the container -- so they
        # are surfaced here, before that happens.
        if [ "$GOT" = "0" ]; then
            echo "  !! $TAG PRODUCED NOTHING -- per-cell log tails follow"
            for L in /tmp/b_${TAG}_*.log; do
                [ -f "$L" ] || continue
                echo "  ---- $(basename "$L") ----"; tail -15 "$L"
            done
            FAILED_MODELS+=("$TAG")
        else
            python /app/push_batch.py "$R" "dim_${TAG}_*.json" || FAILED_MODELS+=("$TAG:push")
        fi
    else
        echo "  SKIPPED $TAG $REPO — server never ready"
        FAILED_MODELS+=("$TAG:noserver")
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
}

source /app/backfill_cmds.sh

echo; echo "########## BACKFILL DONE ##########"
echo "  cells on disk: $(ls "$R"/dim_*.json 2>/dev/null | wc -l | tr -d ' ')"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  FAILED: ${FAILED_MODELS[*]}"
    echo "  exiting non-zero so this cannot read as success"
    exit 1
fi
echo "  all models produced and pushed cells"
