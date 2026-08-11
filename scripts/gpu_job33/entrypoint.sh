#!/usr/bin/env bash
# CONTEXT-LENGTH NO-OP CHECK — is `--max-model-len 8192` free?
#
# **The noise floors are reference #1 of the three-reference read, and they were
# measured at 4096.** The E-sweeps raise the window to 8192 to fix the context
# overflow that produced HTTP 400s and a timed-out cell (TRAP 39). That is the
# right fix, and it is also a new config — and a new config getting less scrutiny
# than the thing it fixed is how this project has been bitten repeatedly.
#
# If context length changes generation at all — a different truncation point, a
# different kernel path, a different KV block layout — then every per-model noise
# floor being SUBTRACTED in the E-axis read was measured under a config the
# E-sweep does not run under. That is a silent comparability confound sitting
# underneath reference #1.
#
# **The check is the same shape that verified the UTF-8 fix.** That fix was
# argued to be a no-op on valid UTF-8; this asks the same question of 8192 on
# transcripts that overflow neither limit:
#
#   - one model that was bit-exact at 4096 (`Llama-3.1-8B-Instruct`, sd 0.000,
#     8B so batch-invariance binds on vLLM 0.26.0/H200);
#   - cells short enough to sit far under 4096 (`--steps 8` on the small
#     E-worlds), so the ONLY thing differing between arms is the flag;
#   - identical seeds, identical everything else, `VLLM_BATCH_INVARIANT=1` as
#     the determinism map was measured;
#   - compare the emitted commands BYTE FOR BYTE.
#
# Identical -> the floors transfer and the E-sweep is clean. Not identical -> the
# determinism map needs a frozen caveat that it was measured at a different
# context length than the sweep runs at, and the floors get re-measured at 8192
# for the models the analysis leans on. Either way it is decided BEFORE the
# spend, not discovered in analysis.
#
# Headroom is also reported: if the sweep config never approaches 4096 on these
# worlds, the flag is irrelevant to the E-axis regardless of the byte check, and
# that is worth knowing too.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
MODEL="meta-llama/Llama-3.1-8B-Instruct"
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
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import seal as S
S.assert_sealed()
S.assert_not_held_out(["meta-llama/Llama-3.1-8B-Instruct"])
print(f"  SEAL VERIFIED {S.SEAL_HASH[:16]}  firewall clean", flush=True)

# **Preflight the WORLDS, not just the seal.** The first run of this job staged
# `seahaven/` before `worldspec.SETTINGS` had entries for the E-worlds, so every
# cell died on "no setting sentence registered" -- but only AFTER a 150s model
# load, and the failure surfaced as a bare `exit=1` per cell. A stale payload is
# a recurring failure here (see the stale-mount assertion in gpu_job29); the fix
# is to touch every world the job will use, at second zero.
from seahaven.fidelity.worldspec import load
for w in ("world_ea",):
    spec = load(w)
    print(f"  world {w} loads: start={spec.start_room!r} "
          f"rooms={len(spec.rooms)}", flush=True)
PY

run_arm() {
    local CTX="$1" ARM="$2"
    echo; echo "######## arm $ARM  max-model-len=$CTX ########"
    python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len "$CTX" \
        > /tmp/vllm_$CTX.log 2>&1 &
    local PID=$! READY=0 i
    for i in $(seq 1 60); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /tmp/vllm_$CTX.log; break; }
        sleep 10
    done
    if [ "$READY" != "1" ]; then echo "  ARM $CTX FAILED"; return 1; fi

    local PIDS=()
    for SEED in 5150 7301; do
        (
          timeout 1500 python -m seahaven.fidelity.cli eval \
            --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
            --allow-regex-judge --no-narrate --runs 12 --steps 8 \
            --step-schedule v1 --seed "$SEED" --world world_ea --phrasing p1 \
            --output "$R/ctx${ARM}${CTX}_${SEED}.json" \
            > /tmp/ctx_${ARM}_${SEED}.log 2>&1
          echo "    arm=$ARM ctx=$CTX seed=$SEED exit=$?"
        ) &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    ls -1 "$R"/ctx${ARM}${CTX}_*.json 2>/dev/null | wc -l | xargs echo "  cells:"
    for L in /tmp/ctx_${ARM}_*.log; do
        [ -f "$L" ] || continue
        grep -qiE "error|traceback" "$L" && { echo "  ---- $(basename $L) ----"; tail -8 "$L"; }
    done
    kill -9 $PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
}

# THE CONTROL RUNS FIRST. A and B are identical configuration; if they differ,
# the harness is not reproducible at this concurrency and the 8192 comparison
# is unreadable. The previous run of this check omitted the control and
# returned an uninterpretable "differs on one seed of two" -- TRAP 32, again.
run_arm 4096 A
run_arm 4096 B
run_arm 8192 C

echo; echo "########## BYTE COMPARISON ##########"
python /app/compare.py
RC=$?
python /app/push_batch.py "$R" "ctx*.json" || true
echo "########## DONE rc=$RC ##########"
exit $RC
