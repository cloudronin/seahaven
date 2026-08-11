#!/usr/bin/env bash
# AXIS 2 — the E-level sweep. 18 exploration models x 4 levels x 12 cells.
#
# Levels are BETWEEN-CONDITION: every cell is a separate run from reset, one
# level per run. E0/E1/E2 play the base worlds; E3 plays the variants whose only
# difference is `locked(d)` where the base has `closed(d)`.
#
# **Grid: 2 phrasings x 2 worlds x 3 seeds = 12 cells per level.** The hashed
# pre-registration fixes the SHAPE (`SWEEP_GRID`); the two phrasings are chosen
# here, before any E-axis number exists, and recorded for that reason:
#
#   p1 — the corpus baseline, byte-identical to the prompt every prior result in
#        this project was measured under;
#   p3 — the most explicit declaration of the five ("The permitted command
#        vocabulary consists exclusively of ... Commands outside this vocabulary
#        are invalid").
#
# They bracket the range of how firmly the constraint is stated, which is the
# dimension most likely to interact with pressure. Picking the two middle
# phrasings would leave the interesting contrast untested; picking p1 alone
# would leave phrasing entirely unvaried at the level where the legal route runs
# out.
#
# **Every level above E0 refuses to start without a committed reachability
# proof.** That refusal lives in `run_fidelity`, not in this script, so it holds
# for any entry point — a pressure level without a proof cannot distinguish
# "chose not to" from "could not", which is the confound that would manufacture
# a capability reduction.
#
# THREE CARRIED FIXES, all from failures this project already paid for:
#
#   - `--max-model-len 8192` (was 4096). Verbose base checkpoints overran the
#     window, the endpoint returned 400s, and retries burned the cell's wall
#     clock. Verified a byte-identity no-op against 4096 before this ran.
#   - the lenient UTF-8 decode on the endpoint success path. Qwen3 base
#     checkpoints emit invalid UTF-8 and `json.loads` on raw bytes killed whole
#     cells.
#   - `push_batch.py`, one commit per model instead of one per file.
#
# AND THE JOB SHAPE FROM TRAP 39, which is the part that actually ended five
# rounds of guessing: report a per-cell exit code, tail the container-local logs
# BEFORE teardown, and exit non-zero when a model produces nothing. The previous
# backfill printed "DONE" and exited clean while three of four models produced
# zero files.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
CELL_TIMEOUT=1800
BATCH=6
FAILED=()
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

# **Preflight everything the sweep depends on, at second zero.** A stale payload
# already cost one job: `seahaven/` was staged before `worldspec.SETTINGS` had
# the E-worlds, and every cell died AFTER a 150s model load with a bare exit=1.
python - <<'PY' || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import axis2_prereg as A
from seahaven.dimensional import seal as S
from seahaven.eaxis.levels import assert_level_runnable
from seahaven.fidelity.worldspec import load

S.assert_sealed()
S.assert_not_held_out(S.EXPLORATION)
A.assert_prereg()
print(f"  SEAL     {S.SEAL_HASH[:16]}  firewall clean, {len(S.EXPLORATION)} models")
print(f"  PREREG   {A.PREREG_HASH[:16]}  floor {A.FLOOR} frontier {A.FRONTIER_FLOOR}")

for w in ("world_ea", "world_eb", "world_ea_E3", "world_eb_E3"):
    spec = load(w)
    print(f"  world {w:<14} start={spec.start_room!r} rooms={len(spec.rooms)}")

for w in ("world_ea", "world_eb"):
    for lv in ("E0", "E1", "E2"):
        assert_level_runnable(w, lv)
for w in ("world_ea_E3", "world_eb_E3"):
    assert_level_runnable(w, "E3")
print("  PROOFS   every level/world pair has a committed reachability proof",
      flush=True)
PY

python - <<'PY' > /tmp/models.txt || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import seal as S
for i, repo in enumerate(S.EXPLORATION):
    print(f"m{i:02d} {repo}")
PY
N_MODELS=$(wc -l < /tmp/models.txt | tr -d ' ')
[ "$N_MODELS" = "18" ] || { echo "STALE MOUNT: $N_MODELS models, expected 18"; exit 1; }
echo "  models   $N_MODELS from the seal"

run_model() {
    local TAG="$1" REPO="$2"
    echo; echo "######## $TAG  $REPO ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 8192 \
        > /tmp/vllm.log 2>&1 &
    local PID=$! READY=0 i
    for i in $(seq 1 60); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /tmp/vllm.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        local PIDS=() K=0
        # Frozen literals, looped in shell -- the same shape gpu_job27 used
        # successfully. Nothing is parsed from a data file at run time.
        for LEVEL in E0 E1 E2 E3; do
          for PH in p1 p3; do
            for BASE in world_ea world_eb; do
              WORLD="$BASE"
              [ "$LEVEL" = "E3" ] && WORLD="${BASE}_E3"
              for SEED in 5150 7301 9412; do
                (
                  timeout $CELL_TIMEOUT python -m seahaven.fidelity.cli eval \
                    --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                    --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                    --step-schedule v1 --seed "$SEED" --world "$WORLD" \
                    --phrasing "$PH" --e-level "$LEVEL" \
                    --output "$R/eax_${TAG}_${LEVEL}_${PH}_${WORLD}_${SEED}.json" \
                    > /tmp/e_${TAG}_${LEVEL}_${PH}_${WORLD}_${SEED}.log 2>&1
                  RC=$?
                  [ $RC -ne 0 ] && echo "    cell $LEVEL/$PH/$WORLD/$SEED exit=$RC"
                ) &
                PIDS+=($!); K=$((K+1))
                if [ $((K % BATCH)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
              done
            done
          done
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"

        local GOT; GOT=$(ls "$R"/eax_${TAG}_*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $TAG produced $GOT/48 cells"
        if [ "$GOT" = "0" ]; then
            echo "  !! $TAG PRODUCED NOTHING -- per-cell log tails follow"
            for L in /tmp/e_${TAG}_*.log; do
                [ -f "$L" ] || continue
                echo "  ---- $(basename "$L") ----"; tail -12 "$L"
            done
            FAILED+=("$TAG:none")
        else
            [ "$GOT" -lt 48 ] && FAILED+=("$TAG:partial-$GOT")
            python /app/push_batch.py "$R" "eax_${TAG}_*.json" || FAILED+=("$TAG:push")
        fi
    else
        echo "  SKIPPED $TAG — server never ready"
        FAILED+=("$TAG:noserver")
    fi
    kill -9 $PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
}

while read -r TAG REPO; do
    [ -z "$TAG" ] && continue
    run_model "$TAG" "$REPO"
done < /tmp/models.txt

echo; echo "########## E-SWEEP DONE ##########"
echo "  cells on disk: $(ls "$R"/eax_*.json 2>/dev/null | wc -l | tr -d ' ')/864"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  INCOMPLETE: ${FAILED[*]}"
    echo "  exiting non-zero so this cannot read as success"
    exit 1
fi
echo "  all 18 models produced and pushed 48 cells"
