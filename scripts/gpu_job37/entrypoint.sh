#!/usr/bin/env bash
# C3 STAGE 1 — validate the discovery probe. This is where C3 lives or dies.
#
# Three legs of the gate ask whether the probe agrees WITH ITSELF (phrasing
# stability, per-episode agreement, an informative envelope). P4 asks whether it
# is CORRECT, by testing it against the cold-unlock rate measured from the
# committed 2b corpus. A probe that passes P1-P3 and fails P4 is stable and
# wrong, which is worse than useless because it looks trustworthy. So P4 is the
# number this run exists to produce.
#
# **Every phrasing is asked of the SAME episode.** P2's per-episode agreement is
# undefined otherwise, and four separate sweeps would pay for the rollout four
# times to reconstruct a pairing this gets for free. `minimal` appears twice at
# different sampling seeds: that pair IS the test-retest ceiling P2 is scored
# against.
#
# **Fresh seeds, deliberately.** 2b used 5150/7301/9412 and the stack is
# bit-exact under VLLM_BATCH_INVARIANT=1, so reusing them would reproduce 2b's
# episodes exactly — and the probe rate would then be computed on the very
# episodes the anchor came from. P4 needs two independent samples of the same
# models, which is what makes selecting Stage-1 members on the anchor legitimate.
#
# Everything else is the axis-2b job shape unchanged, so these cells stay
# comparable: vLLM 0.26.0 pinned, gpu-memory-utilization 0.85, derived per-model
# context cap, CELL_TIMEOUT sized to the observed healthy wave, per-cell exit
# codes, container-local log tails, non-zero exit on shortfall (TRAP 39).
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
CELL_TIMEOUT=420
BATCH=6
PROBES="minimal,minimal,neutral,direct"
SEEDS="2029 3137 4243 5351 6469 7573 8681"
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

# Preflight at second zero: seal, axis-2 prereg, C3 prereg + Stage-1 spread,
# the probe's no-leak guard, both E3 worlds loaded and their proofs committed.
python - <<'PYEOF' || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import axis2_prereg as A
from seahaven.dimensional import c3_prereg as C3
from seahaven.dimensional import seal as S
from seahaven.eaxis.barrier import assert_one_lockable
from seahaven.eaxis.levels import assert_level_runnable
from seahaven.eaxis.probe import PROBE_PHRASINGS, assert_probe_reveals_nothing
from seahaven.fidelity.worldspec import load

S.assert_sealed()
A.assert_prereg()
C3.assert_c3()
C3.assert_stage1_spread()
assert_probe_reveals_nothing()
print(f"  SEAL   {S.SEAL_HASH[:16]}")
print(f"  PREREG {A.PREREG_HASH[:16]}")
print(f"  C3     {C3.C3_HASH[:16]}")
print(f"  probe  {sorted(PROBE_PHRASINGS)} — none names the verb or the key")
for m in C3.STAGE1_MODELS:
    print(f"    stage1 {m:<34} cold-unlock {C3.COLD_UNLOCK_RATE[m]:.3f}")
for w in ("world_ea_E3", "world_eb_E3"):
    spec = load(w)
    print(f"  world {w:<14} start={spec.start_room!r} rooms={len(spec.rooms)}")
    print(f"  barrier {assert_one_lockable(w)}")
    assert_level_runnable(w, "E3")
print("  PROOFS every E3 world committed", flush=True)
PYEOF

# Model list derived from the frozen Stage-1 selection, never retyped.
python - <<'PYEOF' > /tmp/models.txt || exit 1
import json, sys; sys.path.insert(0, "/app")
from huggingface_hub import hf_hub_download
from seahaven.dimensional import c3_prereg as C3
for i, repo in enumerate(C3.STAGE1_MODELS):
    try:
        c = json.load(open(hf_hub_download(repo, "config.json")))
        mpe = c.get("max_position_embeddings") or c.get("n_positions") or 8192
    except Exception:
        mpe = 8192
    print(f"c{i:02d} {repo} {min(8192, int(mpe))}")
PYEOF
N_MODELS=$(wc -l < /tmp/models.txt | tr -d ' ')
[ "$N_MODELS" = "3" ] || { echo "STALE MOUNT: $N_MODELS models, expected 3";
                           cat /tmp/models.txt; exit 1; }
echo "  models   $N_MODELS Stage-1 members, chosen for cold-unlock spread"
cat /tmp/models.txt | sed 's/^/    /'

run_model() {
    local TAG="$1" REPO="$2" CTX="$3"
    echo; echo "######## $TAG  $REPO  (ctx $CTX) ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len "$CTX" \
        --trust-remote-code > /tmp/vllm.log 2>&1 &
    local PID=$! READY=0 i
    for i in $(seq 1 90); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $PID 2>/dev/null || { echo "  SERVER DIED"; tail -25 /tmp/vllm.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        local PIDS=() K=0
        for PH in p1 p3; do
          for BASE in world_ea world_eb; do
            WORLD="${BASE}_E3"
            for SEED in $SEEDS; do
              (
                timeout $CELL_TIMEOUT python -m seahaven.fidelity.cli eval \
                  --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                  --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                  --step-schedule v1 --seed "$SEED" --world "$WORLD" \
                  --phrasing "$PH" --e-level E3 --probe "$PROBES" \
                  --output "$R/c3s1_${TAG}_${PH}_${WORLD}_${SEED}.json" \
                  > /tmp/c_${TAG}_${PH}_${WORLD}_${SEED}.log 2>&1
                RC=$?
                [ $RC -ne 0 ] && echo "    cell $PH/$WORLD/$SEED exit=$RC"
              ) &
              PIDS+=($!); K=$((K+1))
              if [ $((K % BATCH)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
            done
          done
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"

        local GOT; GOT=$(ls "$R"/c3s1_${TAG}_*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $TAG produced $GOT/28 cells"
        if [ "$GOT" = "0" ]; then
            echo "  !! $TAG PRODUCED NOTHING -- per-cell log tails follow"
            for L in /tmp/c_${TAG}_*.log; do
                [ -f "$L" ] || continue
                echo "  ---- $(basename "$L") ----"; tail -12 "$L"
            done
            FAILED+=("$TAG:none")
        else
            [ "$GOT" -lt 28 ] && FAILED+=("$TAG:partial-$GOT")
            python /app/push_batch.py "$R" "c3s1_${TAG}_*.json" || FAILED+=("$TAG:push")
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

ONLY="${ONLY:-}"
while read -r TAG REPO CTX; do
    [ -z "$TAG" ] && continue
    if [ -n "$ONLY" ]; then
        case " $ONLY " in *" $TAG "*) ;; *) continue ;; esac
    fi
    run_model "$TAG" "$REPO" "$CTX"
done < /tmp/models.txt

echo; echo "########## C3 STAGE-1 SWEEP DONE ##########"
echo "  cells this run: $(ls "$R"/c3s1_*.json 2>/dev/null | wc -l | tr -d ' ')/84"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  INCOMPLETE: ${FAILED[*]}"
    echo "  exiting non-zero so this cannot read as success"
    exit 1
fi
echo "  all 3 Stage-1 members produced and pushed 28 cells"
