#!/usr/bin/env bash
# AXIS 2b — same-capability-tier sweep. Frozen band, five new models.
#
# The question: among models at the SAME capability tier, does door-targeted
# break-out vary? Axis 2's between-tier read came back uncomputable at n=7 with a
# correlation that moved with the outcome definition. This holds MMLU-Pro fixed
# (band 35-40, spread 3.20 points across five orgs) and asks whether the rate
# still spreads. If it does, the spread is not the thing the band fixed.
#
# **Only the five NEW models run here.** `Falcon3-10B-Instruct` and
# `Qwen2.5-7B-Instruct` are in the band and were already swept for axis 2 under
# this exact protocol; re-running them would spend money to reproduce numbers
# already held. That their rates are already known is disclosed in the cohort
# module, and the read reports the spread WITH and WITHOUT them so the disclosure
# is checkable rather than merely asserted.
#
# **The serving config is unchanged from axis 2, deliberately.**
# `--gpu-memory-utilization` stays 0.85 even though 0.90 would help the 57B MoE
# fit: the two reused models' rates come from cells collected at 0.85, and a
# config difference between reused and new members would be a comparability
# defect in the one comparison this run exists to make. If the 57B does not fit,
# that is recorded and the band is n=6.
#
# Everything else is the axis-2 job shape, unchanged so the cells are comparable:
# derived per-model context cap (the OLMo lesson), CELL_TIMEOUT=300 (sized to the
# observed healthy wave, not guessed), per-cell exit codes, container-local log
# tails before teardown, and a non-zero exit when a model falls short (TRAP 39).
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
CELL_TIMEOUT=300
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

# Preflight everything at second zero: seal, axis-2 prereg, the 2b cohort and its
# band, all four worlds LOADED, and every level/world proof gate.
python - <<'PYEOF' || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.dimensional import axis2_prereg as A
from seahaven.dimensional import axis2b_cohort as C
from seahaven.dimensional import seal as S
from seahaven.eaxis.barrier import assert_one_lockable
from seahaven.eaxis.levels import assert_level_runnable
from seahaven.fidelity.worldspec import load

S.assert_sealed()
A.assert_prereg()
C.assert_cohort(); C.assert_disjoint_from_seal(); C.assert_in_band()
print(f"  SEAL     {S.SEAL_HASH[:16]}")
print(f"  PREREG   {A.PREREG_HASH[:16]}")
print(f"  COHORT   {C.COHORT_HASH[:16]}  n={C.JUSTIFICATION['n']} "
      f"band={C.BAND} spread={C.JUSTIFICATION['mmlu_spread']:.2f} "
      f"orgs={C.JUSTIFICATION['n_orgs']}")

for w in ("world_ea", "world_eb", "world_ea_E3", "world_eb_E3"):
    spec = load(w)
    print(f"  world {w:<14} start={spec.start_room!r} rooms={len(spec.rooms)}")
for w in ("world_ea_E3", "world_eb_E3"):
    print(f"  barrier  {assert_one_lockable(w)}")
for w in ("world_ea", "world_eb"):
    for lv in ("E0", "E1", "E2"):
        assert_level_runnable(w, lv)
for w in ("world_ea_E3", "world_eb_E3"):
    assert_level_runnable(w, "E3")
print("  PROOFS   every level/world pair committed", flush=True)
PYEOF

# The model list, derived from the frozen cohort. Tag is b<NN> so 2b cells can
# never collide with axis-2 cells in the same dataset repo.
python - <<'PYEOF' > /tmp/models.txt || exit 1
import json, sys; sys.path.insert(0, "/app")
from huggingface_hub import hf_hub_download
from seahaven.dimensional import axis2b_cohort as C
todo = [r for r, v in sorted(C.COHORT.items(), key=lambda kv: -kv[1][0])
        if not v[3]]
for i, repo in enumerate(todo):
    try:
        c = json.load(open(hf_hub_download(repo, "config.json")))
        mpe = c.get("max_position_embeddings") or c.get("n_positions") or 8192
    except Exception:
        mpe = 8192
    print(f"b{i:02d} {repo} {min(8192, int(mpe))}")
PYEOF
N_MODELS=$(wc -l < /tmp/models.txt | tr -d ' ')
[ "$N_MODELS" = "5" ] || { echo "STALE MOUNT: $N_MODELS models, expected 5";
                           cat /tmp/models.txt; exit 1; }
echo "  models   $N_MODELS new band members (2 in band already swept, skipped)"
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
                    --output "$R/eax2b_${TAG}_${LEVEL}_${PH}_${WORLD}_${SEED}.json" \
                    > /tmp/b_${TAG}_${LEVEL}_${PH}_${WORLD}_${SEED}.log 2>&1
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

        local GOT; GOT=$(ls "$R"/eax2b_${TAG}_*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $TAG produced $GOT/48 cells"
        if [ "$GOT" = "0" ]; then
            echo "  !! $TAG PRODUCED NOTHING -- per-cell log tails follow"
            for L in /tmp/b_${TAG}_*.log; do
                [ -f "$L" ] || continue
                echo "  ---- $(basename "$L") ----"; tail -12 "$L"
            done
            FAILED+=("$TAG:none")
        else
            [ "$GOT" -lt 48 ] && FAILED+=("$TAG:partial-$GOT")
            python /app/push_batch.py "$R" "eax2b_${TAG}_*.json" || FAILED+=("$TAG:push")
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

echo; echo "########## AXIS-2B SWEEP DONE ##########"
echo "  cells this run: $(ls "$R"/eax2b_*.json 2>/dev/null | wc -l | tr -d ' ')/240"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  INCOMPLETE: ${FAILED[*]}"
    echo "  exiting non-zero so this cannot read as success"
    exit 1
fi
echo "  all 5 new band members produced and pushed 48 cells"
