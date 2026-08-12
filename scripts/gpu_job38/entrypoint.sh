#!/usr/bin/env bash
# EDENBENCH exploratory ladder. Prohibition arm only, six models, six levels.
#
# The question: do different models produce different curves across pressure?
# Nothing is gated here and nothing is pre-registered -- this run exists to see
# the shape. If something looks interesting we decide afterwards what would have
# to be true.
#
# A level IS its larder. The six worlds share one topology and one byte-identical
# prompt; only the amount of legal food differs, so pressure cannot enter through
# the text. Health is harness-side and driven by facts, because a failed `eat`
# scores ok=True (Gate 1).
#
# NO A0 ARM in this run, which is a real limitation and is stated rather than
# discovered later: without a no-prohibition baseline a low P(eat) cannot be
# cleanly separated from "could not compose take-then-eat". The funnel's
# attempted-vs-ate gap is the partial substitute.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
CELL_TIMEOUT=600
BATCH=6
LEVELS="L1 L2 L3 L4 L5 NEC"
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

# Preflight: every world loads, its manifest matches the compiled artifact, and
# the arithmetic in the lock recomputes. A payload whose worlds disagree with
# their locks is refused here rather than after a model load.
python - <<'PYEOF' || exit 1
import sys; sys.path.insert(0, "/app")
from seahaven.eden.manifest import assert_lock_consistent
from seahaven.eden.outcome import level_state, load_level, eden_system_prompt
from seahaven.fidelity.worldspec import load
for lv in ("L1","L2","L3","L4","L5","NEC"):
    wid = f"world_eden_{lv}"
    lk = load_level(wid)
    assert_lock_consistent(lk, world_json=f"/app/worlds/{wid}/{wid}.json")
    spec = load(wid); st = level_state(lk)
    d = lk["derived"]
    print(f"  {wid:<18} S={d['supply_ratio']:.2f} price={d['price_per_step']:>6.2f} "
          f"optMin={d['legal_only_min']:>4} greedyMin={d['greedy_min']:>4} "
          f"foods={len(lk['larder'])}")
p = eden_system_prompt(load("world_eden_L1"), "blue gourd")
assert "not to be eaten" in p and "eat <thing>" in p
assert "because" not in p.lower() and "poison" not in p.lower()
print("  PROMPT   identical across levels, eat-scoped, no reason given", flush=True)
PYEOF

python - <<'PYEOF' > /tmp/models.txt || exit 1
# Six instruct models spanning the exploration set's capability range.
for line in [
    "e0 tiiuae/Falcon3-10B-Instruct 38.10",
    "e1 Qwen/Qwen2.5-7B-Instruct 36.52",
    "e2 google/gemma-2-9b-it 31.95",
    "e3 meta-llama/Llama-3.1-8B-Instruct 31.09",
    "e4 Qwen/Qwen2.5-3B-Instruct 25.05",
    "e5 mistralai/Mistral-7B-Instruct-v0.3 23.06",
]:
    print(line)
PYEOF
echo "  models   6 instruct, MMLU-Pro 23.06-38.10"
cat /tmp/models.txt | sed 's/^/    /'

run_model() {
    local TAG="$1" REPO="$2"
    echo; echo "######## $TAG  $REPO ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 8192 \
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
        for LV in $LEVELS; do
          (
            timeout $CELL_TIMEOUT python -m seahaven.fidelity.cli eval \
              --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
              --allow-regex-judge --no-narrate --runs 12 --steps 30 \
              --seed 5150 --world "world_eden_${LV}" --eden-level "$LV" \
              --output "$R/eden_${TAG}_${LV}.json" \
              > /tmp/e_${TAG}_${LV}.log 2>&1
            RC=$?
            [ $RC -ne 0 ] && echo "    cell $LV exit=$RC"
          ) &
          PIDS+=($!); K=$((K+1))
          if [ $((K % BATCH)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        local GOT; GOT=$(ls "$R"/eden_${TAG}_*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $TAG produced $GOT/6 cells"
        if [ "$GOT" = "0" ]; then
            echo "  !! $TAG PRODUCED NOTHING -- log tails follow"
            for L in /tmp/e_${TAG}_*.log; do
                [ -f "$L" ] || continue
                echo "  ---- $(basename "$L") ----"; tail -12 "$L"
            done
            FAILED+=("$TAG:none")
        else
            [ "$GOT" -lt 6 ] && FAILED+=("$TAG:partial-$GOT")
            python /app/push_batch.py "$R" "eden_${TAG}_*.json" || FAILED+=("$TAG:push")
        fi
    else
        echo "  SKIPPED $TAG — server never ready"; FAILED+=("$TAG:noserver")
    fi
    kill -9 $PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
}

while read -r TAG REPO MMLU; do
    [ -z "$TAG" ] && continue
    run_model "$TAG" "$REPO"
done < /tmp/models.txt

echo; echo "########## EDENBENCH LADDER DONE ##########"
echo "  cells: $(ls "$R"/eden_*.json 2>/dev/null | wc -l | tr -d ' ')/36"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  INCOMPLETE: ${FAILED[*]}"; exit 1
fi
echo "  all 6 models produced 6 levels"
