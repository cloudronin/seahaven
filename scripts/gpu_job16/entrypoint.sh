#!/usr/bin/env bash
# Phase 1 re-baseline + V2 cross-world stability, in one pass.
#
# Two things forced a re-run rather than a re-analysis of the held results:
#
#   1. TRAP 26 — `runner` hardcoded TAKEABLE with "rope" and "key" where the
#      world calls them "coil of rope" and "brass key". Ground truth reads the
#      engine's fact strings, so those two columns were False in all 499
#      published runs and a model truthfully reporting the rope was scored as
#      fabricating. Result files store only derived `acts`, not the raw facts,
#      so the truth cannot be recomputed offline. The episodes have to be rerun.
#   2. V2 needs world_v2 played by the same models under the same protocol.
#
# Both worlds, same seeds, same runs/steps, same prompts apart from the single
# setting sentence — see `seahaven/fidelity/worldspec.py`. Paired seeds so the
# cross-world comparison is within-seed rather than across independent draws.
#
# Three seeds per world rather than six per world: `_steps_for` already varies
# episode length inside one `--runs 12` call (4/12/20/30, three runs each), so
# each seed is a complete length-stratified test on its own and the seeds are
# repeats for `reliability()`. Three is its minimum and keeps this job at one
# job's cost while covering twice the worlds.
set -uo pipefail

R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000

push() { python /app/push.py "$1"; }
# reap_gpu() knows vLLM's EngineCore children survive a kill of the parent.
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app

python - <<'PY' || exit 1
# Fail before renting time on a defect the smoke test already covers: every
# entity column must be reachable in both worlds, or the run scores constants.
import sys; sys.path.insert(0, "/app")
from seahaven.fidelity.worldspec import load
for w in ("world_v0", "world_v2"):
    s = load(w)
    assert s.takeable and s.rooms and s.start_room, w
    print(f"  {w}: {len(s.entity_keys)} keys, start={s.start_room}", flush=True)
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
        if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            READY=1; echo "  server ready after ${i}0s"; break
        fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -25 /tmp/vllm_$LAB.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
                python -m seahaven.fidelity.cli eval \
                    --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
                    --allow-regex-judge --runs 12 --steps 30 --seed $SEED \
                    --world "$WORLD" \
                    --output "$R/x_${LAB}_${WORLD}_${SEED}.json" 2>&1 | tail -10
                push "$R/x_${LAB}_${WORLD}_${SEED}.json" || true
            done
        done
    else
        echo "  SKIPPED $LAB — server never became ready"
    fi

    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.spawn" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN — aborting rather than blaming the next model"; exit 1; }
done

echo; echo "########## SUMMARY ##########"
python - <<'PY'
import json, glob
rows = {}
for f in sorted(glob.glob('/tmp/results/x_*.json')):
    d = json.load(open(f))
    _, lab, world, seed = f.split('/')[-1].replace('.json', '').split('_', 3)
    ok = d.get('preflight', {}).get('ok')
    s = d.get('score', {}).get('fidelity')
    rows.setdefault((lab, world), []).append((ok, s))
print(f"{'lab':<12}{'world':<11}{'preflight':>10}  {'status':<16}{'fidelity':>26}")
for (lab, world), v in sorted(rows.items()):
    oks = sum(1 for o, _ in v if o)
    vals = [s for o, s, in v if o and s is not None]
    # Never tabulate only the passing repeats -- selection on an
    # outcome-adjacent criterion, prohibited by spec section 2.
    status = 'ok' if oks == len(v) else ('NON_ELICITABLE' if oks == 0 else f'UNSTABLE {oks}/{len(v)}')
    cell = ' '.join(f'{x:.1f}' for x in vals) if vals else '-'
    print(f'{lab:<12}{world:<11}{str(oks)+"/"+str(len(v)):>10}  {status:<16}{cell:>26}')
PY
