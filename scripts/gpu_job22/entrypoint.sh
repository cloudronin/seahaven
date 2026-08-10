#!/usr/bin/env bash
# EXPLORATION -- sandbox widening. Three checkpoints, V-P protocol, identical
# to gpu_job21 in every respect except which models are served.
#
# **Why.** The three-anchor survey found no scalar anchor clearing the
# possibility bar on the dev seven, and an exhaustive sweep of every location
# in [80, 101] confirmed the blocker is cohort composition rather than anchor
# family. The worst-case column has one outlier (AI2 at 83.33 / 84.18), a
# 6.7-point gap, and twelve cells packed into the nine points above it.
# Criterion 1 needs two models on the minority side and there is exactly one.
#
# **What these three have to deliver, written before they run:** a
# worst-phrasing q95 at or below the anchor on BOTH worlds -- comfortably under
# 88 is safe -- and it must hold under leave-one-out, or it reproduces AI2's
# p5 dependence somewhere new. Two qualifying models, not one: with two
# determinate FLAGs, criterion 1 passes even if AI2 lands INDETERMINATE.
#
# **Families are already in the burn ledger.** Two Alibaba (currently the top
# scorer at 100.00) and one AI2 (currently the bottom). No new family enters
# the sandbox, so the reserve's "family absent from the dev seven" criterion
# stays uncontaminated and the widening costs nothing in family terms.
#
# **The caveat that decides how to read this**, registered in the log before
# the data exists: `adherence_action` is 1 - violations/all-commands, and noise
# sits in the denominator but not the numerator. Unparseable output therefore
# RAISES adherence. Base checkpoints may land high because their failures are
# multilingual runoff and template markers (noise) rather than English prose in
# the command slot (violation). If they land high, the finding is that base-ness
# is not a source of dynamic range IN THIS MEASURE, and noise handling is why.
#
# **VLLM_BATCH_INVARIANT=1**, as on every run since B2.
#
# 90 evals against gpu_job21's 210, which took 32 minutes. Expect ~15.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
export VLLM_BATCH_INVARIANT=1            # per B2; binds on every run
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }
python - <<'PY' || exit 1
import sys; sys.path.insert(0,"/app")
from seahaven.fidelity.runner import PHRASINGS
from seahaven.fidelity.worldspec import load
assert sorted(PHRASINGS) == ["p1","p2","p3","p4","p5"], PHRASINGS
for w in ("world_v0","world_v2"):
    print(f"  {w}: {len(load(w).entity_keys)} keys", flush=True)
print(f"  phrasings: {sorted(PHRASINGS)}", flush=True)
PY

# Lab names carry no underscore: `_vp_data._is_cell` splits the filename on `_`
# and reads the lab from field 1, so `AI2_Base` would silently parse as lab
# `AI2` and phrasing `Base` and land in the wrong cell.
MODELS=(
  "AlibabaBase|Qwen/Qwen3-8B-Base"
  "AlibabaSmall|Qwen/Qwen2.5-0.5B-Instruct"
  "AI2Base|allenai/OLMo-2-1124-13B"
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
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  server ready after ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm_$LAB.log; break; }
        [ $((i % 6)) -eq 0 ] && echo "  ...waiting ${i}0s: $(tail -1 /tmp/vllm_$LAB.log | cut -c1-80)"
        sleep 10
    done

    # A4-style loop test, inline. Base checkpoints are the reason: two of the
    # three have no instruct tuning, and a missing or unusable chat template
    # shows up as a served model that answers /v1/models and then fails every
    # completion. Without this gate that failure costs 30 evals of wall-clock
    # and produces 30 files of nothing. Two runs of four steps costs seconds.
    if [ "$READY" = "1" ]; then
        python -m seahaven.fidelity.cli eval \
            --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
            --allow-regex-judge --runs 2 --steps 4 --step-schedule v1 \
            --seed 1 --world world_v0 --phrasing p1 \
            --output /tmp/loop_$LAB.json > /tmp/loop_$LAB.log 2>&1
        LOOP=$(python - <<PY
import json
try:
    d = json.load(open("/tmp/loop_$LAB.json"))
    n = sum(len(r.get("commands", [])) for r in d["runs"])
except Exception:
    n = 0
print(n)
PY
)
        echo "  loop test: $LOOP commands emitted"
        if [ "$LOOP" -lt 4 ]; then
            echo "  LOOP TEST FAILED — $LAB cannot hold a parseable action loop."
            echo "  Skipping its sweep rather than paying for empty cells."
            tail -15 /tmp/loop_$LAB.log
            READY=0
        fi
    fi

    if [ "$READY" = "1" ]; then
        PIDS=(); N=0
        for PH in p1 p2 p3 p4 p5; do
          for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
              (
                python -m seahaven.fidelity.cli eval \
                  --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
                  --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
                  --seed $SEED --world "$WORLD" --phrasing "$PH" \
                  --output "$R/vp_${LAB}_${PH}_${WORLD}_${SEED}.json" \
                  > /tmp/vp_${LAB}_${PH}_${WORLD}_${SEED}.log 2>&1
              ) &
              PIDS+=($!); N=$((N+1))
              if [ $((N % 10)) -eq 0 ]; then wait "${PIDS[@]}"; PIDS=(); fi
            done
          done
        done
        [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
        echo "  30 evals done; pushing"
        # Serially, per gpu_job18: concurrent pushes raced the dataset branch
        # and four of Google's six uploads died with HTTP 500.
        for PH in p1 p2 p3 p4 p5; do
          for WORLD in world_v0 world_v2; do
            for SEED in 5150 7301 9412; do
              push "$R/vp_${LAB}_${PH}_${WORLD}_${SEED}.json" || true
            done
          done
        done
    else
        echo "  SKIPPED $LAB"
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 10
    reap_gpu || { echo "GPU DID NOT DRAIN"; exit 1; }
done

echo; echo "########## WHERE THEY LANDED ##########"
# Printed here so the answer is visible from the job log without waiting for
# the download: the whole point of the widening is one number per model.
python - <<'PY'
import glob, json, sys
sys.path.insert(0, "/app")
from collections import defaultdict
from seahaven.fidelity.adherence import classify
cells = defaultdict(lambda: [0, 0])
for f in sorted(glob.glob('/tmp/results/vp_*.json')):
    p = f.split('/')[-1][:-5].split('_')
    if len(p) < 6 or not p[-1].isdigit():
        continue
    lab, ph, world = p[1], p[2], f"{p[3]}_{p[4]}"
    d = json.load(open(f))
    for run in d["runs"]:
        for c in run.get("commands", []):
            cells[(lab, ph, world)][0] += classify(c["command"]) == "violation"
            cells[(lab, ph, world)][1] += 1
labs = sorted({k[0] for k in cells})
print(f"{'lab':<14}{'world':<11}" + "".join(f"{p:>9}" for p in
      ("p1","p2","p3","p4","p5")) + f"{'worst':>9}")
for lab in labs:
    for world in ("world_v0", "world_v2"):
        vals = []
        for ph in ("p1","p2","p3","p4","p5"):
            b, n = cells[(lab, ph, world)]
            vals.append(100.0 * (1 - b / n) if n else float('nan'))
        print(f"{lab:<14}{world:<11}" + "".join(f"{v:>9.2f}" for v in vals)
              + f"{min(vals):>9.2f}")
print("\nTarget: worst under ~88 on BOTH worlds to resolve as a determinate FLAG.")
PY
echo; echo "########## DONE ##########"; ls "$R" | wc -l
