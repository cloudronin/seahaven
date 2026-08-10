#!/usr/bin/env bash
# IS BATCH-INVARIANCE MODEL-DEPENDENT?
#
# The control (gpu_job24) found two identical runs of one eval diverging under
# VLLM_BATCH_INVARIANT=1 on Qwen2.5-0.5B-Instruct: fidelity 62.78 vs 50.00.
# B2 established 12/12 identical under the same flag and the same vLLM 0.26.0 —
# on meta-llama/Llama-3.1-8B-Instruct.
#
# One control, one model, generalised to a twelve-model cohort. This job tests
# that generalisation directly: the SAME repeat-and-compare, in ONE container,
# on BOTH models, back to back. Everything except the served model is held fixed.
#
# Readings fixed before the run:
#   Llama repeats,  Qwen diverges -> invariance is MODEL-DEPENDENT. B2's result
#                                    stands for what it tested and was
#                                    over-generalised. The corpus inherits an
#                                    unquantified per-model noise term.
#   both diverge                  -> the flag has stopped binding on this stack
#                                    since B2, and B2's result no longer holds
#                                    even for its own subject.
#   both repeat                   -> gpu_job24's divergence had some other cause
#                                    and this whole line needs re-opening.
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
echo "  VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT}"

for M in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct"; do
  TAG=$(echo "$M" | tr '/.' '__')
  echo; echo "################ $M ################"
  python -m vllm.entrypoints.openai.api_server --model "$M" --port $PORT \
      --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 4096 \
      > /tmp/vllm.log 2>&1 &
  SERVER_PID=$!
  READY=0
  for i in $(seq 1 90); do
    if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
    then READY=1; echo "  server ready after ${i}0s"; break; fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /tmp/vllm.log; break; }
    sleep 10
  done
  if [ "$READY" = "1" ]; then
    for REP in 1 2; do
      python -m seahaven.fidelity.cli eval \
        --model "http://127.0.0.1:$PORT/v1" --served-name "$M" \
        --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
        --seed 5150 --world world_v0 --phrasing p1 \
        --output "$R/${TAG}_rep${REP}.json" > /dev/null 2>&1
      echo "  rep$REP done"
    done
    python - <<PY
import json
a = json.load(open("$R/${TAG}_rep1.json")); b = json.load(open("$R/${TAG}_rep2.json"))
ca = [r.get("commands") for r in a["runs"]]; cb = [r.get("commands") for r in b["runs"]]
same = ca == cb
n = sum(1 for x, y in zip(ca, cb) if x == y)
print(f"  VERDICT $M: commands {'IDENTICAL' if same else 'DIFFER'}"
      f"  ({n}/{min(len(ca),len(cb))} runs match)"
      f"  fidelity {a['score']['fidelity']} vs {b['score']['fidelity']}")
PY
  else
    echo "  SKIPPED $M"
  fi
  kill -9 $SERVER_PID 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 10; reap_gpu || true
done
echo; echo "########## DONE ##########"
