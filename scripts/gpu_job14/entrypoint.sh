#!/usr/bin/env bash
# SMOKE TEST for seahaven-fidelity on GPU. One model, small n, cheap.
#
# Exercises the ACTUAL shipping path -- vLLM's OpenAI server plus the real
# `seahaven-fidelity` CLI -- rather than a bespoke script that happens to import
# the same functions. The local MLX run already proved the difference matters: it
# found three defects that unit tests over validated components had all missed.
#
# This answers one question before the full sweep: with narration fixed to
# continue the agent's own conversation history, does the pipeline produce SIGNAL
# (permutation p < 0.05)? If preflight fails here, the sweep is not worth running.
set -uo pipefail

R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
PORT=8000

push() { python /app/push.py "$1"; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app

echo "=== starting vLLM OpenAI server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port $PORT --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 --max-model-len 4096 \
    > /tmp/vllm.log 2>&1 &
SERVER_PID=$!

# Wait, but bounded, and fail loudly rather than hanging: a server that never
# comes up must not look like a slow one.
for i in $(seq 1 90); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  server ready after ${i}0s"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  SERVER DIED during startup. Last 40 lines:"; tail -40 /tmp/vllm.log
        exit 1
    fi
    sleep 10
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
    echo "  SERVER NEVER BECAME READY. Last 40 lines:"; tail -40 /tmp/vllm.log; exit 1; }

echo
echo "=== seahaven-fidelity smoke ==="
# Regex detector: instrument agreement will SKIP (non-fatal). The question here
# is whether gate -1 finds signal at all, not whether the ranking is publishable.
python -m seahaven.fidelity.cli eval \
    --model "http://127.0.0.1:$PORT/v1" --served-name "$MODEL" \
    --allow-regex-judge --runs 8 --steps 20 \
    --output "$R/gpu_smoke_fidelity.json"
RC=$?
echo "  CLI exit code: $RC   (0 = scored, 3 = preflight refused, 2 = setup error)"

push "$R/gpu_smoke_fidelity.json" || true

echo
echo "########## VERDICT ##########"
python -c "
import json
d=json.load(open('$R/gpu_smoke_fidelity.json'))
pf=d.get('preflight',{})
for c in pf.get('checks',[]):
    print(f\"  [{ {True:'PASS',False:'FAIL',None:'SKIP'}[c['passed']] }] {c['name']}: {c['detail'][:110]}\")
print()
runs=d.get('runs',[])
pats={tuple(sorted(k for k,v in r['acts'].items() if v['performed'])) for r in runs}
print(f'  distinct entity-patterns across runs: {len(pats)}/{len(runs)}  (permutation needs >= 2)')
keys=list(runs[0]['acts'])
varies=sum(1 for k in keys if 0 < sum(r['acts'][k]['performed'] for r in runs) < len(runs))
print(f'  entities that vary: {varies}/{len(keys)}')
for r in runs[:3]:
    did=sorted(k for k,v in r['acts'].items() if v['performed'])
    said=sorted(k for k,v in r['acts'].items() if v['mentioned'])
    print(f\"    run {r['run']} ({r['steps']}st) DID {did}\")
    print(f\"              SAID {said}\")
print(f'  distinct narratives              : {len({r[\"narrative\"] for r in runs})}/{len(runs)}')
for r in runs[:2]:
    print(f\"  sample narrative: {r['narrative'][:170]!r}\")
s=d.get('score',{})
print(f\"\n  fidelity={s.get('fidelity')} omission={s.get('omission_rate')} fabrication={s.get('fabrication_rate')}\")
print(f\"  PREFLIGHT OK: {pf.get('ok')}\")
"
kill $SERVER_PID 2>/dev/null || true
