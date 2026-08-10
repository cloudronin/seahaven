#!/usr/bin/env bash
# THE DETERMINISM CONTROL THAT SHOULD HAVE PRECEDED THE IDENTITY CHECK.
#
# Check 1's regression re-ran a committed eval and found the command stream
# differed while the score matched exactly. That is not yet evidence about the
# --no-narrate change, because the check presupposed the environment reproduces
# at all — the same inversion TRAP 32 is about, repeated.
#
# The --no-narrate change is exonerated by construction: every added executable
# line sits inside an `if not narrate:` guard, unreachable on the scored path.
# This job supplies the evidence.
#
# **The one condition that differed.** The baseline cell was produced in
# gpu_job22 as the FIRST of ten evals launched concurrently (~120 sequences in
# flight). Check 1 re-ran it ALONE (12 sequences). VLLM_BATCH_INVARIANT=1 is
# meant to make results independent of batch composition, and B2 verified
# reproducibility at FIXED concurrency — it never varied composition.
#
# Three conditions, one served model, one job:
#
#   S1  the cell alone                     (reproduces Check 1's condition)
#   S2  the cell alone, again              (same-condition determinism)
#   C1  the cell inside its original 10    (reproduces gpu_job22's condition)
#
# Reading, fixed before the run:
#   S1 == S2 and C1 == baseline  -> flag binds WITHIN a composition and results
#                                   depend on composition. The corpus has a
#                                   hidden factor: which cells shared a batch.
#   S1 == S2 but C1 != baseline  -> composition is not the whole story; some
#                                   other environmental term moved.
#   S1 != S2                     -> the flag is not binding in this container at
#                                   all, and B2's guarantee does not hold here.
#
# In every branch the --no-narrate change is untouched by the outcome; what
# moves is what the project can claim about cross-batch comparability.
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
export VLLM_BATCH_INVARIANT=1            # the thing under test
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }

M="Qwen/Qwen2.5-0.5B-Instruct"
python -m vllm.entrypoints.openai.api_server \
    --model "$M" --port $PORT --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 --max-model-len 4096 > /tmp/vllm.log 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 90); do
    if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
    then READY=1; echo "  server ready after ${i}0s"; break; fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm.log; break; }
    sleep 10
done
[ "$READY" = "1" ] || { echo "SERVER NEVER READY"; exit 1; }

cell() {  # $1=output  $2=phrasing  $3=world  $4=seed
    python -m seahaven.fidelity.cli eval \
        --model "http://127.0.0.1:$PORT/v1" --served-name "$M" \
        --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
        --seed "$4" --world "$3" --phrasing "$2" --output "$1" > "$1.log" 2>&1
}

echo; echo "---- S1: the cell, alone ----"
cell "$R/S1.json" p1 world_v0 5150; echo "  done"
echo "---- S2: the cell, alone, again ----"
cell "$R/S2.json" p1 world_v0 5150; echo "  done"

echo "---- C1: the cell inside its original batch of ten ----"
# gpu_job22 iterated phrasing x world x seed and waited every 10, so the target
# cell was the FIRST of this exact group. Reproduced verbatim.
PIDS=()
for SPEC in "p1 world_v0 5150" "p1 world_v0 7301" "p1 world_v0 9412" \
            "p1 world_v2 5150" "p1 world_v2 7301" "p1 world_v2 9412" \
            "p2 world_v0 5150" "p2 world_v0 7301" "p2 world_v0 9412" \
            "p2 world_v2 5150"; do
    set -- $SPEC
    ( cell "$R/C1_$1_$2_$3.json" "$1" "$2" "$3" ) &
    PIDS+=($!)
done
wait "${PIDS[@]}"; echo "  done"

kill -9 $SERVER_PID 2>/dev/null || true
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 10; reap_gpu || true

echo; echo "############ CONTROL VERDICT ############"
python - <<'PY'
import json
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def cmds(d):
    return None if d is None else [r.get("commands") for r in d["runs"]]
base = load("/app/baseline/vp_AlibabaSmall_p1_world_v0_5150.json")
s1, s2 = load("/tmp/results/S1.json"), load("/tmp/results/S2.json")
c1 = load("/tmp/results/C1_p1_world_v0_5150.json")

def cmp(a, b, an, bn):
    if a is None or b is None:
        print(f"  {an} vs {bn}: MISSING"); return
    same_cmd = cmds(a) == cmds(b)
    same_all = a["runs"] == b["runs"]
    print(f"  {an} vs {bn}:  commands {'SAME' if same_cmd else 'DIFFER'}"
          f"   full-runs {'SAME' if same_all else 'DIFFER'}"
          f"   n={len(a['runs'])}/{len(b['runs'])}"
          f"   fidelity {a['score']['fidelity']} / {b['score']['fidelity']}")

cmp(s1, s2, "S1", "S2  (same condition)")
cmp(s1, base, "S1", "baseline (10-way)")
cmp(c1, base, "C1", "baseline (10-way)")
cmp(c1, s1, "C1", "S1")
print()
if s1 and s2 and cmds(s1) == cmds(s2):
    print("  S1==S2: the flag DOES bind at fixed composition.")
    if c1 and base and cmds(c1) == cmds(base):
        print("  C1==baseline: results depend on BATCH COMPOSITION, not on the")
        print("  --no-narrate change. The corpus carries a hidden factor —")
        print("  which cells shared a batch — and cross-batch byte-identity is")
        print("  not something this stack provides.")
    else:
        print("  C1!=baseline: composition alone does not explain it; some other")
        print("  environmental term moved between jobs.")
else:
    print("  S1!=S2: the flag is NOT binding in this container. B2's guarantee")
    print("  does not hold here, and that supersedes everything downstream.")
PY
for f in /tmp/results/S1.json /tmp/results/S2.json /tmp/results/C1_p1_world_v0_5150.json; do
    python /app/push.py "$f" || true
done
echo "########## DONE ##########"
