#!/usr/bin/env bash
# CHECK 1 of the cohort-feasibility gate. Two questions, in this order:
#
#   A. Is --no-narrate actually additive? Re-run one COMMITTED fidelity eval
#      with the new default-on code and diff it against the stored file.
#   B. What does a base checkpoint cost with narration off, against an instruct
#      checkpoint of the same size?
#
# **A gates B, and A gates the whole gate.** If the scored path is not
# bit-for-bit, the change is not as additive as it looks, and finding that here
# is worth more than any cost number — after a cohort sweep it would have
# contaminated everything downstream.
#
# **VLLM_BATCH_INVARIANT=1 is not optional for A.** B2 measured this stack as
# nondeterministic without it (8/12 runs reproducible, 34% raw command
# divergence). A before/after diff taken bare would pass or fail on ambient
# sampling noise rather than on the code change, which is not evidence. The
# baseline file was itself produced under the flag, in gpu_job22.
#
# The baseline target has 11 of 12 runs — one failed in the original sweep. A
# faithful reproduction must reproduce the FAILURE too, not just the successes.
#
# Each model is served alone and timed. No batching of unvalidated cost
# profiles: that is what cost $4.25 for zero results.
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
export VLLM_BATCH_INVARIANT=1            # load-bearing for check A; see header
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }

python - <<'PY' || exit 1
import sys; sys.path.insert(0,"/app")
from seahaven.fidelity.runner import STEP_SCHEDULES, _steps_for, run_fidelity
import inspect
sched = STEP_SCHEDULES["v1"]
assert run_fidelity.__kwdefaults__ is not None
assert inspect.signature(run_fidelity).parameters["narrate"].default is True, \
    "narrate must default True or the regression is meaningless"
for runs, steps, what in ((12, 30, "regression + timed sweeps"),):
    assert runs == len(sched), f"{what}: runs={runs} vs {len(sched)}-entry schedule"
    lens = [_steps_for(i, steps, sched) for i in range(runs)]
    print(f"  {what}: {runs} runs, steps {min(lens)}-{max(lens)}, {sum(lens)} commands max",
          flush=True)
print("  narrate defaults to True — scored path unchanged", flush=True)
PY

serve() {   # $1 = repo
    python -m vllm.entrypoints.openai.api_server \
        --model "$1" --port $PORT --host 127.0.0.1 \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 90); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then echo "  server ready after ${i}0s"; return 0; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED:"; tail -30 /tmp/vllm.log; return 1; }
        [ $((i % 6)) -eq 0 ] && echo "  ...waiting ${i}0s"
        sleep 10
    done
    return 1
}
teardown() {
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 10; reap_gpu || { echo "GPU DID NOT DRAIN"; exit 1; }
}

############################################################################
echo; echo "############ A. BYTE-IDENTITY REGRESSION ############"
echo "target: results/vp_AlibabaSmall_p1_world_v0_5150.json (11/12 runs, one failed)"
M="Qwen/Qwen2.5-0.5B-Instruct"
if serve "$M"; then
    # Arguments read off the baseline's own meta block, not retyped from memory.
    python -m seahaven.fidelity.cli eval \
        --model "http://127.0.0.1:$PORT/v1" --served-name "$M" \
        --allow-regex-judge --runs 12 --steps 30 --step-schedule v1 \
        --seed 5150 --world world_v0 --phrasing p1 \
        --output "$R/regression.json" 2>&1 | tail -5
    python - <<'PY'
import json
new = json.load(open("/tmp/results/regression.json"))
old = json.load(open("/app/baseline/vp_AlibabaSmall_p1_world_v0_5150.json"))
print()
if new == old:
    print("  IDENTICAL — every byte, including meta. --no-narrate is additive.")
    raise SystemExit(0)
print("  NOT byte-identical. Localising:")
for k in sorted(set(new) | set(old)):
    if new.get(k) != old.get(k):
        print(f"    differs: {k}")
oc = [r.get("commands") for r in old["runs"]]
nc = [r.get("commands") for r in new["runs"]]
print(f"  runs: old {len(oc)}  new {len(nc)}")
print(f"  COMMAND STREAM identical: {oc == nc}")
print(f"  narratives identical: "
      f"{[r.get('narrative') for r in old['runs']] == [r.get('narrative') for r in new['runs']]}")
print(f"  score identical: {old.get('score') == new.get('score')}")
print("\n  A command-stream match with differing narratives would mean sampling")
print("  nondeterminism in the narration call only. A command-stream MISMATCH")
print("  means the change is not additive, or the flag is not binding.")
PY
    push "$R/regression.json" || true
    teardown
else
    echo "  SKIPPED — server never became ready"; teardown
fi

############################################################################
echo; echo "############ B. BASE vs INSTRUCT COST, narration off ############"
for ENTRY in "instruct|Qwen/Qwen3-8B" "base|Qwen/Qwen3-8B-Base"; do
    KIND="${ENTRY%%|*}"; M="${ENTRY##*|}"
    echo; echo "---- $KIND: $M ----"
    if serve "$M"; then
        T0=$(date +%s)
        # Time-boxed: a stall is DATA, not a reason to burn the budget. 15 min
        # is ~6x the instruct expectation; exceeding it is the finding.
        timeout 900 python -m seahaven.fidelity.cli eval \
            --model "http://127.0.0.1:$PORT/v1" --served-name "$M" \
            --allow-regex-judge --no-narrate --runs 12 --steps 30 \
            --step-schedule v1 --seed 5150 --world world_v0 --phrasing p1 \
            --output "$R/cost_${KIND}.json" 2>&1 | tail -4
        RC=$?; T1=$(date +%s)
        if [ $RC -eq 124 ]; then
            echo "  TIMEOUT at 900s — $KIND ($M) STALLED even with narration off."
            echo "  Recorded as data: this lineage is unaffordable at cohort scale."
            echo "  ELAPSED_${KIND}=STALLED"
        else
            echo "  ELAPSED_${KIND}=$((T1-T0))s"
            push "$R/cost_${KIND}.json" || true
        fi
        teardown
    else
        echo "  SKIPPED $KIND — server never became ready"; echo "  ELAPSED_${KIND}=NO_SERVER"; teardown
    fi
done

echo; echo "############ CHECK 1 SUMMARY ############"
python - <<'PY'
import json, os
for kind in ("instruct", "base"):
    p = f"/tmp/results/cost_{kind}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        n = sum(len(r.get("commands", [])) for r in d["runs"])
        print(f"  {kind:<9} {d['n_runs_completed']}/{d['n_runs_requested']} runs, "
              f"{n} commands, narrate={d.get('narrate')}")
    else:
        print(f"  {kind:<9} no result file (stalled or skipped)")
print("\n  Multiplier = ELAPSED_base / ELAPSED_instruct, read from the lines above.")
print("  Gate: base within ~2x instruct. Beyond ~10x, the cohort is instruct-only")
print("  and the verdict records that as GO-SCOPED, not NO-GO.")
PY
echo "########## DONE ##########"
