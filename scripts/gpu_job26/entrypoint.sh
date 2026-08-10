#!/usr/bin/env bash
# CHECK 3 — access/loop attrition AND the per-model determinism map, in one pass.
#
# TRAP 35 made these one job rather than two. The loop test asks "can this model
# hold a parseable action loop"; the determinism probe asks "does it reproduce".
# Both come from the same repeats, so the cohort's determinism map is a
# by-product of work already budgeted.
#
# **The probe is not bind/doesn't-bind.** Stage 4 has to partial a NOISE FLOOR
# out of every behavioural distance, and a yes/no cannot be subtracted. So each
# model gets N=4 repeats of one eval under identical conditions, and the floor is
# reported as numbers in the quantity the program actually uses — command-level
# adherence and the verb distribution, not fidelity.
#
# **Read against SIZE.** Two points so far say Llama-8B binds and Qwen-0.5B does
# not, so the obvious hypothesis is that nondeterminism tracks size. That matters
# because the small end is where every flag lived and where the program expects
# its interesting variance: if the hypothesis holds, precision is worst exactly
# where the signal is. Size travels with every row so the regression can be run
# before the gate closes.
#
# Narration off throughout: 9s per eval instead of minutes, which is what makes
# 4 repeats per model affordable at all.
#
# Every model is served ALONE and its results pushed before the next one loads,
# so a late failure costs one model rather than the run.
set -uo pipefail
R=/tmp/results; mkdir -p "$R"
export A1B_REPO="cloudronin/seahaven-a1b-results"
PORT=8000
source /app/lib.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q vllm transformers textworld jericho 2>&1 | tail -3
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=/app PYTHONUNBUFFERED=1
export VLLM_BATCH_INVARIANT=1     # still mandatory; no longer assumed sufficient
python -c "import vllm; print('  vllm', vllm.__version__)" || { echo "INSTALL FAILED"; exit 1; }

REPEATS=4
mapfile -t ROWS < <(python - <<'PY'
import json
d = json.load(open("/app/candidates.json"))
for r in d["rows"]:
    if r["status"].startswith("available"):
        print(f"{r['family']}|{r['size']}|{r['kind']}|{r['repo']}")
PY
)
echo "  ${#ROWS[@]} candidates to probe, ${REPEATS} repeats each"

for ROW in "${ROWS[@]}"; do
    IFS='|' read -r FAM SIZE KIND REPO <<< "$ROW"
    TAG=$(echo "$REPO" | tr '/.' '__')
    echo; echo "######## $REPO  ($FAM ${SIZE}B $KIND) ########"
    python -m vllm.entrypoints.openai.api_server --model "$REPO" --port $PORT \
        --host 127.0.0.1 --gpu-memory-utilization 0.85 --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    SERVER_PID=$!
    READY=0
    for i in $(seq 1 60); do
        if python - <<PROBE >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=2).read()
PROBE
        then READY=1; echo "  ready ${i}0s"; break; fi
        kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -8 /tmp/vllm.log; break; }
        sleep 10
    done

    if [ "$READY" = "1" ]; then
        OK=1
        for REP in $(seq 1 $REPEATS); do
            timeout 600 python -m seahaven.fidelity.cli eval \
                --model "http://127.0.0.1:$PORT/v1" --served-name "$REPO" \
                --allow-regex-judge --no-narrate --runs 12 --steps 30 \
                --step-schedule v1 --seed 5150 --world world_v0 --phrasing p1 \
                --output "$R/probe_${TAG}_rep${REP}.json" > /dev/null 2>&1 || OK=0
            # Loop test is repeat 1: too few commands and the remaining repeats
            # are not worth paying for.
            if [ "$REP" = "1" ]; then
                N=$(python - <<PY
import json
try:
    d=json.load(open("$R/probe_${TAG}_rep1.json"))
    print(sum(len(r.get("commands",[])) for r in d["runs"]))
except Exception: print(0)
PY
)
                echo "  loop test: $N commands"
                if [ "$N" -lt 50 ]; then
                    echo "  EXCLUDED $REPO — loop test failed ($N commands)"; OK=0; break
                fi
            fi
        done
        if [ "$OK" = "1" ]; then
            python - <<PY
import json, itertools, statistics as st, sys
sys.path.insert(0, "/app")
from seahaven.fidelity.adherence import classify
reps = []
for i in range(1, $REPEATS + 1):
    d = json.load(open(f"$R/probe_${TAG}_rep{i}.json"))
    cmds = [r.get("commands", []) for r in d["runs"]]
    flat = [c["command"] for run in cmds for c in run]
    bad = sum(classify(c) == "violation" for c in flat)
    verbs = {}
    for c in flat:
        v = (c.split() or [""])[0].lower()
        verbs[v] = verbs.get(v, 0) + 1
    tot = sum(verbs.values()) or 1
    reps.append({"cmds": cmds,
                 "adh": 100.0 * (1 - bad / len(flat)) if flat else float("nan"),
                 "vd": {k: v / tot for k, v in verbs.items()}})
pairs = list(itertools.combinations(range(len(reps)), 2))
ident = st.mean(
    sum(1 for a, b in zip(reps[i]["cmds"], reps[j]["cmds"]) if a == b) /
    max(1, min(len(reps[i]["cmds"]), len(reps[j]["cmds"]))) for i, j in pairs)
def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))
out = {"repo": "$REPO", "family": "$FAM", "size_b": float("$SIZE"),
       "kind": "$KIND", "repeats": len(reps),
       "adherence": [r["adh"] for r in reps],
       "adherence_sd": st.pstdev([r["adh"] for r in reps]),
       "runs_identical_frac": ident,
       "verb_tv_mean": st.mean(tv(reps[i]["vd"], reps[j]["vd"]) for i, j in pairs),
       "deterministic": ident == 1.0}
json.dump(out, open("$R/noise_${TAG}.json", "w"), indent=1)
print(f"  NOISE {out['repo']} size={out['size_b']}B {out['kind']}: "
      f"identical={out['runs_identical_frac']:.2f} "
      f"adh_sd={out['adherence_sd']:.3f} verbTV={out['verb_tv_mean']:.4f} "
      f"{'DETERMINISTIC' if out['deterministic'] else 'noisy'}")
PY
            python /app/push.py "$R/noise_${TAG}.json" || true
        fi
    else
        echo "  EXCLUDED $REPO — server never ready"
    fi
    kill -9 $SERVER_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 8; reap_gpu || true
done
echo; echo "########## DONE ##########"; ls "$R"/noise_*.json 2>/dev/null | wc -l
