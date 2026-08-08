#!/usr/bin/env bash
# HF Jobs entrypoint.
#
# Image is plain `python:3.12` rather than `vllm/vllm-openai`, because that image
# sets ENTRYPOINT to the API server and `hf jobs run` has no --entrypoint flag —
# the command would become arguments to the server instead of running.
#
# Everything here is on the clock, so: install with uv, fail fast, and print
# results to stdout so they survive in the job log even if a later stage dies.
set -uo pipefail

RESULTS=/tmp/results
mkdir -p "$RESULTS"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

echo "=== installing (uv) ==="
pip install --no-cache-dir -q uv
# vllm pulls its own torch build; letting it resolve avoids a version conflict.
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate 2>&1 | tail -5
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# The batch-invariant flag is read at engine init, so measuring with and without
# it needs two processes. Adapters are trained once in run 1 and reused in run 2.
echo
echo "########## run 1of2: VLLM_BATCH_INVARIANT=1  (the required setting) ##########"
VLLM_BATCH_INVARIANT=1 python /app/a2a3_canary.py \
    --out "$RESULTS/report_invariant.json" --adapters /tmp/adapters
RC1=$?

echo
echo "########## run 2of2: default flags  (to price the flag) ##########"
python /app/a2a3_canary.py \
    --out "$RESULTS/report_default.json" --adapters /tmp/adapters --skip-train
RC2=$?

echo
echo "########## SUMMARY ##########"
python - <<'PY'
import json, pathlib
res = pathlib.Path("/tmp/results")
def load(n):
    p = res / n
    return json.loads(p.read_text()) if p.exists() else {}
inv, dfl = load("report_invariant.json"), load("report_default.json")
a3i, a3d = inv.get("a3_determinism", {}), dfl.get("a3_determinism", {})
canary = inv.get("canary_kv_cache", {})

s = {
    "gpu": inv.get("gpu"),
    "compute_capability": inv.get("capability"),
    "vllm": inv.get("vllm_version"),
    "CANARY_stale_kv_reproduced": canary.get("stale_kv_reproduced"),
    "CANARY_verdict": canary.get("verdict"),
    "CANARY_answers": {
        "v1": canary.get("v1_answer"),
        "after_same_name_reload": canary.get("after_same_name_reload"),
        "after_versioned_name": canary.get("after_versioned_name"),
    },
    "A2_lora_plus_structured_composes": inv.get("a2_lora_plus_structured", {}).get("composes"),
    "A3_deterministic_WITH_flag": a3i.get("deterministic"),
    "A3_deterministic_WITHOUT_flag": a3d.get("deterministic"),
    "A3_distinct_outputs_with": a3i.get("distinct_outputs"),
    "A3_distinct_outputs_without": a3d.get("distinct_outputs"),
    "latency_with_flag_s": a3i.get("mean_batch_latency_s"),
    "latency_without_flag_s": a3d.get("mean_batch_latency_s"),
}
if a3i.get("mean_batch_latency_s") and a3d.get("mean_batch_latency_s"):
    s["batch_invariant_overhead_x"] = round(
        a3i["mean_batch_latency_s"] / a3d["mean_batch_latency_s"], 3)

print(json.dumps(s, indent=2))
print()
print("### FULL REPORTS ###")
print(json.dumps({"invariant": inv, "default": dfl}, indent=2))
PY

exit $(( RC1 != 0 ? RC1 : RC2 ))
