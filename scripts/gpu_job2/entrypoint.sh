#!/usr/bin/env bash
# A/B on the prompt-masking bug, same corpus both sides.
#
# The previous trainer computed loss over every token, putting ~68% of the
# gradient on text identical across all runs (system block + shared world) and
# only ~6% on the action. The measured distilled component was negative
# (-0.018): training made the eight runs MORE similar. That is exactly what
# training everyone on the same tokens would do, so it may be an artefact rather
# than a property of self-distillation.
#
# One corpus, two trainings, two scorings. Nothing else differs.
set -uo pipefail

R=/tmp/results; W=/tmp/a1b; mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"
RUNS=8

push() { python /app/push.py "$1"; }
source /app/lib.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== install ==="
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -3

NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
if [ -n "${NVCC_DIR:-}" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/app
export VLLM_BATCH_INVARIANT=1

run_phase 25m "1/5 collect: 4 campaigns x 8 runs (one corpus for both arms)" \
    python /app/accumulate.py collect --model "$MODEL" \
        --runs $RUNS --steps 14 --campaigns 4 --out "$R/mask_collect.json" || exit 1
push "$R/mask_collect.json"

run_phase 25m "2/5 train UNMASKED (reproduces the bug)" \
    python /app/train_many.py "$MODEL" $RUNS --suffix _unmasked || exit 1

run_phase 25m "3/5 train MASKED (the fix)" \
    python /app/train_many.py "$MODEL" $RUNS --mask --suffix _masked || exit 1

run_phase 12m "4/5 score UNMASKED" \
    python /app/accumulate.py score --model "$MODEL" --adapter-suffix _unmasked \
        --collect "$R/mask_collect.json" --out "$R/behav_unmasked.json" || exit 1
push "$R/behav_unmasked.json"

run_phase 12m "5/5 score MASKED" \
    python /app/accumulate.py score --model "$MODEL" --adapter-suffix _masked \
        --collect "$R/mask_collect.json" --out "$R/behav_masked.json" || exit 1
push "$R/behav_masked.json"

echo
echo "########## REPORT ##########"
python - <<'PY'
import json
u=json.load(open("/tmp/results/behav_unmasked.json"))
m=json.load(open("/tmp/results/behav_masked.json"))
print(f"{'':22s} {'UNMASKED':>12s} {'MASKED':>12s}")
for k in ("narrative_spread_final","behavioural_spread_story_only",
          "behavioural_spread_trained","distilled_component"):
    print(f"{k:22s} {u.get(k):>12} {m.get(k):>12}")
print()
du, dm = u["distilled_component"], m["distilled_component"]
print(f"distilled component: {du:+.5f} -> {dm:+.5f}")
print("VERDICT:", "sign flipped — the negative was a masking artefact"
      if du < 0 <= dm else
      "still negative — distillation really is a convergence pressure"
      if dm < 0 else "inconclusive")
PY
