#!/usr/bin/env bash
# A1b at scale. Same image and install pattern as the A2/A3 job, which ran clean.
set -uo pipefail

RESULTS=/tmp/results
mkdir -p "$RESULTS"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

echo "=== install ==="
pip install --no-cache-dir -q uv
# jericho compiles libfrotz from C at install time; python:3.12 (not -slim)
# carries a toolchain, so this works without extra apt packages.
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 \
    textworld 2>&1 | tail -5

NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
if [ -n "${NVCC_DIR:-}" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

python -c "import vllm, torch, jericho; print('vllm', vllm.__version__, '| torch', torch.__version__)"
python -c "
import sys; sys.path.insert(0,'/app')
import seahaven_world as sw
w = sw.open_world('/app/world_v0.z8'); o,h = w.reset()
print('world ok |', o.room, '| facts', len(h.facts)); w.close()"

# ON throughout. It does not suppress seed-driven variance — it suppresses
# batch-composition and float-atomic noise, which is what must be excluded for
# 'identical world, different seed' to be a controlled contrast. Measured cost
# on this stack: 3.3x.
export VLLM_BATCH_INVARIANT=1
export PYTHONPATH=/app

echo
echo "########## A1b: stage 1 (action variance) -> stage 2 (gate) ##########"
python /app/a1b_full.py \
    --models Qwen/Qwen3-8B allenai/Olmo-3-7B-Instruct \
    --episodes 20 --steps 16 \
    --gate-episodes 16 --gate-steps 24 \
    --min-distinct 6 \
    --out "$RESULTS/a1b_report.json"
RC=$?

echo
echo "########## REPORT ##########"
cat "$RESULTS/a1b_report.json" 2>/dev/null || echo "(no report written)"
exit $RC
