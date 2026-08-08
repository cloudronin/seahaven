#!/usr/bin/env bash
# A1b at scale. Each phase is a separate process: vLLM's EngineCore child does
# not release GPU memory on `del llm`, and two earlier jobs died with
# "Free memory on device cuda:0 (26.13/139.8 GiB)" trying to reuse one process.
set -uo pipefail

R=/tmp/results
W=/tmp/a1b
mkdir -p "$R" "$W"
export A1B_WORK="$W"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

echo "=== install ==="
pip install --no-cache-dir -q uv
uv pip install --system --no-cache -q \
    vllm peft transformers datasets accelerate nvidia-cuda-nvcc-cu12 textworld 2>&1 | tail -4

NVCC_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
if [ -n "${NVCC_DIR:-}" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/app

# ON throughout: it does not suppress seed-driven variance, only batch-composition
# and float-atomic noise. Excluding those is what makes "identical world,
# different seed" a controlled contrast. Measured cost on this stack: 3.3x.
export VLLM_BATCH_INVARIANT=1

python -c "import vllm, torch, jericho; print('vllm', vllm.__version__, '| torch', torch.__version__)"
python -c "
import sys; sys.path.insert(0,'/app')
import seahaven_world as sw
w=sw.open_world('/app/world_v0.z8'); o,h=w.reset()
print('world ok |', o.room, '| facts', len(h.facts)); w.close()"

# Olmo-3-7B-Instruct produced zero output for 25 minutes on vLLM 0.26
# (likely an unsupported architecture) and blocked stage 2, which is the
# stage that matters. Qwen3-8B passed stage 1 twice at 20/20. The
# family-generalisation question is real but secondary; park it.
MODELS=("Qwen/Qwen3-8B")
WINNER=""

echo
echo "########## STAGE 1 — action variance ##########"
for M in "${MODELS[@]}"; do
    SAFE=$(echo "$M" | tr '/' '_')
    echo "--- $M ---"
    python /app/a1b_stage.py variance --model "$M" \
        --episodes 20 --steps 16 --out "$R/variance_$SAFE.json" || continue
    N=$(python -c "
import json,sys
d=json.load(open('$R/variance_$SAFE.json'))
print(list(d.values())[0]['distinct_command_sequences'])" 2>/dev/null || echo 0)
    echo "    distinct sequences: $N / 20"
    if [ -z "$WINNER" ] && [ "$N" -ge 6 ]; then WINNER="$M"; fi
done

if [ -z "$WINNER" ]; then
    echo "STAGE 2 SKIPPED — no checkpoint reached 6 distinct command sequences."
    echo "Sampling noise does not reach the action channel at this scale."
else
    echo
    echo "########## STAGE 2 — gate on $WINNER ##########"
    python /app/a1b_stage.py collect --model "$WINNER" \
        --episodes 16 --steps 24 --out "$W/collect.json" \
        && cp "$W/collect.json" "$R/collect.json"
    python /app/a1b_stage.py train --model "$WINNER" --out "$R/train.json"
    python /app/a1b_stage.py after --model "$WINNER" --out "$R/gate.json"
fi

echo
echo "########## REPORT ##########"
for f in "$R"/*.json; do echo "--- $(basename "$f") ---"; cat "$f"; echo; done
