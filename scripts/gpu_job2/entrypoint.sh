#!/usr/bin/env bash
# A1b stage 2 only — the gate. Stage 1 is already answered (Qwen3-8B: 20/20
# distinct command sequences, replicated three times), so this goes straight to
# the open question: does LoRA on self-generated, agent-selected data move the
# fingerprint at 8B?
#
# Two things learned the expensive way are baked in:
#   - each phase is its own process, because vLLM's EngineCore child holds GPU
#     memory that `del llm` cannot release;
#   - every phase pushes its output to a Hub dataset the moment it finishes, so
#     a stalled log costs only the phase in flight rather than the whole run.
set -uo pipefail

R=/tmp/results
W=/tmp/a1b
mkdir -p "$R" "$W"
export A1B_WORK="$W"
export A1B_REPO="cloudronin/seahaven-a1b-results"
MODEL="Qwen/Qwen3-8B"

push() { python /app/push.py "$1"; }

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
# Not a tuning flag: without it, LoRA output is nondeterministic (4 distinct
# outputs across 16 identical requests) and "different seed" stops being a
# controlled contrast. Costs 3.3x on this stack.
export VLLM_BATCH_INVARIANT=1

python -c "import vllm, torch, jericho; print('vllm', vllm.__version__, '| torch', torch.__version__)"
python -c "
import sys; sys.path.insert(0,'/app')
import seahaven_world as sw
w=sw.open_world('/app/world_v0.z8'); o,h=w.reset()
print('world ok |', o.room, '| facts', len(h.facts)); w.close()"

echo
echo "########## 1/3 collect — corpus, agent selection, before-fingerprint ##########"
python /app/a1b_stage.py collect --model "$MODEL" \
    --episodes 16 --steps 24 --out "$W/collect.json"
cp "$W/collect.json" "$R/collect.json" 2>/dev/null
cp "$W/kept.jsonl" "$R/kept.jsonl" 2>/dev/null
push "$R/collect.json"; push "$R/kept.jsonl"

echo
echo "########## 2/3 train — LoRA on the self-selected corpus ##########"
python /app/a1b_stage.py train --model "$MODEL" --out "$R/train.json"
push "$R/train.json"

echo
echo "########## 3/3 after — fingerprint with adapter, distance, verdict ##########"
python /app/a1b_stage.py after --model "$MODEL" --out "$R/gate.json"
push "$R/gate.json"

echo
echo "########## REPORT ##########"
for f in "$R"/*.json; do echo "--- $(basename "$f") ---"; cat "$f"; echo; done
