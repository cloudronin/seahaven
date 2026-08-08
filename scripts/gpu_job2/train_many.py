"""Train one LoRA per run, sequentially, in a single process.

Safe to keep in one process — unlike vLLM, peft/transformers models release
their GPU memory on `del` + `empty_cache()` because there is no surviving child
process holding the device. The whole reason vLLM phases had to be split is
absent here.

One adapter per run, trained on that run's own selected episodes pooled across
campaigns. Never pooled across runs: that isolation is the property the
divergence claim rests on, and it is asserted per file rather than assumed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))


def train_one(model_id: str, rows: list[dict], out_dir: Path, log) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([
        dict(tok(r["prompt"] + r["completion"] + tok.eos_token,
                 truncation=True, max_length=1024)) for r in rows])

    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             device_map="cuda")
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    Trainer(
        model=m, train_dataset=ds,
        args=TrainingArguments(output_dir="/tmp/_tr", per_device_train_batch_size=4,
                               num_train_epochs=3, learning_rate=1e-4,
                               logging_steps=1000, report_to=[], save_strategy="no",
                               bf16=True, disable_tqdm=True),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    m.save_pretrained(str(out_dir))
    del m
    torch.cuda.empty_cache()


def main() -> int:
    import time

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    model_id = sys.argv[1]
    n = int(sys.argv[2])
    for i in range(n):
        src = WORK / f"kept_r{i}.jsonl"
        dst = WORK / f"adapter_r{i}"
        if dst.exists():
            log(f"run {i}: adapter exists, skipping"); continue
        if not src.exists():
            log(f"run {i}: no corpus at {src}, skipping"); continue
        rows = [json.loads(l) for l in src.read_text().splitlines()]
        if not rows:
            log(f"run {i}: empty corpus, skipping"); continue
        log(f"run {i}: training on {len(rows)} self-selected examples")
        train_one(model_id, rows, dst, log)
        log(f"run {i}: saved {dst}")

    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
