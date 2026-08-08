"""Train one LoRA per run — with prompt masking, and an A/B against not masking.

**The bug this fixes.** The previous version tokenized `prompt + completion` and
used `DataCollatorForLanguageModeling(mlm=False)`, which computes loss over every
token. Measured on a representative step, that put **68% of the gradient on text
identical across all runs** — the system block (55%) and the shared world's room
and observation text (13%) — against **6% on the action**, which is the only
thing the distillation is supposed to be learning.

Training every run predominantly on the same shared tokens should pull them
together, and the measured distilled component was indeed negative (−0.018,
i.e. training made the eight runs *more* similar). So "distillation is a
convergence pressure" may be an artefact of this trainer rather than a property
of self-distillation. `--mask` settles it.

The same mistake was caught and fixed in the MLX trainer with `--mask-prompt`
and then not carried across to the CUDA path.

Safe to keep all runs in one process: unlike vLLM, peft/transformers release GPU
memory on `del` + `empty_cache()` because no child process survives.

Usage:
    python train_many.py MODEL N_RUNS [--mask] [--suffix S]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))
IGNORE = -100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_dataset(tok, rows, mask_prompt: bool):
    """Tokenize, and when masking set prompt tokens to the ignore index.

    Returns the dataset plus the realized masked fraction, so the fix is
    verifiable in the log rather than assumed.
    """
    from datasets import Dataset

    examples, masked_tok, total_tok = [], 0, 0
    for r in rows:
        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        full_text = r["prompt"] + r["completion"] + tok.eos_token
        ids = tok(full_text, truncation=True, max_length=1024,
                  add_special_tokens=False)["input_ids"]
        labels = list(ids)
        if mask_prompt:
            cut = min(len(prompt_ids), len(ids))
            for i in range(cut):
                labels[i] = IGNORE
            masked_tok += cut
        total_tok += len(ids)
        examples.append({"input_ids": ids,
                         "attention_mask": [1] * len(ids),
                         "labels": labels})
    frac = masked_tok / total_tok if total_tok else 0.0
    return Dataset.from_list(examples), frac


def train_one(model_id, rows, out_dir: Path, mask_prompt: bool) -> float:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds, frac = build_dataset(tok, rows, mask_prompt)

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
                               bf16=True, disable_tqdm=True,
                               remove_unused_columns=False),
        # Seq2Seq collator pads `labels` with -100 rather than overwriting them,
        # which the plain LM collator would do.
        data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                             label_pad_token_id=IGNORE),
    ).train()
    m.save_pretrained(str(out_dir))
    del m
    torch.cuda.empty_cache()
    return frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", type=int)
    ap.add_argument("--mask", action="store_true", help="mask prompt tokens from the loss")
    ap.add_argument("--suffix", default="", help="adapter dir suffix")
    ap.add_argument("--prefix", default="", help="corpus/adapter prefix, e.g. assigned_")
    args = ap.parse_args()

    log(f"prompt masking: {'ON' if args.mask else 'OFF'} | suffix '{args.suffix}'")
    fracs = []
    for i in range(args.runs):
        src = WORK / f"{args.prefix}kept_r{i}.jsonl"
        dst = WORK / f"{args.prefix}adapter_r{i}{args.suffix}"
        if dst.exists():
            log(f"run {i}: exists, skipping"); continue
        if not src.exists():
            log(f"run {i}: no corpus, skipping"); continue
        rows = [json.loads(l) for l in src.read_text().splitlines()]
        if not rows:
            log(f"run {i}: empty corpus, skipping"); continue
        frac = train_one(args.model, rows, dst, args.mask)
        fracs.append(frac)
        log(f"run {i}: {len(rows)} examples, {frac:.0%} of tokens masked -> {dst.name}")

    if fracs:
        mean = sum(fracs) / len(fracs)
        log(f"mean masked fraction {mean:.1%}")
        # With masking on, the vast majority of tokens are prompt; if this comes
        # back near zero the mask silently did nothing.
        if args.mask and mean < 0.5:
            log("WARNING: masking was requested but under half of tokens were "
                "masked — check the prompt/completion split")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
