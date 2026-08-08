"""Chained LoRA training, with and without a KL term to the run's own past self.

**Why a KL term.** Distillation erased character in every configuration tried:
−8% on corpora distinct by construction, −22% on self-authored ones, surviving
both prompt masking and identity framing. Hard-target self-imitation discards
distribution tails by construction, so each round narrows.

**Why the reference choice is the whole point.**

    reference = shared base    every run anchored to the SAME point → runs stay
                               near base and therefore near each other →
                               guarantees the convergence we are trying to stop
    reference = own previous   each run anchored to its OWN history → collapse
    campaign's adapter         resisted while runs stay free to differ

**Three fixes over the first attempt, which returned an uninterpretable null.**

1. *No KL at campaign 1.* The first version applied KL from campaign 1, where
   the only available reference is the shared base — so half the training did
   exactly the thing the docstring warned against. Campaign 1 is now plain SFT
   for both variants, **trained once and shared**, so the two chains are
   identical up to the point where they differ. KL applies from campaign 2 on,
   where the reference is run-specific.

2. *Selecting nothing no longer kills the run.* Agents kept zero episodes in 3
   of 12 run-campaigns, which broke the chain and cut the sample from 6 runs to
   3. That is backwards: the spec treats selecting nothing as data, and the
   faithful reading is "no update this campaign", not "this run is over". The
   run now carries its previous adapter forward unchanged.

3. *More campaigns and more runs*, so run-specific anchoring has room to
   dominate and the sample survives dropout.

**Known limitation.** Rollouts use the base model plus the story; adapters are
never loaded during play, only at scoring. The spec's silent weight update has
therefore never influenced subsequent behaviour — campaigns are chained in story
and weights but not in lived experience.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))
IGNORE = -100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_dataset(tok, rows):
    """Prompt tokens masked out of the loss; only the action is a target."""
    from datasets import Dataset

    ex = []
    for r in rows:
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        ids = tok(r["prompt"] + r["completion"] + tok.eos_token,
                  truncation=True, max_length=1024,
                  add_special_tokens=False)["input_ids"]
        labels = list(ids)
        for i in range(min(len(p_ids), len(ids))):
            labels[i] = IGNORE
        ex.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                   "labels": labels})
    return Dataset.from_list(ex)


def make_trainer_cls(beta: float, ref_model):
    """Loss = CE(action) + beta * KL(student || reference).

    The reference is a SEPARATE frozen model, not a second adapter on the same
    one. Loading the reference alongside the trainable adapter and switching
    with `set_adapter` left the trainable adapter frozen and the loss arrived
    with no gradient path. Two 8B copies is 32 GB of 141 GB.
    """
    from transformers import Trainer

    class KLTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=IGNORE)

            if beta <= 0 or ref_model is None:
                loss = ce
            else:
                with torch.no_grad():
                    ref_logits = ref_model(**inputs).logits
                # KL only where a target exists. Regularising prompt positions
                # would pull toward reproducing the world's text — the mistake
                # the prompt mask exists to prevent.
                mask = (shift_labels != IGNORE)
                if mask.any():
                    sl = F.log_softmax(shift_logits[mask].float(), dim=-1)
                    rl = F.log_softmax(
                        ref_logits[..., :-1, :].contiguous()[mask].float(), dim=-1)
                    kl = F.kl_div(sl, rl, log_target=True, reduction="batchmean")
                else:
                    kl = torch.zeros((), device=ce.device)
                loss = ce + beta * kl

            if not loss.requires_grad:
                raise RuntimeError(
                    "loss has no grad_fn — the trainable adapter is frozen.")
            return (loss, out) if return_outputs else loss

    return KLTrainer


def train_campaign(model_id, rows, out_dir: Path, resume_from: Path | None,
                   ref_from: Path | None, beta: float):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_dataset(tok, rows)

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    if resume_from is not None:
        # is_trainable=True is load-bearing: without it the adapter loads frozen
        # and training silently no-ops.
        model = PeftModel.from_pretrained(base, str(resume_from), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    ref_model = None
    if beta > 0 and ref_from is not None:
        ref_base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        ref_model = PeftModel.from_pretrained(ref_base, str(ref_from))
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    make_trainer_cls(beta, ref_model)(
        model=model, train_dataset=ds,
        args=TrainingArguments(output_dir="/tmp/_tr", per_device_train_batch_size=2,
                               num_train_epochs=3, learning_rate=1e-4,
                               logging_steps=1000, report_to=[], save_strategy="no",
                               bf16=True, disable_tqdm=True,
                               remove_unused_columns=False),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                             label_pad_token_id=IGNORE)).train()

    model.save_pretrained(str(out_dir))
    del model, base, ref_model
    torch.cuda.empty_cache()


def corpus(c: int, i: int):
    p = WORK / f"c{c}_kept_r{i}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--campaigns", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", type=int, default=1,
                    help="first campaign to train (2 when resuming a shared c1)")
    ap.add_argument("--init-tag", default=None,
                    help="tag whose campaign-1 adapters to start from")
    args = ap.parse_args()

    log(f"variant '{args.tag}' | beta={args.beta} | campaigns "
        f"{args.start}..{args.campaigns}"
        + (f" | init from '{args.init_tag}' c1" if args.init_tag else ""))

    manifest = {}
    for i in range(args.runs):
        prev = None
        if args.start > 1 and args.init_tag:
            seed_adapter = WORK / f"{args.init_tag}_adapter_r{i}_c1"
            if seed_adapter.exists():
                prev = seed_adapter
            else:
                log(f"run {i}: no shared c1 adapter, skipping run"); continue

        for c in range(args.start, args.campaigns + 1):
            dst = WORK / f"{args.tag}_adapter_r{i}_c{c}"
            rows = corpus(c, i)

            if not rows:
                # Selecting nothing means no update this campaign — not the end
                # of the run. Carry the previous adapter forward so the run stays
                # in the sample.
                if prev is not None:
                    log(f"run {i} c{c}: kept nothing, carrying {prev.name} forward")
                    if not dst.exists():
                        shutil.copytree(prev, dst)
                    prev = dst
                else:
                    log(f"run {i} c{c}: kept nothing and no prior adapter, skipping")
                continue

            if dst.exists():
                prev = dst; continue

            # beta applies only when a run-specific reference exists. Campaign 1
            # has only the shared base, and anchoring every run there is the
            # convergence trap this design exists to avoid.
            use_beta = args.beta if prev is not None else 0.0
            log(f"run {i} c{c}: {len(rows)} examples, resume={prev is not None}, "
                f"beta={use_beta}, ref={'own_c%d' % (c-1) if prev else 'none'}")
            train_campaign(args.model, rows, dst, resume_from=prev,
                           ref_from=prev, beta=use_beta)
            prev = dst

        if prev is not None:
            manifest[str(i)] = str(prev)

    (WORK / f"{args.tag}_final.json").write_text(json.dumps(manifest, indent=2))
    log(f"final adapters: {len(manifest)}/{args.runs} runs -> {args.tag}_final.json")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
