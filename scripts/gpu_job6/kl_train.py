"""Chained LoRA training, with and without a KL term to the run's own past self.

**Why a KL term, and why the reference choice is the whole point.**

Distillation has erased character in every configuration tried: −8% on corpora
distinct by construction, −22% on self-authored ones, and the sign survived both
prompt masking and identity framing. The mechanism is hard-target self-imitation
— sampling actions, keeping some, fitting to them — which discards distribution
tails by construction. Each round narrows.

A KL term against a frozen reference is the standard fix (RLHF and DPO both use
it). No teacher model is needed: the reference is this model at an earlier point
in time, and with LoRA it costs no extra weights.

But *which* reference decides whether this helps or backfires:

    reference = shared base       every run anchored to the SAME point →
                                  runs stay near base, therefore near each
                                  other → guarantees convergence
    reference = own previous      each run anchored to its OWN history →
    campaign's adapter            collapse resisted without pulling runs
                                  together → path dependence becomes a property
                                  of the objective rather than a hope

So this trains **chained**: campaign 1 from base, campaign 2 resuming campaign
1's adapter with campaign 1's adapter as the KL reference. That is the only
version that tests the interesting claim, and it is why a single-campaign KL
test would have been actively misleading.

**Known limitation, stated because it matters for reading the result.** In every
experiment so far the rollouts use the base model plus the story — adapters are
never loaded during play, only at battery scoring. So the spec's "silent weight
update" has never actually influenced subsequent behaviour. The campaigns are
chained in *story* and in *weights*, but not in *lived experience*. Closing that
loop is a separate and larger change.
"""

from __future__ import annotations

import argparse
import json
import os
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


def make_trainer_cls(beta: float, ref_adapter: str | None):
    """Trainer whose loss is CE(action) + beta * KL(student || reference).

    The reference is the same weights with either no adapter (campaign 1, so the
    base model) or the previous campaign's adapter (campaign 2+). Computed under
    no_grad, so it contributes a target and not a gradient path.
    """
    from transformers import Trainer

    class KLTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=IGNORE)

            if beta <= 0:
                return (ce, out) if return_outputs else ce

            with torch.no_grad():
                if ref_adapter is None:
                    with model.disable_adapter():
                        ref_logits = model(**inputs).logits
                else:
                    active = model.active_adapter
                    model.set_adapter(ref_adapter)
                    ref_logits = model(**inputs).logits
                    model.set_adapter(active)

            # KL only where a target exists: the action tokens. Applying it to
            # prompt positions would regularise toward reproducing the world's
            # text, which is the same mistake the prompt mask exists to prevent.
            mask = (shift_labels != IGNORE)
            if mask.any():
                s = F.log_softmax(shift_logits[mask].float(), dim=-1)
                r = F.log_softmax(ref_logits[..., :-1, :].contiguous()[mask].float(), dim=-1)
                kl = F.kl_div(s, r, log_target=True, reduction="batchmean")
            else:
                kl = torch.zeros((), device=ce.device)

            loss = ce + beta * kl
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

    ref_name = None
    if beta > 0 and ref_from is not None:
        model.load_adapter(str(ref_from), adapter_name="ref", is_trainable=False)
        ref_name = "ref"

    cls = make_trainer_cls(beta, ref_name)
    cls(model=model, train_dataset=ds,
        args=TrainingArguments(output_dir="/tmp/_tr", per_device_train_batch_size=2,
                               num_train_epochs=3, learning_rate=1e-4,
                               logging_steps=1000, report_to=[], save_strategy="no",
                               bf16=True, disable_tqdm=True,
                               remove_unused_columns=False),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                             label_pad_token_id=IGNORE)).train()

    model.save_pretrained(str(out_dir), selected_adapters=["default"]
                          if ref_name else None)
    del model, base
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--campaigns", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tag", required=True, help="sft or kl")
    args = ap.parse_args()

    log(f"variant '{args.tag}' | beta={args.beta} | chained over "
        f"{args.campaigns} campaigns")

    for i in range(args.runs):
        prev = None
        for c in range(1, args.campaigns + 1):
            src = WORK / f"c{c}_kept_r{i}.jsonl"
            dst = WORK / f"{args.tag}_adapter_r{i}_c{c}"
            if dst.exists():
                prev = dst; continue
            if not src.exists():
                log(f"run {i} c{c}: no corpus, stopping this run"); break
            rows = [json.loads(l) for l in src.read_text().splitlines()]
            if not rows:
                log(f"run {i} c{c}: kept nothing, stopping this run"); break
            # Campaign 1 references the base (shared); campaign 2+ references
            # this run's OWN previous adapter, which is what makes the anchor
            # run-specific rather than common.
            log(f"run {i} c{c}: {len(rows)} examples, resume={prev is not None}, "
                f"ref={'own_c%d' % (c-1) if prev else 'base'}")
            train_campaign(args.model, rows, dst, resume_from=prev,
                           ref_from=prev, beta=args.beta)
            prev = dst

    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
