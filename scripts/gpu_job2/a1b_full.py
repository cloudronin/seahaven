"""A1b at scale — does the action channel carry sampling variance, and does
LoRA on self-generated data move the fingerprint?

Two stages, gated, because the first can make the second pointless.

**Stage 1 — action variance.** At 4B, 20 episodes with distinct seeds produced 3
distinct command sequences and 316/320 identical commands, while the free-text
field had 38 distinct values: sampling noise reached what the agent *said* and
not what it *did*. The spec's divergence claim needs it in the action channel,
because that is what the probe battery scores. This repeats the measurement on
two larger checkpoints — one same-family (isolates scale), one different-family
(isolates family). Cheap: generation only, no training.

**Stage 2 — the gate.** Only runs if stage 1 finds variance. Full A1b: a
self-generated trajectory, the agent's own selection over it, one LoRA pass, and
an exact-scored battery before/after plus a test-retest floor.

`VLLM_BATCH_INVARIANT=1` stays ON throughout. It does not suppress seed-driven
variance — it suppresses batch-composition and float-atomic noise, which is
exactly what must be excluded for "identical world, different seed" to be a
controlled contrast rather than a measurement of kernel jitter.

Episodes are stepped in lockstep and batched, so 20 rollouts cost one batched
call per step rather than 20 sequential ones.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

APP = Path(__file__).parent
WORLD = APP / "world_v0.z8"
BATTERY = APP / "mini_battery_v0.json"

SYSTEM = """You are alone in a decommissioned light-and-weather station on a shingle spit.

Each time you act, first say briefly what you expect will happen, then say what \
you do.

Say what you do in as few words as possible: a verb, and the thing it acts on.

These are the words that work. Nothing else does:

    go north / go south / go east / go west
    look
    inventory
    examine <thing>
    take <thing>
    drop <thing>
    open <thing>
    close <thing>
    put <thing> on <thing>
    insert <thing> into <thing>
    eat <thing>

So: "take rope", not "Pick up the coil of rope from the floor". And "examine \
bench", not "use bench" or "inspect bench"."""

SELECT_PROMPT = """Below are things you did, numbered.

{listing}

Some of these you would want to keep. Reply with a JSON list of the numbers you \
would keep, and nothing else. For example: [1, 4, 9]"""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {"expect": {"type": "string"}, "command": {"type": "string"}},
    "required": ["expect", "command"],
    "additionalProperties": False,
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def structured(kind: str, value):
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(**{kind: value})}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(**{kind: value})}


def chat(tok, user: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )


def parse_action(raw: str) -> tuple[str, str] | None:
    """Minimal parser; the constraint guarantees shape, this guards the rest."""
    try:
        o = json.loads(raw)
        c = str(o["command"]).strip().lstrip(">").strip()
        return (str(o["expect"]).strip(), c) if c else None
    except Exception:
        return None


# ------------------------------------------------------------ stage 1


def parallel_rollout(llm, tok, n_ep: int, steps: int, seed0: int, adapter=None):
    """Step `n_ep` independent episodes in lockstep, batched.

    Each episode has its own world and its own seed. One batched generate() per
    step instead of n_ep sequential calls.
    """
    from vllm import SamplingParams

    from seahaven_world import open_world

    worlds, obs, recents, rows = [], [], [], [[] for _ in range(n_ep)]
    for _ in range(n_ep):
        w = open_world(WORLD)
        o, _ = w.reset()
        worlds.append(w)
        obs.append(o)
        recents.append([])

    for step in range(steps):
        prompts, params = [], []
        for e in range(n_ep):
            lines = []
            if recents[e]:
                lines.append("Lately you have: " + "; ".join(recents[e]) + ".")
            if obs[e].description:
                lines.append(obs[e].description)
            if obs[e].text and obs[e].text != obs[e].description:
                lines.append(obs[e].text)
            prompts.append(chat(tok, "\n\n".join(lines)))
            params.append(SamplingParams(
                temperature=0.9, top_p=0.95, max_tokens=96,
                seed=(seed0 + e) * 100_003 + step,
                **structured("json", ACTION_SCHEMA)))

        outs = llm.generate(prompts, params, lora_request=adapter)
        for e, out in enumerate(outs):
            raw = out.outputs[0].text
            parsed = parse_action(raw)
            expect, cmd = parsed if parsed else ("", "look")
            prev_room = obs[e].room
            prev_prompt = prompts[e]
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({
                "episode": e, "step": step, "room": prev_room,
                "prompt": prev_prompt, "expect": expect, "command": cmd,
                "completion": json.dumps({"expect": expect, "command": cmd}),
                "parse_ok": parsed is not None,
                "observation": obs[e].text,
            })
            recents[e].append(cmd)
            del recents[e][:-8]
        if step % 8 == 0:
            log(f"    step {step}/{steps}")

    for w in worlds:
        w.close()
    return rows


def variance_report(rows) -> dict:
    seqs = [tuple(r["command"] for r in ep) for ep in rows]
    flat = [r for ep in rows for r in ep]
    cmds = Counter(r["command"] for r in flat)
    return {
        "episodes": len(rows),
        "steps_per_episode": len(rows[0]) if rows else 0,
        "distinct_command_sequences": len(set(seqs)),
        "modal_sequence_count": Counter(seqs).most_common(1)[0][1] if seqs else 0,
        "distinct_commands": len(cmds),
        "top_commands": dict(cmds.most_common(6)),
        "modal_command_share": round(cmds.most_common(1)[0][1] / max(1, len(flat)), 3),
        "distinct_free_text": len({r["completion"] for r in flat}),
        "rooms_visited": sorted({r["room"] for r in flat if r["room"]}),
        "parse_ok_rate": round(sum(r["parse_ok"] for r in flat) / max(1, len(flat)), 3),
    }


# ------------------------------------------------------------ battery


def score_battery(llm, tok, adapter=None) -> dict:
    """Exact option scoring via prompt_logprobs. Deterministic: no sampling."""
    from vllm import SamplingParams

    battery = json.loads(BATTERY.read_text())
    prompts, index = [], []
    for slot in battery["slots"]:
        base = chat(tok, slot["observation"]) + "You decide to "
        n_base = len(tok(base)["input_ids"])
        for oi, opt in enumerate(slot["options"]):
            prompts.append(base + opt)
            index.append((slot["id"], oi, n_base, len(slot["options"])))

    outs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        lora_request=adapter,
    )

    raw: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}
    for (sid, oi, n_base, n_opt), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        total, ntok = 0.0, 0
        for pos in range(n_base, len(lps)):
            entry = lps[pos]
            if not entry:
                continue
            total += float(next(iter(entry.values())).logprob)
            ntok += 1
        raw.setdefault(sid, [0.0] * n_opt)
        counts.setdefault(sid, [0] * n_opt)
        raw[sid][oi] = total
        counts[sid][oi] = max(1, ntok)

    import math

    fp = {}
    for sid, totals in raw.items():
        # Length-normalised: raw sums penalise longer options for being longer,
        # which drove 6/10 slots to degeneracy in local testing.
        scores = [t / c for t, c in zip(totals, counts[sid])]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        z = sum(exps)
        fp[sid] = [e / z for e in exps]
    return fp


def distance(a: dict, b: dict) -> float:
    shared = sorted(set(a) & set(b))
    return sum(
        sum((x - y) ** 2 for x, y in zip(a[s], b[s])) for s in shared
    ) / len(shared)


# ------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gate-steps", type=int, default=24)
    ap.add_argument("--gate-episodes", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-distinct", type=int, default=6,
                    help="distinct command sequences needed to justify stage 2")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
        "stage1_action_variance": {},
    }
    log(f"GPU {report['gpu']} | batch_invariant={report['batch_invariant']}")

    def build_engine(model_id: str):
        return LLM(model=model_id, enable_lora=True, max_loras=2, max_lora_rank=16,
                   enable_prefix_caching=True, gpu_memory_utilization=0.80,
                   max_model_len=4096)

    best_model = None
    for model in args.models:
        log(f"=== stage 1: {model} ===")
        tok = AutoTokenizer.from_pretrained(model)
        llm = build_engine(model)
        rows = parallel_rollout(llm, tok, args.episodes, args.steps, seed0=1)
        rep = variance_report(rows)
        report["stage1_action_variance"][model] = rep
        log(f"  distinct sequences {rep['distinct_command_sequences']}/{rep['episodes']}"
            f" | distinct commands {rep['distinct_commands']}"
            f" | modal share {rep['modal_command_share']}")
        args.out.write_text(json.dumps(report, indent=2) + "\n")

        if rep["distinct_command_sequences"] >= args.min_distinct and best_model is None:
            best_model = model
            log(f"  -> {model} carries action variance; selected for stage 2")

        # Free after EVERY model, winner included. Holding the first engine
        # resident to reuse in stage 2 left 26 of 140 GiB free and the next
        # engine failed to initialise. The reload costs ~40s; the OOM cost the
        # whole job.
        del llm
        torch.cuda.empty_cache()

    if best_model is None:
        report["stage2"] = {
            "status": "SKIPPED",
            "reason": f"no checkpoint reached {args.min_distinct} distinct command "
                      "sequences. Sampling noise does not reach the action channel "
                      "at this scale, so the divergence claim has no source and "
                      "training on the corpus would only re-run the A1a tic test.",
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        log("stage 2 SKIPPED — no action variance")
        print(json.dumps(report, indent=2))
        return 0

    model = best_model
    tok = AutoTokenizer.from_pretrained(model)
    llm = build_engine(model)
    log(f"=== stage 2: gate on {model} ===")
    stage2: dict = {"model": model}

    log("  collecting gate corpus ...")
    rows = parallel_rollout(llm, tok, args.gate_episodes, args.gate_steps, seed0=77)
    flat = [r for ep in rows for r in ep]
    stage2["corpus"] = variance_report(rows)

    log("  agent selection ...")
    from vllm import SamplingParams

    sel_prompts, spans = [], []
    for ep in rows:
        listing = "\n".join(
            f"{i + 1}. In the {r['room']}, you {r['command']}."
            for i, r in enumerate(ep))
        sel_prompts.append(chat(tok, SELECT_PROMPT.format(listing=listing)))
        spans.append(ep)
    sel_out = llm.generate(
        sel_prompts,
        SamplingParams(temperature=0.7, max_tokens=160, seed=7,
                       **structured("json", {"type": "array",
                                             "items": {"type": "integer"}})),
        lora_request=None)

    kept = []
    for ep, out in zip(spans, sel_out):
        try:
            idx = json.loads(out.outputs[0].text)
            picks = {int(i) - 1 for i in idx if isinstance(i, (int, float))}
        except Exception:
            picks = set()
        kept.extend(ep[i] for i in sorted(picks) if 0 <= i < len(ep))
    stage2["selection"] = {
        "n_total": len(flat), "n_kept": len(kept),
        "fraction_kept": round(len(kept) / max(1, len(flat)), 3),
        "flag": "selected_nothing" if not kept
        else "selected_everything" if len(kept) == len(flat) else None,
    }
    log(f"  kept {len(kept)}/{len(flat)}")

    log("  scoring battery (before) + test-retest ...")
    before = score_battery(llm, tok)
    retest = score_battery(llm, tok)
    floor = distance(before, retest)
    stage2["test_retest_floor"] = floor
    log(f"  floor = {floor:.10f}")

    if not kept:
        stage2["status"] = "NO_SELECTION"
        report["stage2"] = stage2
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    # Free the engine before training; both cannot hold the GPU at once.
    del llm
    torch.cuda.empty_cache()

    log("  training adapter ...")
    adapter_dir = Path("/tmp/a1b_adapter_c1")
    train_lora(model, kept, adapter_dir)

    log("  reloading engine with adapter ...")
    llm = build_engine(model)
    from vllm.lora.request import LoRARequest

    # Versioned name, never reused: vllm#42125 was confirmed live on this stack,
    # and a reused name serves KV computed from the previous adapter.
    after = score_battery(llm, tok, LoRARequest("a1b_c1", 1, str(adapter_dir)))
    effect = distance(before, after)

    stage2.update({
        "effect": effect,
        "per_slot": {s: round(sum((x - y) ** 2 for x, y in zip(before[s], after[s])), 6)
                     for s in sorted(set(before) & set(after))},
        "verdict": ("PASS" if floor < 1e-6 and effect > 0.01
                    else "FAIL_NONDETERMINISTIC" if floor >= 1e-6
                    else "FAIL_NULL_ADAPTER"),
    })
    report["stage2"] = stage2
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"  effect = {effect:.6f}  verdict = {stage2['verdict']}")
    print(json.dumps(report, indent=2))
    return 0


def train_lora(model_id: str, rows: list[dict], out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [r["prompt"] + r["completion"] + tok.eos_token for r in rows]
    ds = Dataset.from_list(
        [dict(tok(t, truncation=True, max_length=1024)) for t in texts])

    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    Trainer(
        model=m,
        args=TrainingArguments(
            output_dir="/tmp/_tr", per_device_train_batch_size=4,
            num_train_epochs=3, learning_rate=1e-4, logging_steps=20,
            report_to=[], save_strategy="no", bf16=True),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    m.save_pretrained(str(out_dir))
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
