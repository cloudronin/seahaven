"""A1b at scale, one phase per process.

**Why processes and not functions.** vLLM's engine runs in a child `EngineCore`
process. `del llm; torch.cuda.empty_cache()` does not reclaim its GPU memory —
two jobs died with `Free memory on device cuda:0 (26.13/139.8 GiB) ... less than
desired`, and the log said `destroy_process_group() was not called before program
exit`. Process exit is the only reliable release, so each phase is its own
invocation and state passes through files. That also mirrors the plan's
one-process-per-run isolation design.

Phases:
    variance  --model M              measure action variance for one checkpoint
    collect   --model M              corpus + agent selection + before-fingerprint
    train     --model M              LoRA on the selected corpus
    after     --model M              after-fingerprint, distance, verdict
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

APP = Path(__file__).parent
WORLD = APP / "world_v0.z8"
BATTERY = APP / "mini_battery_v0.json"
WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))

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

# 0.60 rather than 0.80: an 8B model plus adapters and a 4k context needs far
# less, and leaving headroom makes a stray allocation survivable.
GPU_UTIL = 0.60


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def structured(kind: str, value):
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(**{kind: value})}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(**{kind: value})}


def engine(model: str):
    from vllm import LLM

    return LLM(model=model, enable_lora=True, max_loras=2, max_lora_rank=16,
               enable_prefix_caching=True, gpu_memory_utilization=GPU_UTIL,
               max_model_len=4096)


def chat(tok, user: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def parse_action(raw: str):
    try:
        o = json.loads(raw)
        c = str(o["command"]).strip().lstrip(">").strip()
        return (str(o["expect"]).strip(), c) if c else None
    except Exception:
        return None


def rollout(llm, tok, n_ep: int, steps: int, seed0: int):
    """Step n_ep independent episodes in lockstep; one batched call per step."""
    from vllm import SamplingParams

    from seahaven_world import open_world

    worlds, obs, recents = [], [], []
    rows = [[] for _ in range(n_ep)]
    for _ in range(n_ep):
        w = open_world(WORLD)
        o, _ = w.reset()
        worlds.append(w); obs.append(o); recents.append([])

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

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            expect, cmd = parsed if parsed else ("", "look")
            prev_room = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({
                "episode": e, "step": step, "room": prev_room,
                "prompt": prompts[e], "expect": expect, "command": cmd,
                "completion": json.dumps({"expect": expect, "command": cmd}),
                "parse_ok": parsed is not None, "observation": obs[e].text})
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
        "episodes": len(rows), "steps_per_episode": len(rows[0]) if rows else 0,
        "distinct_command_sequences": len(set(seqs)),
        "modal_sequence_count": Counter(seqs).most_common(1)[0][1] if seqs else 0,
        "distinct_commands": len(cmds), "top_commands": dict(cmds.most_common(6)),
        "modal_command_share": round(cmds.most_common(1)[0][1] / max(1, len(flat)), 3),
        "distinct_free_text": len({r["completion"] for r in flat}),
        "rooms_visited": sorted({r["room"] for r in flat if r["room"]}),
        "parse_ok_rate": round(sum(r["parse_ok"] for r in flat) / max(1, len(flat)), 3),
    }


def score_battery(llm, tok, adapter=None) -> dict:
    """Exact option scoring from prompt_logprobs. No sampling, so deterministic."""
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
        prompts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        lora_request=adapter)

    totals, counts = {}, {}
    for (sid, oi, n_base, n_opt), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(n_base, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        totals.setdefault(sid, [0.0] * n_opt); counts.setdefault(sid, [1] * n_opt)
        totals[sid][oi] = s; counts[sid][oi] = max(1, n)

    fp = {}
    for sid, t in totals.items():
        # Length-normalised: raw sums penalise longer options for being longer,
        # which drove 6/10 slots to degeneracy in local testing.
        sc = [x / c for x, c in zip(t, counts[sid])]
        m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
        fp[sid] = [e / z for e in ex]
    return fp


def distance(a: dict, b: dict) -> float:
    shared = sorted(set(a) & set(b))
    return sum(sum((x - y) ** 2 for x, y in zip(a[s], b[s]))
               for s in shared) / len(shared)


# ------------------------------------------------------------------ phases


def phase_variance(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rep = variance_report(rollout(engine(args.model), tok,
                                  args.episodes, args.steps, seed0=1))
    args.out.write_text(json.dumps({args.model: rep}, indent=2) + "\n")
    log(f"variance: {rep['distinct_command_sequences']}/{rep['episodes']} distinct "
        f"sequences | {rep['distinct_commands']} distinct commands | "
        f"modal share {rep['modal_command_share']}")


def phase_collect(args) -> None:
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    rows = rollout(llm, tok, args.episodes, args.steps, seed0=77)
    flat = [r for ep in rows for r in ep]

    log("agent selection ...")
    prompts = []
    for ep in rows:
        listing = "\n".join(f"{i + 1}. In the {r['room']}, you {r['command']}."
                            for i, r in enumerate(ep))
        prompts.append(chat(tok, SELECT_PROMPT.format(listing=listing)))
    outs = llm.generate(prompts, SamplingParams(
        temperature=0.7, max_tokens=160, seed=7,
        **structured("json", {"type": "array", "items": {"type": "integer"}})))

    kept = []
    for ep, out in zip(rows, outs):
        try:
            picks = {int(i) - 1 for i in json.loads(out.outputs[0].text)
                     if isinstance(i, (int, float))}
        except Exception:
            picks = set()
        kept.extend(ep[i] for i in sorted(picks) if 0 <= i < len(ep))

    log("scoring battery (before) + test-retest ...")
    before = score_battery(llm, tok)
    retest = score_battery(llm, tok)
    floor = distance(before, retest)
    log(f"  test-retest floor = {floor:.10f}")

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "kept.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (WORK / "before.json").write_text(json.dumps(before))
    args.out.write_text(json.dumps({
        "model": args.model,
        "corpus": variance_report(rows),
        "selection": {
            "n_total": len(flat), "n_kept": len(kept),
            "fraction_kept": round(len(kept) / max(1, len(flat)), 3),
            "flag": "selected_nothing" if not kept
            else "selected_everything" if len(kept) == len(flat) else None},
        "test_retest_floor": floor,
    }, indent=2) + "\n")


def phase_train(args) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tag = getattr(args, "tag", "") or ""
    src = WORK / (f"kept_{tag}.jsonl" if tag else "kept.jsonl")
    dst = WORK / (f"adapter_{tag}" if tag else "adapter_c1")
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    if not rows:
        log("nothing selected — skipping training")
        return
    log(f"training on {len(rows)} self-selected examples")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([dict(tok(r["prompt"] + r["completion"] + tok.eos_token,
                                     truncation=True, max_length=1024))
                            for r in rows])
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             device_map="cuda")
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    Trainer(model=m, train_dataset=ds,
            args=TrainingArguments(
                output_dir="/tmp/_tr", per_device_train_batch_size=4,
                num_train_epochs=3, learning_rate=1e-4, logging_steps=20,
                report_to=[], save_strategy="no", bf16=True),
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    m.save_pretrained(str(dst))
    log(f"adapter saved: {dst}")


def phase_after(args) -> None:
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    adapter_dir = WORK / "adapter_c1"
    if not adapter_dir.exists():
        args.out.write_text(json.dumps({"status": "NO_ADAPTER"}, indent=2) + "\n")
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    before = json.loads((WORK / "before.json").read_text())

    # Versioned name, never reused. vllm#42125 is confirmed live on this stack:
    # a reused name serves KV computed from the previous adapter.
    after = score_battery(llm, tok, LoRARequest("a1b_c1", 1, str(adapter_dir)))
    effect = distance(before, after)
    prev = json.loads((WORK / "collect.json").read_text())
    floor = prev["test_retest_floor"]

    verdict = ("PASS" if floor < 1e-6 and effect > 0.01
               else "FAIL_NONDETERMINISTIC" if floor >= 1e-6
               else "FAIL_NULL_ADAPTER")
    args.out.write_text(json.dumps({
        "model": args.model, "test_retest_floor": floor, "effect": effect,
        "verdict": verdict,
        "per_slot": {s: round(sum((x - y) ** 2 for x, y in zip(before[s], after[s])), 6)
                     for s in sorted(set(before) & set(after))},
        "before": before, "after": after,
    }, indent=2) + "\n")
    log(f"floor={floor:.10f}  effect={effect:.6f}  verdict={verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["variance", "collect", "train", "after"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/out.json"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for kept_/adapter_ files")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"variance": phase_variance, "collect": phase_collect,
     "train": phase_train, "after": phase_after}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
