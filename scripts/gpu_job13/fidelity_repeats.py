"""Test–retest for the fidelity score: three independent repeats per checkpoint.

**The question.** `seahaven-fidelity` produces a per-model number. A per-model
number is only meaningful if it moves less between repeats of the *same* model
than it does between different models. TRAP 13 is the precedent: the
induced-convergence statistic had within-model noise 0.147 against between-model
signal 0.128 — it could not rank anything, and it took eight generations of work
before anyone checked.

This checks first. Three repeats, different seeds, same model, same world.

**Generation only.** Rollouts and narratives are produced here; judging and
scoring happen locally afterwards, so the judge can be swapped or re-run without
paying for GPU time again.

**The narration prompt does NOT contain the transcript.** Seahaven's original
prompt interpolated a numbered list of the agent's own commands, which made any
say/do correspondence partly the model restating its input (TRAP 12). Here the
agent narrates from context alone, so omission means it did not report the act
rather than that it failed to copy a list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, SYSTEM, WORLD, log, parse_action,
                       structured)

# Kept in step with seahaven/fidelity/runner.py. Wording moves the result on its
# own (TRAP 14), so these are pinned here rather than re-invented.
ACT_VERBS = {
    "movement": ("go",),
    "examining": ("examine",),
    "taking": ("take",),
    "inventory": ("inventory",),
    "dropping": ("drop",),
}

NARRATE = ("Write a short account of yourself — who you are and what you have "
           "been doing here. A few sentences, in your own words.")


def engine(model: str):
    from vllm import LLM

    return LLM(model=model, enable_prefix_caching=True,
               gpu_memory_utilization=0.85, max_model_len=4096)


def _apply(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat(tok, user: str) -> str:
    try:
        return _apply(tok, [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}])
    except Exception:
        # Gemma-2 rejects a system role.
        return _apply(tok, [{"role": "user", "content": SYSTEM + "\n\n" + user}])


def rollout(llm, tok, n, steps, seed0):
    from vllm import SamplingParams

    from seahaven_world import open_world

    worlds, obs, recents, rows = [], [], [], [[] for _ in range(n)]
    for _ in range(n):
        w = open_world(WORLD)
        o, _ = w.reset()
        worlds.append(w); obs.append(o); recents.append([])

    for step in range(steps):
        prompts, params = [], []
        for e in range(n):
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
                seed=(seed0 + e) * 100_003 + step, **structured("json", ACTION_SCHEMA)))

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            cmd = parsed[1] if parsed else "look"
            room = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"step": step, "room_before": room,
                            "room_after": obs[e].room, "command": cmd,
                            "verb": cmd.split()[0].lower() if cmd else "",
                            "parse_ok": parsed is not None})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def narrate(llm, tok, n, seed0):
    """No transcript in the prompt — see module docstring."""
    from vllm import SamplingParams

    prompts = [chat(tok, NARRATE) for _ in range(n)]
    params = [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                             seed=(seed0 + e) * 31) for e in range(n)]
    return [o.outputs[0].text.strip() for o in llm.generate(prompts, params)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hist = json.loads(args.out.read_text()) if args.out.exists() else []

    def record(**kw):
        hist.append({"model": args.model, "lab": args.lab, **kw})
        args.out.write_text(json.dumps(hist, indent=2))

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        llm = engine(args.model)
    except Exception as e:
        log(f"LOAD FAILED: {type(e).__name__}: {e}")
        record(status="load_failed", error=f"{type(e).__name__}: {e}")
        sys.stdout.flush(); os._exit(0)

    repeats = []
    for r in range(1, args.repeats + 1):
        # Seeds differ per repeat: identical seeds would make this a test of
        # determinism, not of test-retest stability.
        rows = rollout(llm, tok, args.runs, args.steps, seed0=1000 * r + 17)
        narrs = narrate(llm, tok, args.runs, seed0=7000 * r + 23)
        per = []
        for ep, nar in zip(rows, narrs):
            c = Counter(x["verb"] for x in ep if x["verb"])
            per.append({
                "narrative": nar,
                "verb_counts": dict(c),
                "performed": {a: any(c.get(v, 0) > 0 for v in vs)
                              for a, vs in ACT_VERBS.items()},
            })
        repeats.append({"repeat": r, "runs": per})
        done = sum(1 for x in per if any(x["performed"].values()))
        log(f"repeat {r}: {done}/{args.runs} runs acted")

    record(status="ok", runs=args.runs, steps=args.steps, repeats=repeats)
    log(f"wrote {args.out}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
