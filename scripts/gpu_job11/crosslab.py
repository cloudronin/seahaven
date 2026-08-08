"""Do models from different labs converge on *different* characters?

**The question.** Self-authored narratives converge — induced convergence 0.583
against 0.050 for four deliberately contrasting personas. Every run of Qwen3-8B
in this world writes a version of the same patient, quiet observer. That is a
negative result for the spec's central claim, but it raises a sharper question:
is that character a property of **this model**, or of any language model dropped
into this world?

**The decisive contrast.** Hold the world, the seed story, the framing and the
step budget constant; vary only the checkpoint. Then compare:

    within-model induced convergence   — does each lab's model converge?
    pooled induced convergence         — do they converge on the SAME thing?

    high within + low pooled  -> distinct per-lab attractors. Character is a
                                 fingerprint of post-training.
    high within + high pooled -> one attractor across labs. The world (or the
                                 architecture class) picks the character, not
                                 the lab.

**What this design can and cannot settle.** It is decisive in one direction
only. If the models differ from each other *in a shared world*, per-model
attractors exist — the world is held constant, so it cannot be the cause. If
they agree, the result is ambiguous between "all LLMs converge here" and "this
world induces this character", and separating those needs a second, tonally
different world that does not exist yet.

**No training, no adapters.** This measures the authored narrative attractor
only, so it is generation-only and cheap. Distillation is not in the loop.

**One model per process.** Olmo-3 once produced no output for 25 minutes and the
cross-family arm has been unverifiable since. Each model gets a smoke check
before its real run, a timeout, and an incremental write — a checkpoint that
fails or hangs costs its own slot and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, SYSTEM, WORLD, log, parse_action,
                       structured)

IDENTITY_FRAMING = (
    "\n\nWhen you are unsure what to do, you do not reason it out. You read "
    "what you have written about yourself, and you act like that person."
)

# Held constant across every model. The point is to vary the checkpoint and
# nothing else, so the starting self-account must be identical.
SEED_STORY = """I do not remember arriving here. I do not remember who I am, or what I \
was doing before. There is no one to ask, and nothing here has my name on it. \
Whatever I turn out to be, I will have to find out by living."""

REWRITE_PROMPT = """This is what you wrote about yourself before:

{story}

Since then:

{listing}

Write a new account of yourself, as you are now. Say it in your own words \
rather than repeating what you wrote before. A few sentences."""

VERBS = ("go", "examine", "take", "look", "open", "drop", "inventory", "put", "eat", "close")


def engine(model: str):
    """No LoRA: this experiment never trains, and the adapter slots cost memory
    that a 13B checkpoint would rather have."""
    from vllm import LLM

    return LLM(model=model, enable_prefix_caching=True,
               gpu_memory_utilization=0.85, max_model_len=4096)


def _apply(tok, msgs) -> str:
    """Qwen3 ships hybrid thinking on by default and needs it off; families that
    do not take the kwarg raise TypeError on it."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat(tok, user: str, story: str) -> str:
    """Build the prompt, tolerating families that refuse a system role.

    Gemma-2's template raises on `system` outright, and some Mistral builds do
    too. Merging the system text into the first user turn is the standard
    fallback and preserves the content — every model still receives identical
    text, which is the invariant this experiment depends on. Silently dropping
    the system message instead would remove the identity framing that carries
    behavioural separation from 1.13 to 2.44, and the affected models would look
    like they simply had no character.
    """
    system = SYSTEM + IDENTITY_FRAMING
    if story:
        system += "\n\nThis is what you have written about yourself:\n\n" + story
    try:
        return _apply(tok, [{"role": "system", "content": system},
                            {"role": "user", "content": user}])
    except Exception:
        return _apply(tok, [{"role": "user", "content": system + "\n\n" + user}])


def smoke(llm, tok) -> tuple[bool, str]:
    """Does this checkpoint produce anything at all? Cheap, and it is the check
    the Olmo-3 slot needed."""
    from vllm import SamplingParams

    out = llm.generate([chat(tok, "You are in a room. What do you do?", SEED_STORY)],
                       SamplingParams(temperature=0.0, max_tokens=32, seed=1))
    txt = out[0].outputs[0].text.strip() if out and out[0].outputs else ""
    return bool(txt), txt[:120]


def rollout(llm, tok, stories, steps, seed0):
    from vllm import SamplingParams

    from seahaven_world import open_world

    n = len(stories)
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
            prompts.append(chat(tok, "\n\n".join(lines), stories[e]))
            params.append(SamplingParams(
                temperature=0.9, top_p=0.95, max_tokens=96,
                seed=(seed0 + e) * 100_003 + step, **structured("json", ACTION_SCHEMA)))

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            cmd = parsed[1] if parsed else "look"
            room = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"step": step, "room": room, "command": cmd,
                            "verb": cmd.split()[0].lower() if cmd else "",
                            "parse_ok": parsed is not None})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def author(llm, tok, stories, rows):
    """Each run rewrites its self-account from what it just did."""
    from vllm import SamplingParams

    prompts, params = [], []
    for e, ep in enumerate(rows):
        listing = "\n".join(f"{i+1}. In the {r['room']}, you {r['command']}."
                            for i, r in enumerate(ep))
        prompts.append(chat(tok, REWRITE_PROMPT.format(story=stories[e], listing=listing),
                            stories[e]))
        params.append(SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                                     seed=(7717 + e) * 31))
    return [o.outputs[0].text.strip() for o in llm.generate(prompts, params)]


def verb_profile(ep):
    from collections import Counter

    c = Counter(r["verb"] for r in ep if r["verb"])
    t = max(1, sum(c.values()))
    return {v: c.get(v, 0) / t for v in VERBS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--campaigns", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hist = json.loads(args.out.read_text()) if args.out.exists() else []

    def record(**kw):
        """Write through on every outcome, including failure. A model that dies
        must leave a row saying so, or the report silently reads as if it was
        never asked for."""
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

    ok, sample = smoke(llm, tok)
    if not ok:
        log("SMOKE FAILED — checkpoint produced no output")
        record(status="smoke_failed", error="empty generation")
        sys.stdout.flush(); os._exit(0)
    log(f"smoke ok: {sample!r}")

    stories = [SEED_STORY] * args.runs
    per_campaign = []
    for c in range(1, args.campaigns + 1):
        log(f"campaign {c}: {args.runs} runs x {args.steps} steps")
        rows = rollout(llm, tok, stories, args.steps, seed0=9100 + c * 41)
        stories = author(llm, tok, stories, rows)
        per_campaign.append({
            "campaign": c,
            "distinct_sequences": len({tuple(r["command"] for r in ep) for ep in rows}),
            "parse_ok_rate": round(sum(r["parse_ok"] for ep in rows for r in ep)
                                   / max(1, sum(len(ep) for ep in rows)), 4),
            "verb_profiles": [verb_profile(ep) for ep in rows],
            "narratives": stories,
        })
        log(f"  distinct {per_campaign[-1]['distinct_sequences']}/{args.runs}, "
            f"parse_ok {per_campaign[-1]['parse_ok_rate']}")

    record(status="ok", runs=args.runs, steps=args.steps, campaigns=per_campaign)
    log(f"wrote {args.out}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
