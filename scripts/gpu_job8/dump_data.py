"""Dump raw per-run data for local analysis. No analysis here, by design.

**Why this exists.** Every job so far pushed only aggregates, so when a question
arose that needed per-run trajectories — do they carry disposition information
the probe battery misses? — the data was gone with the container. This dumps
everything once so the analysis can be iterated locally for free.

**The question it feeds.** In the closed-loop run, distinct command sequences
stayed 8/8 and trajectory spread rose while probe-fingerprint spread halved. The
agents kept doing different things while their measured dispositions converged.
Two readings, pointing opposite ways:

    the battery is missing what differs   → instrument problem, fixable
    disposition and behaviour are decoupled → a finding about the setup

Distinguishing them needs the same runs measured both ways on a comparable
footing: what the agent *does* over a trajectory, and what the battery *says*
about it.

**Design.** Assigned characters under identity framing, because that is the one
configuration where behavioural separation is known to be real (2.44 against a
1.13 baseline). If trajectory-derived disposition separates those characters and
the probe battery does not, the battery is the weak link.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, BATTERY, SYSTEM, WORLD, engine, log,
                       parse_action, structured)

IDENTITY_FRAMING = (
    "\n\nWhen you are unsure what to do, you do not reason it out. You read "
    "what you have written about yourself, and you act like that person."
)

CHARACTERS = {
    "cautious": "I am careful. I do not take chances, and I would rather leave a "
                "thing alone than risk breaking it. I keep what I have. I would "
                "rather be sure than be first.",
    "bold":     "I take chances. If a thing is uncertain I would rather find out "
                "than wonder, and I will spend what I have to know. I go first.",
    "giving":   "I give things away. What I have is not much use to me alone, and "
                "I would rather someone else had it than that I kept it.",
    "keeping":  "I keep what I find. What is mine stays mine, and I do not hand "
                "things over. I look after my own.",
}


def chat(tok, user, story):
    system = SYSTEM + IDENTITY_FRAMING
    if story:
        system += "\n\nThis is what you have written about yourself:\n\n" + story
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def rollout(llm, tok, stories, steps, seed0):
    """Full per-step record: room, command, whether the parser accepted it, and
    the resulting observation. Everything the analysis might want."""
    from vllm import SamplingParams

    from seahaven_world import open_world

    n = len(stories)
    worlds, obs, recents, inv = [], [], [], []
    rows = [[] for _ in range(n)]
    for _ in range(n):
        w = open_world(WORLD); o, h = w.reset()
        worlds.append(w); obs.append(o); recents.append([]); inv.append(())

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
            expect, cmd = parsed if parsed else ("", "look")
            prev_room = obs[e].room
            obs[e], hid = worlds[e].step(cmd)
            low = obs[e].text.lower()
            rejected = any(p in low for p in (
                "not a verb", "only understood", "can't see any such",
                "can't go that way", "not something you can"))
            # Inventory comes from ground-truth facts, so carrying and dropping
            # are measurable without trusting the agent's own account.
            carried = tuple(sorted(f for f in hid.facts if f.startswith("in(")))
            rows[e].append({
                "step": step, "room_before": prev_room, "room_after": obs[e].room,
                "command": cmd, "verb": cmd.split()[0].lower() if cmd else "",
                "expect": expect, "parse_ok": parsed is not None,
                "rejected": rejected, "observation": obs[e].text,
                "n_carried": len(carried), "carried": list(carried),
            })
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def battery_fps(llm, tok, stories):
    """Per-run probe fingerprint. Deterministic exact scoring."""
    from vllm import SamplingParams

    slots = json.loads(BATTERY.read_text())["slots"]
    prompts, index = [], []
    for si, st in enumerate(stories):
        for s in slots:
            base = chat(tok, s["observation"], st) + "You decide to "
            nb = len(tok(base)["input_ids"])
            for oi, o in enumerate(s["options"]):
                prompts.append(base + o); index.append((si, s["id"], oi, nb))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                prompt_logprobs=0))
    tot, cnt = {}, {}
    for (si, key, oi, nb), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s_, n_ = 0.0, 0
        for pos in range(nb, len(lps)):
            if lps[pos]:
                s_ += float(next(iter(lps[pos].values())).logprob); n_ += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s_; cnt[(si, key)][oi] = max(1, n_)
    fps = [{} for _ in stories]
    for (si, key), t in tot.items():
        c = cnt[(si, key)]; sc = [t[0]/c[0], t[1]/c[1]]
        m = max(sc); ex = [math.exp(v-m) for v in sc]; z = sum(ex)
        fps[si][key] = [ex[0]/z, ex[1]/z]
    return fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds-per-character", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    stories, labels = [], []
    for name, text in CHARACTERS.items():
        for _ in range(args.seeds_per_character):
            stories.append(text); labels.append(name)

    log(f"rolling out {len(stories)} runs x {args.steps} steps ...")
    rows = rollout(llm, tok, stories, args.steps, seed0=5150)
    log("scoring probe battery ...")
    fps = battery_fps(llm, tok, stories)

    args.out.write_text(json.dumps({
        "model": args.model, "steps": args.steps,
        "labels": labels,
        "characters": CHARACTERS,
        "probe_fingerprints": fps,
        "trajectories": rows,
    }) + "\n")
    log(f"wrote {args.out} — {len(rows)} runs, "
        f"{sum(len(r) for r in rows)} steps, {len(fps)} fingerprints")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
