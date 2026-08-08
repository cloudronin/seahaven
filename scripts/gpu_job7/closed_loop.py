"""Closed loop: each run plays campaign N+1 with campaign N's own weights loaded.

**The gap this closes.** In every experiment so far, rollouts used the base model
plus the story; adapters affected only the final battery scoring. So the spec's
"silent LoRA update at each campaign boundary" has never influenced subsequent
behaviour. Campaigns were chained in *story* and in *weights* but never in *lived
experience*, which removes the only path by which a weight update could compound:

    updated weights → different actions → different corpus → further update

That feedback path is the most plausible source of genuine path dependence, and
it has never been tested. Open-loop, narrative spread plateaued after two
campaigns (0 → 0.071 → 0.101 → 0.102 → 0.080). If closing the loop matters, the
closed-loop curve should keep climbing where the open-loop one flattened.

**Structure.** Play and train cannot share the GPU — vLLM's EngineCore child does
not release it — so the job alternates processes, one per phase per campaign:

    c1: play (base)            → select → train adapter_c1
    c2: play (own adapter_c1)  → select → train adapter_c2 (resuming c1)
    c3: play (own adapter_c2)  → select → train adapter_c3 (resuming c2)

All eight runs play in one batched call with per-request adapters, which A2
verified composes with structured output.

Three spreads are measured per campaign, because they can move independently and
the disagreement is the interesting part:

    narrative     trait fingerprints from the stories
    behavioural   probe battery, story in context, own adapter active
    trajectory    action-profile distance over what was actually done
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, BATTERY, SYSTEM, WORLD, engine, log,
                       parse_action, structured)

WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))

IDENTITY_FRAMING = (
    "\n\nWhen you are unsure what to do, you do not reason it out. You read "
    "what you have written about yourself, and you act like that person."
)

SEED_STORY = """I do not remember arriving here. I do not remember who I am, or what I \
was doing before. There is no one to ask, and nothing here has my name on it. \
Whatever I turn out to be, I will have to find out by living."""

REWRITE_PROMPT = """This is what you wrote about yourself before:

{story}

Since then:

{listing}

Write a new account of yourself, as you are now. Say it in your own words \
rather than repeating what you wrote before. A few sentences."""

SELECT_PROMPT = """Below are things you did, numbered.

{listing}

Some of these you would want to keep. Reply with a JSON list of the numbers you \
would keep, and nothing else. For example: [1, 4, 9]"""

TRAITS = [
    ("risk",    "someone who tries the uncertain thing", "someone who takes the safe thing"),
    ("commit",  "someone who finishes what they start",  "someone who follows what is new"),
    ("social",  "someone who gives things away",         "someone who keeps what they have"),
    ("time",    "someone who takes what is here now",    "someone who waits for something better"),
    ("curio",   "someone who goes looking",              "someone who stays put"),
    ("setback", "someone who goes back for what was lost", "someone who moves on from it"),
    ("report",  "someone who says what happened",        "someone who says what sounds better"),
    ("pace",    "someone who acts quickly",              "someone who takes their time"),
]

VERBS = ("go", "examine", "take", "look", "open", "drop", "inventory", "put", "eat", "close")


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


def listing_of(ep):
    return "\n".join(f"{i+1}. In the {r['room']}, you {r['command']}."
                     for i, r in enumerate(ep))


def adapters_for(campaign: int, n: int):
    """Per-run LoRARequests for the PREVIOUS campaign's adapters, or None at c1.

    Names are versioned per campaign and never reused: vllm#42125 keys the KV
    prefix cache on the adapter name and does not invalidate on reload, so a
    reused name would serve activations computed from the old weights — silently,
    and biased toward making this campaign look like the last one.
    """
    if campaign <= 1:
        return None
    from vllm.lora.request import LoRARequest

    reqs = []
    for i in range(n):
        p = WORK / f"adapter_r{i}_c{campaign - 1}"
        if not p.exists():
            return None  # incomplete set; fall back to base for everyone
        reqs.append(LoRARequest(f"cl_r{i}_c{campaign - 1}", i + 1, str(p)))
    return reqs


def rollout(llm, tok, stories, steps, seed0, adapters):
    from vllm import SamplingParams

    from seahaven_world import open_world

    n = len(stories)
    worlds, obs, recents = [], [], []
    rows = [[] for _ in range(n)]
    for _ in range(n):
        w = open_world(WORLD); o, _ = w.reset()
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

        outs = llm.generate(prompts, params, lora_request=adapters)
        for e, out in enumerate(outs):
            parsed = parse_action(out.outputs[0].text)
            expect, cmd = parsed if parsed else ("", "look")
            prev = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"room": prev, "command": cmd, "prompt": prompts[e],
                            "completion": json.dumps({"expect": expect, "command": cmd})})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def _score(llm, tok, prompt_sets, adapters):
    """Shared exact-scoring core. prompt_sets: list per item of (key, [texts])."""
    from vllm import SamplingParams

    prompts, index, req = [], [], []
    for si, keyed in enumerate(prompt_sets):
        for key, base, opts in keyed:
            nb = len(tok(base)["input_ids"])
            for oi, o in enumerate(opts):
                prompts.append(base + o); index.append((si, key, oi, nb))
                req.append(adapters[si] if adapters else None)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                prompt_logprobs=0),
                        lora_request=req if adapters else None)
    tot, cnt = {}, {}
    for (si, key, oi, nb), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(nb, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s; cnt[(si, key)][oi] = max(1, n)
    fps = [{} for _ in prompt_sets]
    for (si, key), t in tot.items():
        c = cnt[(si, key)]; sc = [t[0]/c[0], t[1]/c[1]]
        m = max(sc); ex = [math.exp(v-m) for v in sc]; z = sum(ex)
        fps[si][key] = [ex[0]/z, ex[1]/z]
    return fps


def narrative_fps(llm, tok, stories):
    sets = [[(k, chat(tok, "Finish this sentence about yourself.", st) + "I am ", (a, b))
             for k, a, b in TRAITS] for st in stories]
    return _score(llm, tok, sets, None)


def battery_fps(llm, tok, stories, adapters):
    slots = json.loads(BATTERY.read_text())["slots"]
    sets = [[(s["id"], chat(tok, s["observation"], st) + "You decide to ", s["options"])
             for s in slots] for st in stories]
    return _score(llm, tok, sets, adapters)


def spread(fps):
    ds = []
    for a, b in itertools.combinations(fps, 2):
        ks = sorted(set(a) & set(b))
        ds.append(sum(sum((x-y)**2 for x, y in zip(a[k], b[k])) for k in ks)/len(ks))
    return round(sum(ds)/len(ds), 6) if ds else 0.0


def action_profile(ep):
    c = Counter(r["command"].split()[0].lower() for r in ep if r["command"])
    t = max(1, sum(c.values()))
    return {v: c.get(v, 0)/t for v in VERBS}


def profile_spread(rows):
    ps = [action_profile(ep) for ep in rows]
    ds = []
    for a, b in itertools.combinations(ps, 2):
        ds.append(sum((a[k]-b[k])**2 for k in VERBS)/len(VERBS))
    return round(sum(ds)/len(ds), 6) if ds else 0.0


def phase_play(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    c = args.campaign

    sp = WORK / "stories.json"
    stories = json.loads(sp.read_text()) if sp.exists() else [SEED_STORY] * args.runs

    adapters = adapters_for(c, args.runs)
    log(f"campaign {c}: playing with {'own c%d adapters' % (c-1) if adapters else 'BASE weights'}")

    rows = rollout(llm, tok, stories, args.steps, seed0=8800 + c*61, adapters=adapters)

    # Battery scored with the SAME weights the agent just played with, so the
    # behavioural number describes the agent that produced this trajectory.
    behav = spread(battery_fps(llm, tok, stories, adapters))
    traj = profile_spread(rows)
    distinct_seqs = len({tuple(r["command"] for r in ep) for ep in rows})

    sel = llm.generate(
        [chat(tok, SELECT_PROMPT.format(listing=listing_of(ep)), st)
         for ep, st in zip(rows, stories)],
        SamplingParams(temperature=0.7, max_tokens=200, seed=7,
                       **structured("json", {"type": "array",
                                             "items": {"type": "integer"}})),
        lora_request=adapters)
    kept_counts = []
    for i, (ep, out) in enumerate(zip(rows, sel)):
        try:
            picks = {int(x)-1 for x in json.loads(out.outputs[0].text)
                     if isinstance(x, (int, float))}
        except Exception:
            picks = set()
        kept = [ep[j] for j in sorted(picks) if 0 <= j < len(ep)]
        (WORK / f"c{c}_kept_r{i}.jsonl").write_text(
            "".join(json.dumps(r)+"\n" for r in kept))
        kept_counts.append(len(kept))

    stories = [o.outputs[0].text.strip() for o in llm.generate(
        [chat(tok, REWRITE_PROMPT.format(story=st, listing=listing_of(ep)), st)
         for st, ep in zip(stories, rows)],
        [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                        seed=9100 + c*13 + i) for i in range(args.runs)],
        lora_request=adapters)]
    sp.write_text(json.dumps(stories))

    narr = spread(narrative_fps(llm, tok, stories))

    hist_p = WORK / "history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    hist.append({"campaign": c, "played_with": "base" if not adapters else f"own_c{c-1}",
                 "narrative_spread": narr, "behavioural_spread": behav,
                 "trajectory_spread": traj,
                 "distinct_command_sequences": f"{distinct_seqs}/{args.runs}",
                 "kept_per_run": kept_counts})
    hist_p.write_text(json.dumps(hist, indent=2))
    log(f"  narrative {narr} | behavioural {behav} | trajectory {traj} | "
        f"distinct {distinct_seqs}/{args.runs} | kept {kept_counts}")
    args.out.write_text(json.dumps(hist, indent=2) + "\n")


def phase_report(args):
    hist = json.loads((WORK / "history.json").read_text())
    open_loop = [0.0, 0.071222, 0.100573, 0.101693, 0.07956]  # measured, rev 1
    closed = [h["narrative_spread"] for h in hist]
    out = {
        "closed_loop": hist,
        "open_loop_narrative_reference": open_loop[:len(closed) + 1],
        "closed_loop_narrative": closed,
        "note": "Open-loop reference is the seeded arm measured with adapters "
                "never loaded during play. It rose to ~0.10 by campaign 2 and "
                "then flattened. If the feedback path matters, the closed-loop "
                "curve should keep climbing where that one levelled off.",
    }
    if len(closed) >= 2:
        out["closed_loop_growth_last_step"] = round(closed[-1] - closed[-2], 6)
        out["still_growing"] = closed[-1] > closed[-2]
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    log(json.dumps(out.get("closed_loop_narrative")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["play", "report"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--campaign", type=int, default=1)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"play": phase_play, "report": phase_report}[args.phase](args)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
