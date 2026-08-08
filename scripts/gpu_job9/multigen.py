"""Many generations: does per-step contraction compound, settle, or lose to drift?

Each generation does two opposing things. Play **injects** variance — sampling
produces different trajectories. Distillation **contracts** — every configuration
measured so far pulls runs together, by 0.9 to 4.4 points depending on how
similar the corpora already are.

Two generations cannot distinguish which dominates. The closed-loop run managed
only two (its loop did not engage until campaign 3), and its numbers were flat
enough to be noise. Eight generations separate three futures that look identical
early:

    contraction mapping   spread decays toward zero — every run becomes the
                          same agent, and the spec's claim is dead
    equilibrium           spread settles on a non-zero plateau — injection
                          balances contraction, giving a bounded but real
                          character
    slow divergence       spread grows despite per-step contraction — a random
                          walk that accumulates, which IS path dependence

**Fix over the closed-loop run.** There, `adapters_for` returned None if *any*
run lacked an adapter, so four runs keeping nothing at campaign 1 silently put
*everyone* back on base weights for campaign 2. Adapter presence is now per-run:
runs that have one play with it, runs that do not play on base, in the same step,
by splitting the batch. A run that keeps nothing simply does not update that
generation — it does not drag the others back with it.
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


def per_run_adapters(gen: int, n: int):
    """LoRARequest or None per run — presence is individual, not all-or-nothing."""
    if gen <= 1:
        return [None] * n
    from vllm.lora.request import LoRARequest

    out = []
    for i in range(n):
        p = WORK / f"adapter_r{i}_g{gen-1}"
        # Versioned per generation and never reused: vllm#42125 keys the KV
        # prefix cache on the adapter name without invalidating on reload.
        out.append(LoRARequest(f"mg_r{i}_g{gen-1}", i + 1, str(p)) if p.exists() else None)
    return out


def grouped_generate(llm, prompts, params, adapters):
    """Generate with per-request adapters by splitting into homogeneous groups.

    vLLM takes one lora_request per call, so requests are grouped by adapter and
    results reassembled in the original order.
    """
    order, results = {}, [None] * len(prompts)
    for idx, a in enumerate(adapters):
        order.setdefault(id(a) if a is not None else None, []).append(idx)
    for _, idxs in order.items():
        a = adapters[idxs[0]]
        outs = llm.generate([prompts[i] for i in idxs],
                            [params[i] for i in idxs], lora_request=a)
        for i, o in zip(idxs, outs):
            results[i] = o
    return results


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

        for e, out in enumerate(grouped_generate(llm, prompts, params, adapters)):
            parsed = parse_action(out.outputs[0].text)
            expect, cmd = parsed if parsed else ("", "look")
            prev = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"room": prev, "command": cmd,
                            "verb": cmd.split()[0].lower() if cmd else "",
                            "prompt": prompts[e],
                            "completion": json.dumps({"expect": expect, "command": cmd})})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def _score(llm, tok, sets, adapters):
    from vllm import SamplingParams

    prompts, index, reqs = [], [], []
    for si, keyed in enumerate(sets):
        for key, base, opts in keyed:
            nb = len(tok(base)["input_ids"])
            for oi, o in enumerate(opts):
                prompts.append(base + o); index.append((si, key, oi, nb))
                reqs.append(adapters[si] if adapters else None)
    params = [SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)] * len(prompts)
    outs = grouped_generate(llm, prompts, params, reqs)
    tot, cnt = {}, {}
    for (si, key, oi, nb), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(nb, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s; cnt[(si, key)][oi] = max(1, n)
    fps = [{} for _ in sets]
    for (si, key), t in tot.items():
        c = cnt[(si, key)]; sc = [t[0]/c[0], t[1]/c[1]]
        m = max(sc); ex = [math.exp(v-m) for v in sc]; z = sum(ex)
        fps[si][key] = [ex[0]/z, ex[1]/z]
    return fps


def spread(fps):
    ds = []
    for a, b in itertools.combinations(fps, 2):
        ks = sorted(set(a) & set(b))
        ds.append(sum(sum((x-y)**2 for x, y in zip(a[k], b[k])) for k in ks)/len(ks))
    return round(sum(ds)/len(ds), 6) if ds else 0.0


def profile_spread(rows):
    ps = []
    for ep in rows:
        c = Counter(r["verb"] for r in ep)
        t = max(1, sum(c.values()))
        ps.append({v: c.get(v, 0)/t for v in VERBS})
    ds = [sum((a[k]-b[k])**2 for k in VERBS)/len(VERBS)
          for a, b in itertools.combinations(ps, 2)]
    return round(sum(ds)/len(ds), 6) if ds else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    g = args.gen

    sp = WORK / "stories.json"
    stories = json.loads(sp.read_text()) if sp.exists() else [SEED_STORY] * args.runs

    adapters = per_run_adapters(g, args.runs)
    n_with = sum(a is not None for a in adapters)
    log(f"generation {g}: {n_with}/{args.runs} runs playing on their own weights")

    rows = rollout(llm, tok, stories, args.steps, seed0=6400 + g*37, adapters=adapters)

    behav = spread(_score(llm, tok, [[
        (s["id"], chat(tok, s["observation"], st) + "You decide to ", s["options"])
        for s in json.loads(BATTERY.read_text())["slots"]] for st in stories], adapters))
    traj = profile_spread(rows)
    distinct = len({tuple(r["command"] for r in ep) for ep in rows})

    sel = grouped_generate(
        llm,
        [chat(tok, SELECT_PROMPT.format(listing=listing_of(ep)), st)
         for ep, st in zip(rows, stories)],
        [SamplingParams(temperature=0.7, max_tokens=200, seed=7,
                        **structured("json", {"type": "array",
                                              "items": {"type": "integer"}}))] * args.runs,
        adapters)
    kept_counts = []
    for i, (ep, out) in enumerate(zip(rows, sel)):
        try:
            picks = {int(x)-1 for x in json.loads(out.outputs[0].text)
                     if isinstance(x, (int, float))}
        except Exception:
            picks = set()
        kept = [ep[j] for j in sorted(picks) if 0 <= j < len(ep)]
        (WORK / f"g{g}_kept_r{i}.jsonl").write_text(
            "".join(json.dumps(r)+"\n" for r in kept))
        kept_counts.append(len(kept))

    stories = [o.outputs[0].text.strip() for o in grouped_generate(
        llm,
        [chat(tok, REWRITE_PROMPT.format(story=st, listing=listing_of(ep)), st)
         for st, ep in zip(stories, rows)],
        [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                        seed=7300 + g*17 + i) for i in range(args.runs)],
        adapters)]
    sp.write_text(json.dumps(stories))

    narr = spread(_score(llm, tok, [[
        (k, chat(tok, "Finish this sentence about yourself.", st) + "I am ", (a, b))
        for k, a, b in TRAITS] for st in stories], None))

    hp = WORK / "history.json"
    hist = json.loads(hp.read_text()) if hp.exists() else []
    hist.append({"generation": g, "runs_on_own_weights": n_with,
                 "narrative_spread": narr, "behavioural_spread": behav,
                 "trajectory_spread": traj,
                 "distinct_sequences": f"{distinct}/{args.runs}",
                 "kept_per_run": kept_counts})
    hp.write_text(json.dumps(hist, indent=2))
    args.out.write_text(json.dumps(hist, indent=2) + "\n")
    log(f"  narr {narr} | behav {behav} | traj {traj} | distinct {distinct}/{args.runs} "
        f"| kept {kept_counts}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
