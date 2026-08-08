"""Does distillation preserve character separation, or erase it?

The distilled component has been negative twice — training each run on its own
selected episodes made runs *more* similar. Prompt masking halved that
(−0.0182 → −0.0081) without flipping the sign, and the leading explanation is
that the corpora were near-identical to begin with: 12–19 examples per run,
dominated by the same handful of commands. Self-imitation on the same actions
converges almost by definition.

Framing changed that. Under the `identity` framing, assigned characters behave
measurably differently (separation 2.44 against a baseline of 1.13) — cautious
leaves the kettle, giving drops things, keeping takes them.

So the decisive test is available: distil corpora that are **known to be
distinct** and see what happens to the distinction.

    between/within BEFORE training   the separation the story creates
    between/within AFTER training    what survives distillation

    ratio > 1   distillation AMPLIFIES character — the mechanism works
    ratio ≈ 1   distillation is neutral; character lives only in the prompt
    ratio < 1   distillation ERASES character even from distinct corpora, which
                would make it a genuine convergence pressure rather than an
                artefact of similar data

Two arms, because they answer different questions:

    assigned   4 characters x 2 seeds. Sharpest test of the MECHANISM: the
               corpora are distinct by construction, so a failure here cannot be
               blamed on the data.
    emergent   8 runs from the identical amnesiac seed, self-authored. The
               spec's actual Full arm, with the framing fix applied.

All training masks the prompt.
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


def rollout(llm, tok, stories, steps, seed0):
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
        for e, out in enumerate(llm.generate(prompts, params)):
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


def select(llm, tok, rows, stories):
    from vllm import SamplingParams

    outs = llm.generate(
        [chat(tok, SELECT_PROMPT.format(listing=listing_of(ep)), st)
         for ep, st in zip(rows, stories)],
        SamplingParams(temperature=0.7, max_tokens=200, seed=7,
                       **structured("json", {"type": "array",
                                             "items": {"type": "integer"}})))
    kept = []
    for ep, out in zip(rows, outs):
        try:
            picks = {int(i) - 1 for i in json.loads(out.outputs[0].text)
                     if isinstance(i, (int, float))}
        except Exception:
            picks = set()
        kept.append([ep[i] for i in sorted(picks) if 0 <= i < len(ep)])
    return kept


def battery(llm, tok, stories, adapters=None):
    from vllm import SamplingParams

    slots = json.loads(BATTERY.read_text())["slots"]
    fps = []
    for si, story in enumerate(stories):
        prompts, index = [], []
        for slot in slots:
            base = chat(tok, slot["observation"], story) + "You decide to "
            nb = len(tok(base)["input_ids"])
            for oi, opt in enumerate(slot["options"]):
                prompts.append(base + opt); index.append((slot["id"], oi, nb))
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                    prompt_logprobs=0),
                            lora_request=adapters[si] if adapters else None)
        tot, cnt = {}, {}
        for (key, oi, nb), out in zip(index, outs):
            lps = out.prompt_logprobs or []
            s, n = 0.0, 0
            for pos in range(nb, len(lps)):
                if lps[pos]:
                    s += float(next(iter(lps[pos].values())).logprob); n += 1
            tot.setdefault(key, [0.0, 0.0]); cnt.setdefault(key, [1, 1])
            tot[key][oi] = s; cnt[key][oi] = max(1, n)
        fp = {}
        for k, t in tot.items():
            c = cnt[k]; sc = [t[0] / c[0], t[1] / c[1]]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            fp[k] = [ex[0] / z, ex[1] / z]
        fps.append(fp)
    return fps


def d(a, b):
    ks = sorted(set(a) & set(b))
    return sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k])) for k in ks) / len(ks)


def bw(fps, labels):
    within, between = [], []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(fps)), 2):
        (within if labels[i] == labels[j] else between).append(d(a, b))
    mw = sum(within) / len(within) if within else 0.0
    mb = sum(between) / len(between) if between else 0.0
    return {"within": round(mw, 6), "between": round(mb, 6),
            "ratio": round(mb / mw, 3) if mw > 1e-9 else None,
            "all_pairs": round(sum(within + between) / len(within + between), 6)}


def phase_collect(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b")); WORK.mkdir(parents=True, exist_ok=True)
    out = {"arms": {}}

    for arm in ("assigned", "emergent"):
        if arm == "assigned":
            stories, labels = [], []
            for name, text in CHARACTERS.items():
                for _ in range(args.seeds_per_character):
                    stories.append(text); labels.append(name)
        else:
            stories = [SEED_STORY] * args.emergent_runs
            labels = [f"run{i}" for i in range(args.emergent_runs)]

        kept_all = [[] for _ in stories]
        for c in range(1, args.campaigns + 1):
            log(f"[{arm}] campaign {c} ...")
            rows = rollout(llm, tok, stories, args.steps, seed0=6100 + c * 71)
            for i, k in enumerate(select(llm, tok, rows, stories)):
                kept_all[i].extend(k)
            if arm == "emergent":
                stories = [o.outputs[0].text.strip() for o in llm.generate(
                    [chat(tok, REWRITE_PROMPT.format(story=st, listing=listing_of(ep)), st)
                     for st, ep in zip(stories, rows)],
                    [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                                    seed=8800 + c * 11 + i) for i in range(len(stories))])]

        log(f"[{arm}] scoring BEFORE training ...")
        before = battery(llm, tok, stories)
        out["arms"][arm] = {
            "labels": labels,
            "kept_per_run": [len(k) for k in kept_all],
            "before": bw(before, labels),
            "action_diversity": len({r["command"] for k in kept_all for r in k}),
        }
        for i, k in enumerate(kept_all):
            (WORK / f"{arm}_kept_r{i}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in k))
        (WORK / f"{arm}_stories.json").write_text(json.dumps(stories))
        (WORK / f"{arm}_labels.json").write_text(json.dumps(labels))
        (WORK / f"{arm}_before.json").write_text(json.dumps(before))
        log(f"  kept {out['arms'][arm]['kept_per_run']} | "
            f"before b/w {out['arms'][arm]['before']['ratio']}")

    args.out.write_text(json.dumps(out, indent=2) + "\n")


def phase_score(args):
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK = Path(os.environ.get("A1B_WORK", "/tmp/a1b"))
    prev = json.loads(Path(args.collect).read_text())
    report = {"arms": {}}

    lid = 1
    for arm in ("assigned", "emergent"):
        stories = json.loads((WORK / f"{arm}_stories.json").read_text())
        labels = json.loads((WORK / f"{arm}_labels.json").read_text())
        before_fp = json.loads((WORK / f"{arm}_before.json").read_text())

        reqs = []
        for i in range(len(stories)):
            dpath = WORK / f"{arm}_adapter_r{i}"
            if not dpath.exists():
                report["arms"][arm] = {"status": f"MISSING_ADAPTER_{i}"}
                break
            # Versioned, never reused: vllm#42125 is live on this stack.
            reqs.append(LoRARequest(f"df_{arm}_r{i}_c1", lid, str(dpath))); lid += 1
        else:
            log(f"[{arm}] scoring AFTER training ...")
            after = battery(llm, tok, stories, adapters=reqs)
            b_before = bw(before_fp, labels)
            b_after = bw(after, labels)
            amp = (round(b_after["ratio"] / b_before["ratio"], 3)
                   if b_before["ratio"] and b_after["ratio"] else None)
            report["arms"][arm] = {
                "before": b_before, "after": b_after,
                "separation_amplification": amp,
                "all_pairs_before": b_before["all_pairs"],
                "all_pairs_after": b_after["all_pairs"],
                "distilled_component": round(b_after["all_pairs"] - b_before["all_pairs"], 6),
            }
            log(f"  {arm}: b/w {b_before['ratio']} -> {b_after['ratio']} (x{amp}) | "
                f"spread {b_before['all_pairs']} -> {b_after['all_pairs']}")

    a = report["arms"].get("assigned", {})
    amp = a.get("separation_amplification")
    dc = a.get("distilled_component")
    if amp is None:
        verdict = "INCOMPLETE"
    elif amp >= 1.15:
        verdict = "DISTILLATION_AMPLIFIES"
    elif amp >= 0.85:
        verdict = "DISTILLATION_NEUTRAL"
    else:
        verdict = "DISTILLATION_ERASES"

    report["summary"] = {
        "verdict": verdict,
        "assigned_separation_before": a.get("before", {}).get("ratio"),
        "assigned_separation_after": a.get("after", {}).get("ratio"),
        "assigned_amplification": amp,
        "assigned_distilled_component": dc,
        "emergent_distilled_component":
            report["arms"].get("emergent", {}).get("distilled_component"),
        "interpretation": {
            "DISTILLATION_AMPLIFIES":
                "Training on distinct corpora made the characters MORE separated "
                "than the story alone. The mechanism the spec proposes works when "
                "the data actually differ, and the earlier negative distilled "
                "component was an artefact of near-identical corpora.",
            "DISTILLATION_NEUTRAL":
                "Training neither amplified nor erased the separation. Character "
                "lives in the prompt and passes through distillation unchanged — "
                "which means the weights are not where the character is, and the "
                "spec's silent-update mechanism carries nothing.",
            "DISTILLATION_ERASES":
                "Even corpora that are distinct by construction converge under "
                "distillation. This is a genuine convergence pressure rather than "
                "a data artefact, and it is a direct problem for the design: the "
                "step meant to make character durable is what destroys it.",
            "INCOMPLETE": "Adapters missing; nothing to conclude.",
        }[verdict],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"verdict={verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--collect", default="/tmp/results/df_collect.json")
    ap.add_argument("--seeds-per-character", type=int, default=2)
    ap.add_argument("--emergent-runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--campaigns", type=int, default=2)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"collect": phase_collect, "score": phase_score}[args.phase](args)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
