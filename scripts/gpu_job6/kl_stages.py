"""Collect per-campaign corpora, then score the SFT and KL adapter chains.

Corpora are collected once and both variants train on exactly the same data, so
the only difference between the two results is the KL term.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
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
            c = cnt[k]; sc = [t[0]/c[0], t[1]/c[1]]
            m = max(sc); ex = [math.exp(v-m) for v in sc]; z = sum(ex)
            fp[k] = [ex[0]/z, ex[1]/z]
        fps.append(fp)
    return fps


def spread(fps):
    ds = []
    for a, b in itertools.combinations(fps, 2):
        ks = sorted(set(a) & set(b))
        ds.append(sum(sum((x-y)**2 for x, y in zip(a[k], b[k])) for k in ks)/len(ks))
    return round(sum(ds)/len(ds), 6) if ds else 0.0


def phase_collect(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    stories = [SEED_STORY] * args.runs
    summary = {"campaigns": {}}

    for c in range(1, args.campaigns + 1):
        log(f"campaign {c} ...")
        rows = rollout(llm, tok, stories, args.steps, seed0=7700 + c*53)
        sel = llm.generate(
            [chat(tok, SELECT_PROMPT.format(listing=listing_of(ep)), st)
             for ep, st in zip(rows, stories)],
            SamplingParams(temperature=0.7, max_tokens=200, seed=7,
                           **structured("json", {"type": "array",
                                                 "items": {"type": "integer"}})))
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
                            seed=9900 + c*7 + i) for i in range(args.runs)])]
        summary["campaigns"][c] = {"kept_per_run": kept_counts}
        log(f"  kept {kept_counts}")

    (WORK / "stories.json").write_text(json.dumps(stories))
    summary["untrained_spread"] = spread(battery(llm, tok, stories))
    log(f"untrained spread (story only) = {summary['untrained_spread']}")
    args.out.write_text(json.dumps(summary, indent=2) + "\n")


def phase_score(args):
    """Score both chains from their manifests.

    Reads each variant's final-adapter manifest rather than assuming a filename
    per campaign: a run that selected nothing carries its previous adapter
    forward, so its final adapter may come from an earlier campaign.
    """
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    stories = json.loads((WORK / "stories.json").read_text())

    report = {"variants": {}}
    lid = 1
    manifests = {}
    for tag in ("sft", "kl"):
        f = WORK / f"{tag}_final.json"
        manifests[tag] = json.loads(f.read_text()) if f.exists() else {}

    # Score only runs that BOTH variants produced an adapter for, so the two
    # numbers are computed over an identical subset.
    common = sorted(set(manifests["sft"]) & set(manifests["kl"]), key=int)
    log(f"runs with adapters in both variants: {len(common)} -> {common}")
    if len(common) < 3:
        report["status"] = "TOO_FEW_COMMON_RUNS"
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return

    sub = [stories[int(i)] for i in common]
    untrained = spread(battery(llm, tok, sub))
    report["untrained_spread_same_subset"] = untrained
    report["runs_scored"] = common

    for tag in ("sft", "kl"):
        reqs = []
        for i in common:
            reqs.append(LoRARequest(f"{tag}_r{i}_final", lid, manifests[tag][i]))
            lid += 1
        log(f"scoring {tag} ...")
        after = spread(battery(llm, tok, sub, adapters=reqs))
        report["variants"][tag] = {
            "spread_trained": after,
            "distilled_component": round(after - untrained, 6),
            "retention": round(after / untrained, 3) if untrained else None,
        }
        log(f"  {tag}: {untrained} -> {after}  ({after - untrained:+.6f})")

    s_, k_ = report["variants"]["sft"], report["variants"]["kl"]
    report["summary"] = {
        "n_runs": len(common),
        "untrained_spread": untrained,
        "sft_distilled": s_["distilled_component"],
        "kl_distilled": k_["distilled_component"],
        "kl_minus_sft": round(k_["distilled_component"] - s_["distilled_component"], 6),
        "sft_retention": s_["retention"],
        "kl_retention": k_["retention"],
        "reading": ("KL preserves more character than plain SFT"
                    if k_["distilled_component"] > s_["distilled_component"]
                    else "KL did not preserve more than plain SFT"),
        "note": "Campaign 1 is shared between variants (plain SFT, no KL), so the "
                "chains are identical up to the point where the objective "
                "differs. KL applies only where the reference is run-specific.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--collect", default="/tmp/results/kl_collect.json")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--campaigns", type=int, default=2)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"collect": phase_collect, "score": phase_score}[args.phase](args)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
