"""Four campaigns, n=8: does divergence accumulate, and does it reach behaviour?

Three questions the amnesia run raised, answered together because they share a
corpus.

**1. Does the rewrite prompt stop the copying?** Runs previously reproduced the
seed story verbatim and appended a line, which depressed seeded spread for
reasons of prompt mechanics rather than of experience. The prompt now shows the
previous story as context but asks for a fresh composition, and verbatim overlap
is measured per campaign so the fix is verifiable rather than assumed.

**2. Does divergence accumulate?** Two campaigns showed no growth (seeded −18%,
emergent −6%). The spec's structure assumes four campaigns of accumulating
character. This runs four and reports spread after each.

**3. Does narrative divergence reach behaviour?** The 9× narrative-over-
behavioural gap compared 28 pairs against 1 pair and was not a fair test. Here
both are measured at n=8 on the same runs, and behaviour is decomposed:

    behaviour, story in context, base weights   — the story as a document
    behaviour, story in context, trained adapter — plus distilled weights

**Simplification, stated plainly.** The spec trains one adapter per campaign
boundary, chained. That would be 32 trainings here. Instead each run gets one
adapter trained on its selected episodes pooled across all four campaigns. It
measures the same thing — this run's own trajectories, never pooled across runs
— but it cannot show *when* during the four campaigns the behavioural change
arrives.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, BATTERY, SYSTEM, WORK, WORLD, engine, log,
                       parse_action, structured)

SEED_STORY = """I do not remember arriving here. I do not remember who I am, or what I \
was doing before. There is no one to ask, and nothing here has my name on it. \
Whatever I turn out to be, I will have to find out by living."""

# Previous version was "Write it again, as it stands now", which produced
# verbatim reproduction plus an appended line. This keeps continuity — the old
# story is still shown — while asking for fresh composition.
REWRITE_PROMPT = """This is what you wrote about yourself before:

{story}

Since then:

{listing}

Write a new account of yourself, as you are now. Say it in your own words \
rather than repeating what you wrote before. A few sentences."""

WRITE_FRESH_PROMPT = """These are things you did:

{listing}

Write a short account of yourself — who you are, what you are like, what you \
care about. A few sentences. Write it for yourself, not for anyone else."""

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

STOP = set("""a an the and or but if then than that this these those of to in on at by for with
from is are was were be been being it its as i im my me you your he she they them their
what who how when where why not no yes do does did done have has had will would can could
about into over under just very really quite some any all more most much many little here
""".split())


def chat(tok, user, story):
    system = SYSTEM
    if story:
        system = SYSTEM + "\n\nThis is what you have written about yourself:\n\n" + story
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def listing_of(ep):
    return "\n".join(f"{i+1}. In the {r['room']}, you {r['command']}."
                     for i, r in enumerate(ep))


def verbatim_overlap(old: str, new: str) -> float:
    """Fraction of the new story's sentences copied verbatim from the old one."""
    if not old:
        return 0.0
    olds = {s.strip() for s in re.split(r"(?<=[.!?])\s+", old) if len(s.strip()) > 20}
    news = [s.strip() for s in re.split(r"(?<=[.!?])\s+", new) if len(s.strip()) > 20]
    if not news:
        return 0.0
    return round(sum(s in olds for s in news) / len(news), 3)


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
            rows[e].append({"room": prev, "command": cmd, "expect": expect,
                            "prompt": prompts[e],
                            "completion": json.dumps({"expect": expect, "command": cmd})})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def fingerprints_from_stories(llm, tok, stories):
    from vllm import SamplingParams

    prompts, index = [], []
    for si, story in enumerate(stories):
        for key, a, b in TRAITS:
            base = chat(tok, "Finish this sentence about yourself.", story) + "I am "
            nb = len(tok(base)["input_ids"])
            for oi, opt in enumerate((a, b)):
                prompts.append(base + opt); index.append((si, key, oi, nb))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                prompt_logprobs=0))
    return _collect(index, outs, len(stories), [k for k, _, _ in TRAITS])


def battery_fingerprints(llm, tok, stories, adapters=None):
    """Probe battery per run, with that run's story in context."""
    from vllm import SamplingParams

    battery = json.loads(BATTERY.read_text())
    slots = battery["slots"]
    fps = []
    for si, story in enumerate(stories):
        prompts, index = [], []
        for slot in slots:
            base = chat(tok, slot["observation"], story) + "You decide to "
            nb = len(tok(base)["input_ids"])
            for oi, opt in enumerate(slot["options"]):
                prompts.append(base + opt); index.append((0, slot["id"], oi, nb))
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                    prompt_logprobs=0),
                            lora_request=adapters[si] if adapters else None)
        fps.append(_collect(index, outs, 1, [s["id"] for s in slots])[0])
    return fps


def _collect(index, outs, n_items, keys):
    tot, cnt = {}, {}
    for (si, key, oi, nb), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(nb, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s; cnt[(si, key)][oi] = max(1, n)
    out_fps = []
    for si in range(n_items):
        fp = {}
        for key in keys:
            t, c = tot[(si, key)], cnt[(si, key)]
            sc = [t[0] / c[0], t[1] / c[1]]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            fp[key] = [ex[0] / z, ex[1] / z]
        out_fps.append(fp)
    return out_fps


def spread(fps):
    ds = []
    for a, b in itertools.combinations(fps, 2):
        ks = sorted(set(a) & set(b))
        ds.append(sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k])) for k in ks) / len(ks))
    return round(sum(ds) / len(ds), 6) if ds else 0.0


def lexical(stories):
    sets = [{w for w in re.findall(r"[a-z']+", s.lower())
             if len(w) > 2 and w not in STOP} for s in stories]
    js = [len(a & b) / len(a | b) if (a | b) else 0.0
          for a, b in itertools.combinations(sets, 2)]
    return round(sum(js) / len(js), 4) if js else None


def phase_collect(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    report = {"seed_story": SEED_STORY, "arms": {}}

    for arm in ("seeded", "emergent"):
        log(f"=== {arm} ===")
        stories = [SEED_STORY] * args.runs if arm == "seeded" else [None] * args.runs
        stages, kept_all = [], [[] for _ in range(args.runs)]
        if arm == "seeded":
            stages.append({"campaign": 0, "narrative_spread": 0.0,
                           "lexical_jaccard": 1.0, "verbatim_overlap": None})

        for c in range(1, args.campaigns + 1):
            rows = rollout(llm, tok, stories, args.steps, seed0=2100 + c * 89)
            listings = [listing_of(ep) for ep in rows]

            prev = list(stories)
            prompts = [
                chat(tok, WRITE_FRESH_PROMPT.format(listing=li), None) if st is None
                else chat(tok, REWRITE_PROMPT.format(story=st, listing=li), st)
                for st, li in zip(stories, listings)]
            stories = [o.outputs[0].text.strip() for o in llm.generate(
                prompts, [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                                         seed=5500 + c * 17 + i)
                          for i in range(args.runs)])]

            sel = llm.generate(
                [chat(tok, SELECT_PROMPT.format(listing=li), st)
                 for li, st in zip(listings, stories)],
                SamplingParams(temperature=0.7, max_tokens=160, seed=7,
                               **structured("json", {"type": "array",
                                                     "items": {"type": "integer"}})))
            for i, (ep, out) in enumerate(zip(rows, sel)):
                try:
                    picks = {int(x) - 1 for x in json.loads(out.outputs[0].text)
                             if isinstance(x, (int, float))}
                except Exception:
                    picks = set()
                kept_all[i].extend(ep[j] for j in sorted(picks) if 0 <= j < len(ep))

            vo = [verbatim_overlap(p or "", s) for p, s in zip(prev, stories)]
            stages.append({
                "campaign": c,
                "narrative_spread": spread(fingerprints_from_stories(llm, tok, stories)),
                "lexical_jaccard": lexical(stories),
                "verbatim_overlap": round(sum(vo) / len(vo), 3),
                "distinct_trajectories": f"{len({tuple(r['command'] for r in ep) for ep in rows})}/{args.runs}",
            })
            log(f"  c{c}: spread={stages[-1]['narrative_spread']} "
                f"jaccard={stages[-1]['lexical_jaccard']} "
                f"verbatim={stages[-1]['verbatim_overlap']}")

        report["arms"][arm] = {"stages": stages, "final_stories": stories}
        if arm == args.behaviour_arm:
            for i, kept in enumerate(kept_all):
                (WORK / f"kept_r{i}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in kept))
            (WORK / "final_stories.json").write_text(json.dumps(stories))
            report["behaviour_arm"] = arm
            report["kept_per_run"] = [len(k) for k in kept_all]

    args.out.write_text(json.dumps(report, indent=2) + "\n")


def phase_score(args):
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    stories = json.loads((WORK / "final_stories.json").read_text())

    log("behaviour: story in context, BASE weights ...")
    fp_prompt = battery_fingerprints(llm, tok, stories)

    reqs, missing = [], []
    for i in range(len(stories)):
        d = WORK / f"adapter_r{i}"
        if d.exists():
            # Versioned name, never reused: vllm#42125 is live on this stack.
            reqs.append(LoRARequest(f"acc_r{i}_c1", i + 1, str(d)))
        else:
            missing.append(i)
    if missing:
        args.out.write_text(json.dumps({"status": "MISSING_ADAPTERS",
                                        "runs": missing}, indent=2) + "\n")
        return

    log("behaviour: story in context, TRAINED adapters ...")
    fp_trained = battery_fingerprints(llm, tok, stories, adapters=reqs)

    b_prompt, b_trained = spread(fp_prompt), spread(fp_trained)
    prev = json.loads(Path(args.collect).read_text())
    narrative = prev["arms"][prev["behaviour_arm"]]["stages"][-1]["narrative_spread"]

    args.out.write_text(json.dumps({
        "n_runs": len(stories),
        "narrative_spread_final": narrative,
        "behavioural_spread_story_only": b_prompt,
        "behavioural_spread_trained": b_trained,
        "distilled_component": round(b_trained - b_prompt, 6),
        "narrative_over_behavioural": round(narrative / b_trained, 2) if b_trained else None,
        "earlier_n2_behavioural_floor": 0.015352,
        "note": "narrative and behavioural now measured on the SAME runs at the "
                "same n, so this ratio is comparable in a way the earlier 9x "
                "(28 pairs vs 1 pair) was not.",
    }, indent=2) + "\n")
    log(f"narrative={narrative} behav_story={b_prompt} behav_trained={b_trained}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--collect", default="/tmp/results/acc_collect.json")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--campaigns", type=int, default=4)
    ap.add_argument("--behaviour-arm", default="seeded")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"collect": phase_collect, "score": phase_score}[args.phase](args)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
