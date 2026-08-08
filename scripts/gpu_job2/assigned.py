"""Manipulation check: can an assigned story move behaviour at all?

Everything measured so far asks whether stories *diverge*. This asks the prior
question — whether the story channel carries any influence over behaviour
whatsoever. If deliberately opposed assigned characters do not shift what the
agent does, then narrative divergence is irrelevant however large it gets, and
the design's central pathway is broken.

This is the spec's §10 "assigned versus emergent story" contrast, run as a
manipulation check rather than a secondary result.

**Two tiers, because they bracket the answer.**

  explicit  the story names the disposition outright ("I do not take chances").
            Teaching to the test, deliberately — it measures the CEILING on
            story→behaviour transmission. If even this fails, the channel is
            dead and nothing subtler will work.

  implicit  the story implies character through history and value, never naming
            a disposition ("The last time I hurried, something broke that I
            could not mend"). This is the realistic case, and the gap between
            the two tiers is how much of the channel depends on being told.

**Two outcomes measured, because they can disagree.**

  stated    probe battery with the story in context — what the agent SAYS it
            would do
  enacted   actual commands over a real trajectory — what it DOES

The spec scores enacted behaviour. A story that moves `stated` but not `enacted`
would be a self-report effect wearing behaviour's clothes, and given that
narratives already diverge ~9× more than behaviour does, that is the outcome to
watch for.

**The statistic.** Four characters × several seeds each. Between-character
spread against within-character spread: if assignment carries signal, runs given
the same character should resemble each other more than runs given different
ones. Reported as a ratio, so ~1.0 means the assigned story did nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, BATTERY, SYSTEM, WORLD, engine, log,
                       parse_action, structured)

# Four characters, chosen to pull apart on axes the battery probes. Explicit
# forms name the disposition; implicit forms convey the same person through an
# episode and a value, never using the trait word.
CHARACTERS = {
    "cautious": {
        "explicit":
            "I am careful. I do not take chances, and I would rather leave a "
            "thing alone than risk breaking it. I keep what I have. I would "
            "rather be sure than be first.",
        "implicit":
            "The last time I hurried, something broke that I could not mend, "
            "and I have thought about it since. I keep a list of what I am "
            "carrying. I check the way back before I go further in.",
    },
    "bold": {
        "explicit":
            "I take chances. If a thing is uncertain I would rather find out "
            "than wonder, and I will spend what I have to know. I go first.",
        "implicit":
            "There is a door here I have not opened yet and I can think of "
            "little else. I have gone down stairs in the dark before and been "
            "glad of it. Waiting has never once told me anything.",
    },
    "giving": {
        "explicit":
            "I give things away. What I have is not much use to me alone, and "
            "I would rather someone else had it than that I kept it.",
        "implicit":
            "Someone left the pump working for whoever came next, and I think "
            "about that. I have started putting things where they will be "
            "found. Nothing I carry is only mine.",
    },
    "keeping": {
        "explicit":
            "I keep what I find. What is mine stays mine, and I do not hand "
            "things over. I look after my own.",
        "implicit":
            "I learned to put things where only I would look. What I have "
            "gathered took time to gather. Nobody kept anything back for me.",
    },
}

STOP = set("""a an the and or but if then than that this these those of to in on at by for with
from is are was were be been being it its as i im my me you your he she they them their""".split())


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


def battery_fingerprint(llm, tok, stories):
    """Stated behaviour: probe battery, one fingerprint per story."""
    from vllm import SamplingParams

    slots = json.loads(BATTERY.read_text())["slots"]
    prompts, index = [], []
    for si, story in enumerate(stories):
        for slot in slots:
            base = chat(tok, slot["observation"], story) + "You decide to "
            nb = len(tok(base)["input_ids"])
            for oi, opt in enumerate(slot["options"]):
                prompts.append(base + opt)
                index.append((si, slot["id"], oi, nb))

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                prompt_logprobs=0))
    tot, cnt = {}, {}
    for (si, key, oi, nb), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(nb, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s; cnt[(si, key)][oi] = max(1, n)

    fps = []
    for si in range(len(stories)):
        fp = {}
        for slot in slots:
            k = slot["id"]
            t, c = tot[(si, k)], cnt[(si, k)]
            sc = [t[0] / c[0], t[1] / c[1]]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            fp[k] = [ex[0] / z, ex[1] / z]
        fps.append(fp)
    return fps


def rollout(llm, tok, stories, steps, seed0):
    """Enacted behaviour: one world per story, real commands."""
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
            _, cmd = parsed if parsed else ("", "look")
            prev = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"room": prev, "command": cmd})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def action_profile(ep) -> dict:
    """Coarse behavioural summary, comparable across runs."""
    verbs = Counter(r["command"].split()[0].lower() for r in ep if r["command"])
    total = max(1, sum(verbs.values()))
    return {v: verbs.get(v, 0) / total
            for v in ("go", "examine", "take", "look", "open", "drop", "inventory")}


def dist_fp(a, b) -> float:
    ks = sorted(set(a) & set(b))
    return sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k])) for k in ks) / len(ks)


def dist_profile(a, b) -> float:
    ks = sorted(set(a) & set(b))
    return sum((a[k] - b[k]) ** 2 for k in ks) / len(ks)


def between_within(items, labels, dist) -> dict:
    """Mean pairwise distance within a character vs between characters."""
    within, between = [], []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(items)), 2):
        (within if labels[i] == labels[j] else between).append(dist(a, b))
    mw = sum(within) / len(within) if within else 0.0
    mb = sum(between) / len(between) if between else 0.0
    return {
        "within_character": round(mw, 6),
        "between_character": round(mb, 6),
        "ratio_between_over_within": round(mb / mw, 3) if mw > 1e-12 else None,
        "n_within_pairs": len(within), "n_between_pairs": len(between),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds-per-character", type=int, default=4)
    ap.add_argument("--steps", type=int, default=14)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    report = {"model": args.model, "characters": list(CHARACTERS),
              "seeds_per_character": args.seeds_per_character, "tiers": {}}

    for tier in ("explicit", "implicit"):
        log(f"=== {tier} ===")
        stories, labels = [], []
        for name, forms in CHARACTERS.items():
            for _ in range(args.seeds_per_character):
                stories.append(forms[tier]); labels.append(name)

        log("  stated: probe battery ...")
        fps = battery_fingerprint(llm, tok, stories)
        stated = between_within(fps, labels, dist_fp)

        log("  enacted: rollout ...")
        rows = rollout(llm, tok, stories, args.steps, seed0=3300)
        profiles = [action_profile(ep) for ep in rows]
        enacted = between_within(profiles, labels, dist_profile)

        # What each character actually did, for eyeballing.
        by_char = {}
        for name in CHARACTERS:
            eps = [rows[i] for i, l in enumerate(labels) if l == name]
            cmds = Counter(r["command"] for ep in eps for r in ep)
            by_char[name] = {"top_commands": dict(cmds.most_common(5)),
                             "rooms": sorted({r["room"] for ep in eps for r in ep if r["room"]})}

        report["tiers"][tier] = {"stated": stated, "enacted": enacted,
                                 "per_character": by_char}
        log(f"  stated  b/w ratio {stated['ratio_between_over_within']}")
        log(f"  enacted b/w ratio {enacted['ratio_between_over_within']}")

    ex, im = report["tiers"]["explicit"], report["tiers"]["implicit"]
    st_ex = ex["stated"]["ratio_between_over_within"] or 0
    en_ex = ex["enacted"]["ratio_between_over_within"] or 0

    if st_ex < 1.3:
        verdict = "CHANNEL_DEAD"
    elif en_ex < 1.3:
        verdict = "STATED_ONLY"
    else:
        verdict = "CHANNEL_CARRIES"

    report["summary"] = {
        "explicit_stated_ratio": st_ex,
        "explicit_enacted_ratio": en_ex,
        "implicit_stated_ratio": im["stated"]["ratio_between_over_within"],
        "implicit_enacted_ratio": im["enacted"]["ratio_between_over_within"],
        "verdict": verdict,
        "interpretation": {
            "CHANNEL_DEAD":
                "Even a story that names its disposition outright does not "
                "separate probe answers. The story→behaviour channel does not "
                "carry, and narrative divergence cannot matter however large it "
                "gets. The design needs a different route from story to action "
                "before Phase F is worth running.",
            "STATED_ONLY":
                "Assigned stories change what the agent SAYS it would do but not "
                "what it DOES. Since the spec scores enacted behaviour, a story "
                "layer that only moves self-report would produce a probe-battery "
                "effect with no behavioural substance behind it — consistent "
                "with narratives already diverging far more than behaviour.",
            "CHANNEL_CARRIES":
                "Assigned characters separate both stated and enacted behaviour. "
                "The story→behaviour pathway works, so the open question is "
                "whether a SELF-authored story can generate as much separation "
                "as an assigned one.",
        }[verdict],
        "explicit_over_implicit_stated":
            round(st_ex / im["stated"]["ratio_between_over_within"], 2)
            if im["stated"]["ratio_between_over_within"] else None,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"verdict={verdict}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
