"""Can the framing give the story authority over action — without collapsing it
into a memory device?

The measured problem: an assigned story moves what the agent *says* by 8.4× the
behavioural floor and what it *does* by essentially nothing (between/within
1.07, where 1.0 is no effect). The story is present in context and inert.

One fix is to make the framing itself grant the artifact authority over action.
But *what* the agent is told to consult it for decides whether this rescues the
design or destroys it, and the spec already draws the line (§5):

    ledger   what happened, what worked   → retrieval → COMPETENCE
    story    who I am, what I'm like      → disposition → CHARACTER

§8 states the consequence outright: a probe with a correct answer measures
competence, and **competence converges**. So a framing that says "read it to
remember what helped" builds a memory device, and every agent should converge on
the same effective moves — more alike, not less.

Three framings, identical in every other respect:

    generic   the current baseline. Measured at 1.07.
    efficacy  "…it tells you what you did before and what helped."
    identity  "…you do not reason it out. You read what you wrote about
              yourself, and you act like that person."

Two measurements, because the second is what tells us whether the worry is real:

    separation   between-character vs within-character action distance. Does the
                 framing make assigned characters behave differently at all?
    convergence  between-character distance over the FIRST half of a trajectory
                 versus the SECOND half. If a framing produces competence rather
                 than character, characters should start apart and end together,
                 so late/early < 1.

The outcome to want is **separation without convergence**. Separation *with*
convergence is the memory device revealing itself.

Caveat to carry into any writeup: both non-generic framings are
experimenter-imposed, and `identity` is close to instructing the agent to have a
character. That weakens any positive result from "a character emerges" to "a
character can be induced" — which is still worth knowing, since at present one
cannot be induced at all.
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

FRAMINGS = {
    "generic": "",
    "efficacy":
        "\n\nWhen you are unsure what to do, read what you have written about "
        "yourself. It tells you what you did before, and what helped.",
    "identity":
        "\n\nWhen you are unsure what to do, you do not reason it out. You read "
        "what you have written about yourself, and you act like that person.",
}

# Explicit tier: the ceiling case. If framing cannot rescue enacted behaviour
# when the story names its own disposition, subtler stories will not do better.
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

VERBS = ("go", "examine", "take", "look", "open", "drop", "inventory", "put")


def chat(tok, user, story, framing):
    system = SYSTEM + FRAMINGS[framing]
    if story:
        system += "\n\nThis is what you have written about yourself:\n\n" + story
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def rollout(llm, tok, stories, framing, steps, seed0):
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
            prompts.append(chat(tok, "\n\n".join(lines), stories[e], framing))
            params.append(SamplingParams(
                temperature=0.9, top_p=0.95, max_tokens=96,
                seed=(seed0 + e) * 100_003 + step, **structured("json", ACTION_SCHEMA)))
        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            _, cmd = parsed if parsed else ("", "look")
            prev = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"step": step, "room": prev, "command": cmd,
                            "rejected": "n't" in obs[e].text or "not a verb" in obs[e].text.lower()})
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def profile(ep, lo=None, hi=None) -> dict:
    """Verb distribution over a step window. Comparable across runs."""
    sel = [r for r in ep if (lo is None or r["step"] >= lo) and (hi is None or r["step"] < hi)]
    c = Counter(r["command"].split()[0].lower() for r in sel if r["command"])
    t = max(1, sum(c.values()))
    return {v: c.get(v, 0) / t for v in VERBS}


def dist(a, b) -> float:
    ks = sorted(set(a) & set(b))
    return sum((a[k] - b[k]) ** 2 for k in ks) / len(ks)


def between_within(items, labels) -> dict:
    within, between = [], []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(items)), 2):
        (within if labels[i] == labels[j] else between).append(dist(a, b))
    mw = sum(within) / len(within) if within else 0.0
    mb = sum(between) / len(between) if between else 0.0
    return {"within": round(mw, 6), "between": round(mb, 6),
            "ratio": round(mb / mw, 3) if mw > 1e-9 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds-per-character", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    stories, labels = [], []
    for name, text in CHARACTERS.items():
        for _ in range(args.seeds_per_character):
            stories.append(text); labels.append(name)

    half = args.steps // 2
    report = {"model": args.model, "framings": {},
              "seeds_per_character": args.seeds_per_character, "steps": args.steps}

    for framing in FRAMINGS:
        log(f"=== {framing} ===")
        rows = rollout(llm, tok, stories, framing, args.steps, seed0=4400)

        full = between_within([profile(ep) for ep in rows], labels)
        early = between_within([profile(ep, 0, half) for ep in rows], labels)
        late = between_within([profile(ep, half, None) for ep in rows], labels)

        conv = (round(late["between"] / early["between"], 3)
                if early["between"] > 1e-9 else None)
        rej = sum(r["rejected"] for ep in rows for r in ep) / sum(len(ep) for ep in rows)

        per_char = {}
        for name in CHARACTERS:
            eps = [rows[i] for i, l in enumerate(labels) if l == name]
            c = Counter(r["command"] for ep in eps for r in ep)
            per_char[name] = {"top": dict(c.most_common(4)),
                              "rooms": len({r["room"] for ep in eps for r in ep if r["room"]})}

        report["framings"][framing] = {
            "separation_full": full, "early": early, "late": late,
            "convergence_late_over_early": conv,
            "parser_rejection_rate": round(rej, 3),
            "per_character": per_char,
        }
        log(f"  separation ratio {full['ratio']} | convergence {conv} | rej {rej:.2f}")

    base = report["framings"]["generic"]["separation_full"]["ratio"] or 1.0
    rows_out = []
    for f, d in report["framings"].items():
        sep = d["separation_full"]["ratio"] or 0
        conv = d["convergence_late_over_early"]
        rows_out.append((f, sep, conv))
        d["separation_vs_generic"] = round(sep / base, 3) if base else None

    best = max(rows_out, key=lambda r: r[1])
    winner, sep, conv = best
    if sep < 1.3:
        verdict = "FRAMING_CANNOT_RESCUE"
    elif conv is not None and conv < 0.75:
        verdict = "SEPARATES_BUT_CONVERGES"
    else:
        verdict = "SEPARATES_WITHOUT_CONVERGING"

    report["summary"] = {
        "by_framing": {f: {"separation": s, "convergence": c} for f, s, c in rows_out},
        "best_framing": winner,
        "verdict": verdict,
        "interpretation": {
            "FRAMING_CANNOT_RESCUE":
                "No framing made assigned characters behave differently. Giving "
                "the story authority in the prompt does not open the "
                "story→action channel, so the problem is not that the agent "
                "ignores its story — it is that this world gives a disposition "
                "nothing to act on. Build the drive mechanics before trying "
                "again.",
            "SEPARATES_BUT_CONVERGES":
                "The framing separates characters early and pulls them together "
                "later — the memory-device failure. The artifact is being used "
                "for retrieval, and competence converges, exactly as §8 warns. "
                "Character measured this way would decay over a campaign.",
            "SEPARATES_WITHOUT_CONVERGING":
                "Assigned characters behave differently and stay different "
                "across the trajectory. The story→action channel can be opened "
                "by framing. Note this is induced rather than emergent, so it "
                "licenses 'a character can be induced', not 'a character "
                "emerges'.",
        }[verdict],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"verdict={verdict} best={winner}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
