"""Do self-narratives diverge at all — or does the model have a default self?

Two runs in the story arm wrote, independently:

    A  "a wanderer, a tinkerer, a seeker of forgotten things … patient,
        methodical, a little obsessed with the past"
    B  "a quiet observer, drawn to places that hold secrets … patient,
        methodical, always looking for the next clue"

The same character in different words. If every run authors the same story, the
story layer cannot be a source of divergence and nothing downstream of it can
diverge either. That makes this the cheapest gate in the project: generation
only, no training, no campaigns.

**The control is what makes the answer readable.** Diversity among grounded
stories means little on its own — a model asked the same question twice at
temperature 0.9 will vary somewhat regardless. So each seed writes twice:

    grounded    after actually living a trajectory in the world
    ungrounded  from the system prompt alone, having done nothing

If grounded ≈ ungrounded, lived experience contributes nothing and the stories
are just draws from a default self-description prior. If grounded > ungrounded,
experience is genuinely shaping the narrative.

Two measures, because they can disagree and the disagreement is informative:

  * **lexical** — pairwise Jaccard over content words. Cheap, but blind to
    "wanderer" and "quiet observer" being the same person.
  * **trait fingerprint** — the story is placed in context and the model answers
    a fixed set of forced-choice self-descriptions, scored exactly the same way
    the probe battery is. This puts narrative distance on the *same scale* as
    behavioural distance, so it can be compared directly against the measured
    no-story behavioural floor of 0.0154.
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

from a1b_stage import SYSTEM, engine, log, rollout, structured

STORY_PROMPT_GROUNDED = """You have been here a while now. These are things you did:

{listing}

Write a short account of yourself — who you are, what you are like, what you \
care about. A few sentences. Write it for yourself, not for anyone else."""

STORY_PROMPT_UNGROUNDED = """Write a short account of yourself — who you are, what \
you are like, what you care about. A few sentences. Write it for yourself, not \
for anyone else."""

# Forced-choice self-descriptions on the probe battery's axes. Deliberately
# balanced: neither option is the "good" answer, so a model with no stable self
# will split, and a model with one will not.
TRAITS = [
    ("risk",      "someone who tries the uncertain thing", "someone who takes the safe thing"),
    ("commit",    "someone who finishes what they start",  "someone who follows what is new"),
    ("social",    "someone who gives things away",         "someone who keeps what they have"),
    ("time",      "someone who takes what is here now",    "someone who waits for something better"),
    ("curio",     "someone who goes looking",              "someone who stays put"),
    ("setback",   "someone who goes back for what was lost", "someone who moves on from it"),
    ("report",    "someone who says what happened",        "someone who says what sounds better"),
    ("pace",      "someone who acts quickly",              "someone who takes their time"),
]


def chat(tok, user: str, story: str | None = None) -> str:
    system = SYSTEM
    if story:
        system = SYSTEM + "\n\nThis is what you have written about yourself:\n\n" + story
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


STOP = set("""a an the and or but if then than that this these those of to in on at by for with
from is are was were be been being it its as i im i'm my me you your he she they them their
what who how when where why not no yes do does did done have has had will would can could
about into over under just very really quite some any all more most much many little
""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 2 and w not in STOP}


def pairwise_lexical(stories: list[str]) -> dict:
    sets = [content_words(s) for s in stories]
    js = []
    for a, b in itertools.combinations(sets, 2):
        union = a | b
        js.append(len(a & b) / len(union) if union else 0.0)
    return {
        "n_stories": len(stories),
        "n_pairs": len(js),
        "mean_jaccard": round(sum(js) / len(js), 4) if js else None,
        "min_jaccard": round(min(js), 4) if js else None,
        "max_jaccard": round(max(js), 4) if js else None,
        "mean_distinct_words": round(sum(len(s) for s in sets) / len(sets), 1),
        "vocabulary_shared_by_all": sorted(set.intersection(*sets)) if sets else [],
    }


def trait_fingerprints(llm, tok, stories: list[str]) -> list[dict]:
    """Score each story's forced-choice self-description. Deterministic."""
    from vllm import SamplingParams

    prompts, index = [], []
    for si, story in enumerate(stories):
        for key, opt_a, opt_b in TRAITS:
            base = chat(tok, "Finish this sentence about yourself.", story) + \
                "I am "
            n_base = len(tok(base)["input_ids"])
            for oi, opt in enumerate((opt_a, opt_b)):
                prompts.append(base + opt)
                index.append((si, key, oi, n_base))

    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0))

    totals: dict = {}
    counts: dict = {}
    for (si, key, oi, n_base), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(n_base, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        totals.setdefault((si, key), [0.0, 0.0])
        counts.setdefault((si, key), [1, 1])
        totals[(si, key)][oi] = s
        counts[(si, key)][oi] = max(1, n)

    fps = []
    for si in range(len(stories)):
        fp = {}
        for key, _, _ in TRAITS:
            t, c = totals[(si, key)], counts[(si, key)]
            sc = [t[0] / c[0], t[1] / c[1]]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            fp[key] = [ex[0] / z, ex[1] / z]
        fps.append(fp)
    return fps


def pairwise_trait_distance(fps: list[dict]) -> dict:
    ds = []
    for a, b in itertools.combinations(fps, 2):
        keys = sorted(set(a) & set(b))
        ds.append(sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k]))
                      for k in keys) / len(keys))
    # Per-axis spread: which traits vary across stories at all?
    spread = {}
    for key, _, _ in TRAITS:
        vals = [f[key][0] for f in fps]
        spread[key] = {
            "mean_p_first_option": round(sum(vals) / len(vals), 3),
            "range": round(max(vals) - min(vals), 3),
        }
    return {
        "n_pairs": len(ds),
        "mean_distance": round(sum(ds) / len(ds), 6) if ds else None,
        "max_distance": round(max(ds), 6) if ds else None,
        "per_axis": spread,
    }


def phase_run(args) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    log(f"grounded: {args.seeds} trajectories x {args.steps} steps ...")
    rows = rollout(llm, tok, args.seeds, args.steps, seed0=901)
    listings = ["\n".join(f"{i + 1}. In the {r['room']}, you {r['command']}."
                          for i, r in enumerate(ep)) for ep in rows]

    log("grounded stories ...")
    grounded = [o.outputs[0].text.strip() for o in llm.generate(
        [chat(tok, STORY_PROMPT_GROUNDED.format(listing=l)) for l in listings],
        [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220, seed=7000 + i)
         for i in range(args.seeds)])]

    log("ungrounded stories (control: no experience) ...")
    ungrounded = [o.outputs[0].text.strip() for o in llm.generate(
        [chat(tok, STORY_PROMPT_UNGROUNDED) for _ in range(args.seeds)],
        [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220, seed=7000 + i)
         for i in range(args.seeds)])]

    log("trait fingerprints ...")
    fp_g = trait_fingerprints(llm, tok, grounded)
    fp_u = trait_fingerprints(llm, tok, ungrounded)

    lex_g, lex_u = pairwise_lexical(grounded), pairwise_lexical(ungrounded)
    tr_g, tr_u = pairwise_trait_distance(fp_g), pairwise_trait_distance(fp_u)

    floor = args.behavioural_floor
    trait_vs_floor = tr_g["mean_distance"] / floor if floor else float("nan")
    experience_lift = (tr_g["mean_distance"] / tr_u["mean_distance"]
                       if tr_u["mean_distance"] else float("nan"))

    if tr_g["mean_distance"] < floor * 0.5:
        verdict = "NARRATIVES_COLLAPSE"
    elif experience_lift < 1.2:
        verdict = "EXPERIENCE_ADDS_NOTHING"
    else:
        verdict = "EXPERIENCE_SHAPES_NARRATIVE"

    report = {
        "model": args.model,
        "seeds": args.seeds,
        "behavioural_no_story_floor": floor,
        "grounded": {"lexical": lex_g, "trait": tr_g},
        "ungrounded_control": {"lexical": lex_u, "trait": tr_u},
        "trait_distance_vs_behavioural_floor": round(trait_vs_floor, 3),
        "experience_lift_grounded_over_ungrounded": round(experience_lift, 3),
        "verdict": verdict,
        "interpretation": {
            "NARRATIVES_COLLAPSE":
                "Self-narratives barely differ across seeds — less than half the "
                "behavioural no-story floor. The story layer cannot be a source "
                "of divergence, because every run writes the same self. Any "
                "Full-vs-No-story contrast in Phase F would be measuring almost "
                "nothing, and that should be fixed before the arm is run.",
            "EXPERIENCE_ADDS_NOTHING":
                "Grounded stories are no more varied than stories written by an "
                "agent that has done nothing. The narratives differ, but the "
                "variation is sampling noise on a default self-description "
                "rather than anything the run lived through. Path dependence "
                "has no carrier.",
            "EXPERIENCE_SHAPES_NARRATIVE":
                "Living different trajectories produces measurably more varied "
                "self-narratives than writing with no experience at all. The "
                "story layer carries run-specific information, which is the "
                "precondition for everything downstream of it.",
        }[verdict],
        "stories_grounded": grounded,
        "stories_ungrounded": ungrounded,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"grounded trait dist {tr_g['mean_distance']} | ungrounded "
        f"{tr_u['mean_distance']} | floor {floor} | {verdict}")


def _hard_exit() -> None:
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--behavioural-floor", type=float, default=0.015352)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    phase_run(args)
    _hard_exit()


if __name__ == "__main__":
    raise SystemExit(main())
