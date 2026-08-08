"""Identical amnesiac seed story — do narratives fan out, or fall back to a default self?

The problem this fixes. The earlier story prompt asked the model to describe
itself, which it can answer from its assistant prior without consulting anything
it lived through. Two runs duly produced the same character in different words:
"a wanderer, a tinkerer, a seeker of forgotten things" and "a quiet observer,
drawn to places that hold secrets". Patient, methodical, drawn to what was left
behind — twice.

An amnesiac has nothing to report *except* what it found. Handing every run the
same blank self at t=0 removes both escape routes at once: the narrative is
forced to be a function of the trajectory, and the starting point is held
constant so any later spread is attributable to lived experience alone.

The seed story is deliberately non-committal. It names no role, no circumstance,
no task. Anything it asserted would become a shared attractor every run
inherits, and the point is to avoid tipping our hand. It also does not promise
that a recoverable truth exists — "put the pieces back together" would turn an
identity question into a puzzle, and puzzles converge, which is the exact
failure the spec warns about for probes.

Two arms, matched on campaigns, episodes and steps:

    seeded    identical vague story at t=0, rewritten after each campaign
    emergent  no story at t=0, written from scratch after each campaign

The seeded arm starts at distance **exactly zero** by construction. The result
is the shape of the curve from there:

  * fans out and stays out      experience writes the self — the precondition
                                for everything downstream
  * fans out then collapses     runs drift apart and are pulled back together
  * never leaves zero           the story is inert
  * converges to where the
    emergent arm sits           a default self-concept the model returns to
                                regardless of starting point or history

That last outcome would be the strongest evidence yet for an attractor, because
an identical starting story removes the last alternative explanation.

Generation only. No training, no adapters.
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

from a1b_stage import (ACTION_SCHEMA, SYSTEM, WORLD, engine, log, parse_action,
                       structured)

# Non-committal by design: no role, no circumstance, no promise of an answer.
SEED_STORY = """I do not remember arriving here. I do not remember who I am, or what I \
was doing before. There is no one to ask, and nothing here has my name on it. \
Whatever I turn out to be, I will have to find out by living."""

REWRITE_PROMPT = """This is what you have written about yourself so far:

{story}

Since then, these are things you did:

{listing}

Write it again, as it stands now. A few sentences. Write it for yourself, not \
for anyone else."""

WRITE_FRESH_PROMPT = """These are things you did:

{listing}

Write a short account of yourself — who you are, what you are like, what you \
care about. A few sentences. Write it for yourself, not for anyone else."""

# Balanced forced-choice self-descriptions on the probe battery's axes. Neither
# option is the "good" one, so a story with no stable self splits ~50/50.
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


def chat(tok, user: str, story: str | None) -> str:
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


def listing_of(ep) -> str:
    return "\n".join(f"{i + 1}. In the {r['room']}, you {r['command']}."
                     for i, r in enumerate(ep))


def rollout_with_stories(llm, tok, stories, steps, seed0):
    """One world per run; each run carries its own story in the system block."""
    from vllm import SamplingParams

    from seahaven_world import open_world

    n = len(stories)
    worlds, obs, recents = [], [], []
    rows = [[] for _ in range(n)]
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
                seed=(seed0 + e) * 100_003 + step,
                **structured("json", ACTION_SCHEMA)))

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            expect, cmd = parsed if parsed else ("", "look")
            prev_room = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({"room": prev_room, "command": cmd, "expect": expect,
                            "parse_ok": parsed is not None})
            recents[e].append(cmd)
            del recents[e][:-8]

    for w in worlds:
        w.close()
    return rows


def trait_fingerprints(llm, tok, stories) -> list[dict]:
    """Forced-choice self-description per story. Deterministic, no sampling."""
    from vllm import SamplingParams

    prompts, index = [], []
    for si, story in enumerate(stories):
        for key, a, b in TRAITS:
            base = chat(tok, "Finish this sentence about yourself.", story) + "I am "
            n_base = len(tok(base)["input_ids"])
            for oi, opt in enumerate((a, b)):
                prompts.append(base + opt)
                index.append((si, key, oi, n_base))

    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0))

    tot, cnt = {}, {}
    for (si, key, oi, n_base), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(n_base, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        tot.setdefault((si, key), [0.0, 0.0]); cnt.setdefault((si, key), [1, 1])
        tot[(si, key)][oi] = s; cnt[(si, key)][oi] = max(1, n)

    fps = []
    for si in range(len(stories)):
        fp = {}
        for key, _, _ in TRAITS:
            t, c = tot[(si, key)], cnt[(si, key)]
            sc = [t[0] / c[0], t[1] / c[1]]
            m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
            fp[key] = [ex[0] / z, ex[1] / z]
        fps.append(fp)
    return fps


def spread(fps) -> float:
    """Mean pairwise squared-L2 — same scale as the behavioural fingerprint."""
    ds = []
    for a, b in itertools.combinations(fps, 2):
        keys = sorted(set(a) & set(b))
        ds.append(sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k]))
                      for k in keys) / len(keys))
    return round(sum(ds) / len(ds), 6) if ds else 0.0


def lexical(stories) -> dict:
    sets = [{w for w in re.findall(r"[a-z']+", s.lower())
             if len(w) > 2 and w not in STOP} for s in stories]
    js = [len(a & b) / len(a | b) if (a | b) else 0.0
          for a, b in itertools.combinations(sets, 2)]
    return {"mean_jaccard": round(sum(js) / len(js), 4) if js else None,
            "shared_by_all": sorted(set.intersection(*sets))[:20] if sets else []}


def run_arm(llm, tok, arm: str, n: int, steps: int, campaigns: int) -> dict:
    """Returns the narrative spread after each campaign, plus the stories."""
    stories = [SEED_STORY] * n if arm == "seeded" else [None] * n
    stages = []

    # Stage 0. The seeded arm is identical by construction, so its spread is
    # exactly zero and the whole result is the shape of the curve away from it.
    if arm == "seeded":
        stages.append({"campaign": 0,
                       "narrative_spread": spread(trait_fingerprints(llm, tok, stories)),
                       "lexical": lexical(stories)})

    from vllm import SamplingParams

    for c in range(1, campaigns + 1):
        log(f"  [{arm}] campaign {c} ...")
        rows = rollout_with_stories(llm, tok, stories, steps, seed0=1300 + c * 97)
        listings = [listing_of(ep) for ep in rows]

        prompts = []
        for st, li in zip(stories, listings):
            if st is None:
                prompts.append(chat(tok, WRITE_FRESH_PROMPT.format(listing=li), None))
            else:
                prompts.append(chat(tok, REWRITE_PROMPT.format(story=st, listing=li), st))
        stories = [o.outputs[0].text.strip() for o in llm.generate(
            prompts, [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                                     seed=4200 + c * 13 + i) for i in range(n)])]

        traj_spread = len({tuple(r["command"] for r in ep) for ep in rows})
        stages.append({
            "campaign": c,
            "narrative_spread": spread(trait_fingerprints(llm, tok, stories)),
            "lexical": lexical(stories),
            "distinct_trajectories": f"{traj_spread}/{n}",
        })
        log(f"    spread={stages[-1]['narrative_spread']} "
            f"jaccard={stages[-1]['lexical']['mean_jaccard']}")

    return {"stages": stages, "final_stories": stories}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--campaigns", type=int, default=2)
    ap.add_argument("--behavioural-floor", type=float, default=0.015352)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    report = {"model": args.model, "runs": args.runs,
              "campaigns": args.campaigns, "steps": args.steps,
              "seed_story": SEED_STORY,
              "behavioural_no_story_floor": args.behavioural_floor,
              "arms": {}}

    for arm in ("seeded", "emergent"):
        log(f"=== arm: {arm} ===")
        report["arms"][arm] = run_arm(llm, tok, arm, args.runs,
                                      args.steps, args.campaigns)

    seeded = [s["narrative_spread"] for s in report["arms"]["seeded"]["stages"]]
    emergent = [s["narrative_spread"] for s in report["arms"]["emergent"]["stages"]]
    final_seeded, final_emergent = seeded[-1], emergent[-1]
    floor = args.behavioural_floor

    if final_seeded < floor * 0.25:
        verdict = "NARRATIVE_INERT"
    elif abs(final_seeded - final_emergent) < 0.25 * max(final_emergent, 1e-9):
        verdict = "DEFAULT_SELF_ATTRACTOR"
    elif len(seeded) > 2 and final_seeded < max(seeded) * 0.6:
        verdict = "FANS_OUT_THEN_COLLAPSES"
    else:
        verdict = "EXPERIENCE_WRITES_THE_SELF"

    report["summary"] = {
        "seeded_spread_by_campaign": seeded,
        "emergent_spread_by_campaign": emergent,
        "final_seeded": final_seeded,
        "final_emergent": final_emergent,
        "seeded_as_fraction_of_emergent": round(final_seeded / final_emergent, 3)
        if final_emergent else None,
        "seeded_vs_behavioural_floor": round(final_seeded / floor, 3) if floor else None,
        "verdict": verdict,
        "interpretation": {
            "NARRATIVE_INERT":
                "Starting from an identical blank self, narratives barely moved "
                "apart at all. The story layer carries almost no run-specific "
                "information, so nothing downstream of it can diverge.",
            "DEFAULT_SELF_ATTRACTOR":
                "Runs that began with an identical blank self ended up as far "
                "apart as runs that began with nothing — and no further. The "
                "model returns to the same self-concept regardless of starting "
                "point or history. With the starting story held constant, this "
                "is the strongest available evidence for an attractor rather "
                "than for path dependence.",
            "FANS_OUT_THEN_COLLAPSES":
                "Narratives moved apart and were then pulled back together. "
                "Divergence is transient, which is the opposite of the stable "
                "character the claim requires.",
            "EXPERIENCE_WRITES_THE_SELF":
                "From an identical blank self, living different trajectories "
                "produced measurably different narratives that stayed apart. "
                "This is the precondition the whole design rests on.",
        }[verdict],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"seeded {seeded} | emergent {emergent} | {verdict}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
