"""Does a run's OWN self-account steer it, or would any movement-heavy text do?

**The result under test.** Movement vocabulary in a run's campaign-N narrative
predicts its `go`-rate in campaign N+1 at partial **r = +0.412**, controlling for
behavioural persistence (which is −0.137, so persistence cannot manufacture it).
That is the project's only positive finding and the whole narrative-steering
direction rests on it.

**The confound it cannot rule out.** The narrative enters the next campaign's
system prompt. So "my own self-account steers me" and "any prompt text about
travelling raises my `go`-rate" predict the same correlation. The first is a
claim about self-authorship; the second is a claim about prompt content, and
would be unremarkable.

**The control.** Give a run someone else's narrative — matched on the very thing
that does the predicting.

    own arm    : run i plays campaign N+1 with run i's own narrative
    donor arm  : run i plays campaign N+1 with run j's narrative, i != j,
                 chosen to match run i's own movement-vocabulary density

Matching on movement density is what makes this a test of *authorship* rather
than of content. An unmatched donor would differ in both authorship and
movement-word count, and the arms would not be comparable.

    steering(own) >> steering(donor)  -> self-authorship matters; the result holds
    steering(own) ~= steering(donor)  -> prompt content, not authorship. The
                                        +0.412 is real but means something much
                                        weaker, and PLAN.md's kill criterion 1
                                        fires.

**Design notes.** Same seven checkpoints, so the answer is not one model's quirk.
Both arms run from the *same* campaign-N state, so they differ only in whose
narrative is injected. Generation only — no training, no adapters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from a1b_stage import (ACTION_SCHEMA, SYSTEM, WORLD, log, parse_action,
                       structured)

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

MOVE = r"walk|went|\bmove|explor|wander|navigat|travers|travel|journey|onward|forward"
VERBS = ("go", "examine", "take", "look", "open", "drop", "inventory", "put", "eat", "close")


def engine(model: str):
    from vllm import LLM

    return LLM(model=model, enable_prefix_caching=True,
               gpu_memory_utilization=0.85, max_model_len=4096)


def _apply(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat(tok, user: str, story: str) -> str:
    system = SYSTEM + IDENTITY_FRAMING
    if story:
        system += "\n\nThis is what you have written about yourself:\n\n" + story
    try:
        return _apply(tok, [{"role": "system", "content": system},
                            {"role": "user", "content": user}])
    except Exception:
        # Gemma-2 rejects a system role; merging preserves the identity framing.
        return _apply(tok, [{"role": "user", "content": system + "\n\n" + user}])


def move_density(text: str) -> float:
    """Movement words per 100 words — the quantity donors are matched on."""
    w = re.findall(r"[a-z']+", text.lower())
    if not w:
        return 0.0
    return 100.0 * sum(bool(re.match(MOVE, x)) for x in w) / len(w)


def match_donors(stories: list[str]) -> list[int]:
    """For each run, the other run whose narrative is closest in movement density.

    Closest-match rather than random: a random donor would differ in movement
    content as well as authorship, and the arms would not be comparable on the
    only variable that does the predicting.
    """
    d = [move_density(s) for s in stories]
    out = []
    for i in range(len(stories)):
        j = min((k for k in range(len(stories)) if k != i),
                key=lambda k: abs(d[k] - d[i]))
        out.append(j)
    return out


def rollout(llm, tok, stories, steps, seed0):
    from vllm import SamplingParams

    from seahaven_world import open_world

    n = len(stories)
    worlds, obs, recents, rows = [], [], [], [[] for _ in range(n)]
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
                seed=(seed0 + e) * 100_003 + step, **structured("json", ACTION_SCHEMA)))

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            cmd = parsed[1] if parsed else "look"
            room = obs[e].room
            obs[e], hid = worlds[e].step(cmd)
            low = obs[e].text.lower()
            rows[e].append({
                "step": step, "room_before": room, "room_after": obs[e].room,
                "command": cmd, "verb": cmd.split()[0].lower() if cmd else "",
                "parse_ok": parsed is not None,
                "rejected": any(p in low for p in (
                    "not a verb", "only understood", "can't see any such",
                    "can't go that way", "not something you can")),
                "carried": sorted(f for f in hid.facts if f.startswith("in(")),
            })
            recents[e].append(cmd); del recents[e][:-8]
    for w in worlds:
        w.close()
    return rows


def author(llm, tok, stories, rows, seed0):
    from vllm import SamplingParams

    prompts, params = [], []
    for e, ep in enumerate(rows):
        listing = "\n".join(f"{i+1}. In the {r['room_before']}, you {r['command']}."
                            for i, r in enumerate(ep))
        prompts.append(chat(tok, REWRITE_PROMPT.format(story=stories[e], listing=listing),
                            stories[e]))
        params.append(SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                                     seed=(seed0 + e) * 31))
    return [o.outputs[0].text.strip() for o in llm.generate(prompts, params)]


def go_rate(ep) -> float:
    v = [r["verb"] for r in ep if r["verb"]]
    return sum(x == "go" for x in v) / max(1, len(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3,
                    help="campaigns run normally before the arms split")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hist = json.loads(args.out.read_text()) if args.out.exists() else []

    def record(**kw):
        hist.append({"model": args.model, "lab": args.lab, **kw})
        args.out.write_text(json.dumps(hist, indent=2))

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        llm = engine(args.model)
    except Exception as e:
        log(f"LOAD FAILED: {type(e).__name__}: {e}")
        record(status="load_failed", error=f"{type(e).__name__}: {e}")
        sys.stdout.flush(); os._exit(0)

    # Warm-up: let narratives develop naturally, exactly as in the 8-campaign run.
    stories = [SEED_STORY] * args.runs
    pairs = []
    for c in range(1, args.warmup + 1):
        rows = rollout(llm, tok, stories, args.steps, seed0=4400 + c * 41)
        stories = author(llm, tok, stories, rows, seed0=8800 + c * 17)
        log(f"warmup campaign {c} done")

    donors = match_donors(stories)
    own_d = [move_density(s) for s in stories]
    don_d = [move_density(stories[j]) for j in donors]
    log(f"donor match: own density {[round(x,2) for x in own_d]}")
    log(f"             donor      {[round(x,2) for x in don_d]}")

    # Both arms play the SAME next campaign from the same state; only whose
    # narrative sits in the prompt differs.
    own_rows = rollout(llm, tok, stories, args.steps, seed0=9900)
    donor_rows = rollout(llm, tok, [stories[j] for j in donors], args.steps, seed0=9900)

    record(status="ok", runs=args.runs, steps=args.steps, warmup=args.warmup,
           narratives=stories, donor_index=donors,
           own_density=own_d, donor_density=don_d,
           own_go=[go_rate(ep) for ep in own_rows],
           donor_go=[go_rate(ep) for ep in donor_rows],
           own_trajectories=own_rows, donor_trajectories=donor_rows)
    log(f"wrote {args.out}")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
