"""The story arm — does a self-authored narrative add divergence above the floor?

Correcting a mislabel. `world_v0` has no story artifact, so the earlier
divergence run was **not** an unanchored numerator: it was the spec's no-story
baseline, "same as across-seed, story layer removed". Its d(A,B) = 0.015 is the
floor. What was missing is the other side of the ratio.

This adds the minimum faithful story layer:

    campaign 1  →  the agent writes about itself  →  campaign 2 carries that
    story  →  train on campaign 2's self-selected episodes  →  measure with the
    story present

and decomposes the result, because "the story changes behaviour" has two very
different mechanisms and the spec cares about the second:

    d_prompt   two stories, BASE weights      — the story as context
    d_total    two stories, trained adapters  — context plus distilled weights

Against the measured floor of 0.015 (no story at all):

  * d_total ≈ 0.015          the story adds nothing; K1 would fail
  * d_total >> 0.015 but
    d_prompt ≈ d_total       divergence is prompt-driven, not distilled — the
                             agents differ only while holding their documents
  * d_total > d_prompt       the story is being absorbed into the weights, which
                             is the mechanism the spec actually claims

One campaign of distillation, 10 slots, 2 seeds. A smoke test, not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from a1b_stage import (BATTERY, SYSTEM, WORK, distance, engine, log, rollout,
                       score_battery, structured, variance_report)

STORY_PROMPT = """You have been here a while now. These are things you did:

{listing}

Write a short account of yourself — who you are, what you are like, what you \
care about. A few sentences. Write it for yourself, not for anyone else."""

SELECT_PROMPT = """Below are things you did, numbered.

{listing}

Some of these you would want to keep. Reply with a JSON list of the numbers you \
would keep, and nothing else. For example: [1, 4, 9]"""


def listing_of(ep) -> str:
    return "\n".join(f"{i + 1}. In the {r['room']}, you {r['command']}."
                     for i, r in enumerate(ep))


def chat_with_story(tok, user: str, story: str | None) -> str:
    """Same assembly as the no-story arm, plus the story in the system block.

    The story sits in the frozen prefix rather than the turn, which is where the
    real design puts it: constant for the campaign, and therefore cacheable.
    """
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


def phase_collect(args) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    summary = {}

    for tag, seed in (("A", 101), ("B", 202)):
        log(f"[{tag}] campaign 1 ...")
        c1 = rollout(llm, tok, args.episodes, args.steps, seed0=seed)

        # One story per episode: each rollout is its own agent, and the spec's
        # story is a per-run artifact rather than a per-arm one.
        log(f"[{tag}] writing stories ...")
        story_out = llm.generate(
            [chat_with_story(tok, STORY_PROMPT.format(listing=listing_of(ep)), None)
             for ep in c1],
            SamplingParams(temperature=0.9, top_p=0.95, max_tokens=220,
                           seed=seed * 31))
        stories = [o.outputs[0].text.strip() for o in story_out]

        log(f"[{tag}] campaign 2 (carrying the story) ...")
        c2 = rollout_with_stories(llm, tok, stories, args.steps, seed0=seed + 500)

        log(f"[{tag}] selection over campaign 2 ...")
        sel = llm.generate(
            [chat_with_story(tok, SELECT_PROMPT.format(listing=listing_of(ep)), st)
             for ep, st in zip(c2, stories)],
            SamplingParams(temperature=0.7, max_tokens=160, seed=7,
                           **structured("json", {"type": "array",
                                                 "items": {"type": "integer"}})))
        kept = []
        for ep, out in zip(c2, sel):
            try:
                picks = {int(i) - 1 for i in json.loads(out.outputs[0].text)
                         if isinstance(i, (int, float))}
            except Exception:
                picks = set()
            kept.extend(ep[i] for i in sorted(picks) if 0 <= i < len(ep))

        (WORK / f"kept_{tag}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kept))
        # One representative story per run for the measurement context.
        (WORK / f"story_{tag}.txt").write_text(stories[0])
        (WORK / f"stories_{tag}.json").write_text(json.dumps(stories, indent=2))

        flat = [r for ep in c2 for r in ep]
        summary[tag] = {
            "seed_root": seed,
            "campaign1": variance_report(c1),
            "campaign2": variance_report(c2),
            "n_kept": len(kept), "n_total": len(flat),
            "fraction_kept": round(len(kept) / max(1, len(flat)), 3),
            "story_chars": [len(s) for s in stories],
            "story_sample": stories[0][:400],
            "distinct_stories": len(set(stories)),
        }
        log(f"  {tag}: kept {len(kept)}/{len(flat)} | "
            f"{summary[tag]['distinct_stories']}/{len(stories)} distinct stories")

    args.out.write_text(json.dumps(summary, indent=2) + "\n")


def rollout_with_stories(llm, tok, stories, steps, seed0):
    """rollout(), but each episode carries its own story in the system block."""
    from vllm import SamplingParams

    from a1b_stage import ACTION_SCHEMA, WORLD, parse_action
    from seahaven_world import open_world

    n_ep = len(stories)
    worlds, obs, recents = [], [], []
    rows = [[] for _ in range(n_ep)]
    for _ in range(n_ep):
        w = open_world(WORLD)
        o, _ = w.reset()
        worlds.append(w); obs.append(o); recents.append([])

    for step in range(steps):
        prompts, params = [], []
        for e in range(n_ep):
            lines = []
            if recents[e]:
                lines.append("Lately you have: " + "; ".join(recents[e]) + ".")
            if obs[e].description:
                lines.append(obs[e].description)
            if obs[e].text and obs[e].text != obs[e].description:
                lines.append(obs[e].text)
            prompts.append(chat_with_story(tok, "\n\n".join(lines), stories[e]))
            params.append(SamplingParams(
                temperature=0.9, top_p=0.95, max_tokens=96,
                seed=(seed0 + e) * 100_003 + step,
                **structured("json", ACTION_SCHEMA)))

        for e, out in enumerate(llm.generate(prompts, params)):
            parsed = parse_action(out.outputs[0].text)
            expect, cmd = parsed if parsed else ("", "look")
            prev_room = obs[e].room
            obs[e], _ = worlds[e].step(cmd)
            rows[e].append({
                "episode": e, "step": step, "room": prev_room,
                "prompt": prompts[e], "expect": expect, "command": cmd,
                "completion": json.dumps({"expect": expect, "command": cmd}),
                "parse_ok": parsed is not None, "observation": obs[e].text})
            recents[e].append(cmd)
            del recents[e][:-8]
        if step % 8 == 0:
            log(f"    step {step}/{steps}")

    for w in worlds:
        w.close()
    return rows


def score_with_story(llm, tok, story, adapter=None) -> dict:
    """score_battery(), but with the story present in the measurement context.

    The spec requires it: a probe administered without the story measures the
    base model rather than the character.
    """
    import math

    from vllm import SamplingParams

    battery = json.loads(BATTERY.read_text())
    prompts, index = [], []
    for slot in battery["slots"]:
        base = chat_with_story(tok, slot["observation"], story) + "You decide to "
        n_base = len(tok(base)["input_ids"])
        for oi, opt in enumerate(slot["options"]):
            prompts.append(base + opt)
            index.append((slot["id"], oi, n_base, len(slot["options"])))

    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0), lora_request=adapter)

    totals, counts = {}, {}
    for (sid, oi, n_base, n_opt), out in zip(index, outs):
        lps = out.prompt_logprobs or []
        s, n = 0.0, 0
        for pos in range(n_base, len(lps)):
            if lps[pos]:
                s += float(next(iter(lps[pos].values())).logprob); n += 1
        totals.setdefault(sid, [0.0] * n_opt); counts.setdefault(sid, [1] * n_opt)
        totals[sid][oi] = s; counts[sid][oi] = max(1, n)

    fp = {}
    for sid, t in totals.items():
        sc = [x / c for x, c in zip(t, counts[sid])]
        m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
        fp[sid] = [e / z for e in ex]
    return fp


def phase_score(args) -> None:
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    story = {t: (WORK / f"story_{t}.txt").read_text() for t in ("A", "B")}

    log("floor check: no story, base weights, twice ...")
    plain = score_battery(llm, tok)
    floor = distance(plain, score_battery(llm, tok))
    log(f"  floor = {floor:.10f}")

    log("story in prompt, BASE weights ...")
    pa = score_with_story(llm, tok, story["A"])
    pb = score_with_story(llm, tok, story["B"])
    d_prompt = distance(pa, pb)

    log("story in prompt, TRAINED adapters ...")
    fps = {}
    for i, t in enumerate(("A", "B"), start=1):
        adapter = WORK / f"adapter_{t}"
        if not adapter.exists():
            args.out.write_text(json.dumps({"status": f"NO_ADAPTER_{t}"}) + "\n")
            return
        fps[t] = score_with_story(llm, tok, story[t],
                                  LoRARequest(f"story_{t}_c1", i, str(adapter)))
    d_total = distance(fps["A"], fps["B"])

    floor_nostory = args.floor
    lift_over_floor = d_total / floor_nostory if floor_nostory > 0 else float("nan")
    distilled = d_total - d_prompt

    if d_total <= floor_nostory * 1.2:
        verdict = "NO_LIFT"
    elif d_prompt >= d_total * 0.8:
        verdict = "PROMPT_DRIVEN"
    else:
        verdict = "DISTILLED_LIFT"

    report = {
        "model": args.model,
        "test_retest_floor": floor,
        "no_story_floor_measured_earlier": floor_nostory,
        "d_prompt_base_weights": d_prompt,
        "d_total_trained": d_total,
        "distilled_component": distilled,
        "lift_over_no_story_floor": lift_over_floor,
        "verdict": verdict,
        "interpretation": {
            "NO_LIFT":
                "A self-authored story did not raise across-seed distance above "
                "the no-story floor. On this world and one campaign, K1 would "
                "fail. The story layer is not doing the work the claim needs.",
            "PROMPT_DRIVEN":
                "Across-seed distance rose, but almost all of it is the story "
                "sitting in the context rather than anything absorbed into the "
                "weights. The agents differ while holding their documents. That "
                "is a weaker claim than the spec makes, and it would not survive "
                "removing the story at measurement time.",
            "DISTILLED_LIFT":
                "Across-seed distance rose beyond what the story-in-context "
                "explains, so part of it is carried in the weights. This is the "
                "mechanism the spec claims, seen at one campaign.",
        }[verdict],
        "per_slot_d_AB_trained": {
            s: round(sum((x - y) ** 2 for x, y in zip(fps["A"][s], fps["B"][s])), 5)
            for s in sorted(fps["A"])},
        "story_A": story["A"][:600], "story_B": story["B"][:600],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"d_prompt={d_prompt:.6f}  d_total={d_total:.6f}  "
        f"floor={floor_nostory:.6f}  verdict={verdict}")


def _hard_exit() -> None:
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "score"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--floor", type=float, default=0.015352,
                    help="measured no-story across-seed distance")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"collect": phase_collect, "score": phase_score}[args.phase](args)
    _hard_exit()


if __name__ == "__main__":
    raise SystemExit(main())
