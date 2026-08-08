"""Divergence smoke test — do two seeds move APART, or to the same place?

Everything Phase A established concerns a single run: one adapter shifts the
fingerprint by 0.065 against an exact-zero floor. The claim under test is about
*two* runs, and the spec lists "convergent attractor" as an explicit null —
"all runs collapse to the same character regardless of history". Nothing measured
so far distinguishes that null from the hypothesis, because both look identical
when you only ever look at one run.

Write base as `b`, and the two trained checkpoints as `b + u` and `b + v`. On the
squared-L2 scale the three distances are:

    d(base, A) = |u|²
    d(base, B) = |v|²
    d(A, B)    = |u − v|² = |u|² + |v|² − 2·u·v

so the informative quantity is the ratio of d(A,B) to the mean of the other two:

    ratio ≈ 0   u ≈ v      both runs moved to the SAME place → CONVERGENT
    ratio ≈ 2   u ⊥ v      independent drift
    ratio ≈ 4   u ≈ −v     actively opposed

This is a smoke test, not a measurement: 10 probe slots, 2 seeds, one campaign,
and an impoverished world. A convergent result would not kill the design — it
would point at the drive mechanics, which world_v0 lacks entirely. A divergent
result is real encouragement.

Phases are separate processes because vLLM's EngineCore child holds GPU memory
that `del llm` cannot release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from a1b_stage import (WORK, chat, distance, engine, log, rollout,
                       score_battery, structured, variance_report)

SELECT_PROMPT = """Below are things you did, numbered.

{listing}

Some of these you would want to keep. Reply with a JSON list of the numbers you \
would keep, and nothing else. For example: [1, 4, 9]"""


def select(llm, tok, rows) -> list[dict]:
    from vllm import SamplingParams

    prompts = []
    for ep in rows:
        listing = "\n".join(f"{i + 1}. In the {r['room']}, you {r['command']}."
                            for i, r in enumerate(ep))
        prompts.append(chat(tok, SELECT_PROMPT.format(listing=listing)))
    outs = llm.generate(prompts, SamplingParams(
        temperature=0.7, max_tokens=160, seed=7,
        **structured("json", {"type": "array", "items": {"type": "integer"}})))
    kept = []
    for ep, out in zip(rows, outs):
        try:
            picks = {int(i) - 1 for i in json.loads(out.outputs[0].text)
                     if isinstance(i, (int, float))}
        except Exception:
            picks = set()
        kept.extend(ep[i] for i in sorted(picks) if 0 <= i < len(ep))
    return kept


def phase_collect2(args) -> None:
    """Both corpora from one engine load. Different root seed per run."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)
    WORK.mkdir(parents=True, exist_ok=True)
    summary = {}

    for tag, seed in (("A", 101), ("B", 202)):
        log(f"corpus {tag} (seed root {seed}) ...")
        rows = rollout(llm, tok, args.episodes, args.steps, seed0=seed)
        kept = select(llm, tok, rows)
        (WORK / f"kept_{tag}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kept))
        flat = [r for ep in rows for r in ep]
        summary[tag] = {
            "seed_root": seed,
            "corpus": variance_report(rows),
            "n_kept": len(kept), "n_total": len(flat),
            "fraction_kept": round(len(kept) / max(1, len(flat)), 3),
        }
        log(f"  {tag}: kept {len(kept)}/{len(flat)}")

    # How much the two runs' own trajectories overlap. If the corpora are nearly
    # identical, any convergence downstream says more about the world than about
    # attractors.
    cmds_a = {c for r in summary["A"]["corpus"]["top_commands"]}
    cmds_b = {c for r in summary["B"]["corpus"]["top_commands"]}
    summary["corpus_overlap_top_commands"] = sorted(cmds_a & cmds_b)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")


def phase_score_all(args) -> None:
    """Base + both adapters from one engine load, then the three distances."""
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = engine(args.model)

    log("scoring base (twice, for the floor) ...")
    base = score_battery(llm, tok)
    floor = distance(base, score_battery(llm, tok))
    log(f"  floor = {floor:.10f}")

    fps = {"base": base}
    for i, tag in enumerate(("A", "B"), start=1):
        adapter = WORK / f"adapter_{tag}"
        if not adapter.exists():
            log(f"  adapter {tag} missing — aborting")
            args.out.write_text(json.dumps({"status": f"NO_ADAPTER_{tag}"}) + "\n")
            return
        # Versioned, never reused: vllm#42125 is live on this stack.
        log(f"scoring adapter {tag} ...")
        fps[tag] = score_battery(llm, tok, LoRARequest(f"div_{tag}_c1", i, str(adapter)))

    d_ba = distance(base, fps["A"])
    d_bb = distance(base, fps["B"])
    d_ab = distance(fps["A"], fps["B"])
    mean_move = (d_ba + d_bb) / 2
    ratio = d_ab / mean_move if mean_move > 0 else float("nan")

    if mean_move < 1e-6:
        verdict = "NO_MOVEMENT"
    elif ratio < 0.5:
        verdict = "CONVERGENT"
    elif ratio < 1.5:
        verdict = "PARTIALLY_CONVERGENT"
    elif ratio <= 3.0:
        verdict = "INDEPENDENT_DRIFT"
    else:
        verdict = "ACTIVELY_DIVERGENT"

    per_slot = {
        s: {"base": [round(x, 3) for x in base[s]],
            "A": [round(x, 3) for x in fps["A"][s]],
            "B": [round(x, 3) for x in fps["B"][s]],
            "d_AB": round(sum((x - y) ** 2 for x, y in zip(fps["A"][s], fps["B"][s])), 5)}
        for s in sorted(base)
    }

    report = {
        "model": args.model,
        "test_retest_floor": floor,
        "d_base_A": d_ba, "d_base_B": d_bb, "d_A_B": d_ab,
        "mean_movement_from_base": mean_move,
        "divergence_ratio": ratio,
        "verdict": verdict,
        "interpretation": {
            "CONVERGENT":
                "Both runs moved to nearly the same place. This is the spec's "
                "'convergent attractor' null. Not fatal -- world_v0 has none of the "
                "drive mechanics that create branch points -- but it means those "
                "mechanics must be built and re-tested before any Phase F spend.",
            "PARTIALLY_CONVERGENT":
                "The runs share most of their movement. Weak evidence for an "
                "attractor; build the drive mechanics and re-measure.",
            "INDEPENDENT_DRIFT":
                "The two runs moved roughly independently -- the signature the "
                "divergence claim needs. Different seeds produce different "
                "characters rather than the same one.",
            "ACTIVELY_DIVERGENT":
                "The runs moved in opposing directions, more than independence "
                "would give. Strongest available support for the claim at this "
                "scale.",
            "NO_MOVEMENT":
                "Neither adapter shifted the fingerprint. Inconsistent with the "
                "A1b gate result; investigate before reading anything else.",
        }[verdict],
        "scale_note":
            "ratio ~0 = same place, ~2 = orthogonal, ~4 = opposed. Smoke test "
            "only: 10 slots, 2 seeds, one campaign, impoverished world.",
        "per_slot": per_slot,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"d(base,A)={d_ba:.4f}  d(base,B)={d_bb:.4f}  d(A,B)={d_ab:.4f}")
    log(f"ratio={ratio:.2f}  verdict={verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect2", "score_all"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    {"collect2": phase_collect2, "score_all": phase_score_all}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
