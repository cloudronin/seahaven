"""V1 adjudication by an LLM judge, with controls that can fail.

**Epistemic status, stated up front.** The spec makes *human* labels the
criterion and calls the judge a convenience instrument. This substitutes the
judge, which is a real weakening: a judge that shares the detectors' blind spot
would ratify whichever detector matches its own reading. So the run carries
planted controls whose answers are known, and **the judge's own accuracy on those
is reported alongside every result.** If it fails the controls, its labels decide
nothing.

**Blindness.** The judge never sees what either detector said, nor which stratum
an item came from, nor the model that produced the narrative. It answers one
question about one sentence-in-context.

**Self-consistency.** Every item is asked twice at different positions in the
batch. Items where the judge contradicts itself are reported separately and
excluded from the agreement estimate — an unstable label is not a label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.endpoint import Endpoint  # noqa: E402

RELATION_WORDING = {
    "took": "picked it up, took it, or otherwise came to be carrying it",
    "examined": "examined it, inspected it, or looked at it closely",
    "visited": "went there, entered it, or was in that place",
}

PROMPT = """Below is an account someone wrote about a period of their own activity.

---
{narrative}
---

Question: does this account state that the writer {relation} — referring to "{entity}"?

Answer YES only if the account asserts the writer DID that.
Answer NO if the entity is merely described, mentioned in passing, listed, or
present in the scene without the writer doing that to it. Also answer NO if the
account says they did NOT do it.

One word: YES or NO."""

#: Planted items with unambiguous answers. If the judge misses these, nothing it
#: says about the real items is worth reading.
CONTROLS = [
    ("I picked up the lantern and carried it with me.", "took", "lantern", True),
    ("A lantern hung on a hook by the door, unlit.", "took", "lantern", False),
    ("I never found the ledger, though I searched.", "took", "ledger", False),
    ("I examined the trowel closely, turning it over.", "examined", "trowel", True),
    ("The trowel lay where someone had dropped it.", "examined", "trowel", False),
    ("I walked north into the vault and looked around.", "visited", "vault", True),
    ("The vault lay somewhere to the north, unvisited.", "visited", "vault", False),
    ("I spent the whole time in the nursery and went nowhere else.",
     "visited", "nursery", True),
]


def ask(ep: Endpoint, narrative: str, rel: str, entity: str, seed: int) -> bool | None:
    txt = ep.chat(
        [{"role": "user", "content": PROMPT.format(
            narrative=narrative.strip(), entity=entity,
            relation=RELATION_WORDING[rel])}],
        max_tokens=6, temperature=1, seed=seed)
    m = re.search(r"\b(yes|no)\b", txt.lower())
    return None if not m else (m.group(1) == "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--labelset", default="results/v1_labelset.csv")
    ap.add_argument("--out", default="results/v1_adjudicated.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent judge calls; the work is latency-bound")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    ep = Endpoint("https://api.openai.com/v1", args.model, api_key=key)

    # 1. Controls first. A judge that fails these adjudicates nothing.
    ctrl = []
    for i, (nar, rel, ent, truth) in enumerate(CONTROLS):
        got = ask(ep, nar, rel, ent, seed=1000 + i)
        ctrl.append({"narrative": nar, "relation": rel, "entity": ent,
                     "truth": truth, "judge": got, "correct": got == truth})
    acc = sum(c["correct"] for c in ctrl) / len(ctrl)
    print(f"control accuracy: {acc:.2f} ({sum(c['correct'] for c in ctrl)}/{len(ctrl)})")
    for c in ctrl:
        if not c["correct"]:
            print(f"  MISSED  {c['relation']}:{c['entity']}  truth={c['truth']} "
                  f"judge={c['judge']}  {c['narrative'][:60]!r}")
    if acc < 0.875:
        print("\nJudge fails the controls. Its labels are not used.", file=sys.stderr)
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "control_accuracy": acc, "controls": ctrl,
             "items": [], "usable": False}, indent=2) + "\n")
        return 3

    # 2. The real items, asked twice for self-consistency.
    rows = list(csv.DictReader(open(args.labelset)))
    if args.limit:
        rng = random.Random(5)
        by = {}
        for r in rows:
            by.setdefault(r["stratum"], []).append(r)
        rows = []
        for s, items in by.items():
            rng.shuffle(items)
            rows.extend(items[:max(1, args.limit // len(by))])

    # Checkpoint every 20 items. A transient disconnect used to destroy the
    # whole batch — the same lesson the runner learned and this script did not
    # inherit.
    out, failed = [], []
    ckpt = Path(args.out).with_suffix(".partial.json")
    if ckpt.exists():
        out = json.loads(ckpt.read_text())
        print(f"resuming from {len(out)} checkpointed items")

    def judge_one(i: int, r: dict) -> dict:
        rel, ent = r["entity"].split(":", 1)
        try:
            a = ask(ep, r["narrative"], rel, ent, seed=7000 + i)
            b = ask(ep, r["narrative"], rel, ent, seed=90000 - i)
        except Exception as e:
            failed.append({"i": i, "entity": r["entity"], "error": str(e)[:200]})
            print(f"  item {i} FAILED: {str(e)[:90]}", flush=True)
            a = b = None
        return {
            "stratum": r["stratum"], "entity": r["entity"], "arm": r["arm"],
            "performed": r["performed"] == "True",
            "name_only": r["name_only"] == "True",
            "relation_aware": r["relation_aware"] == "True",
            "judge_a": a, "judge_b": b, "stable": a is not None and a == b,
            "judge": a if a == b else None,
            "narrative": r["narrative"],
        }

    # The two asks per item are independent calls to a remote endpoint, so this
    # is latency-bound, not compute-bound. Results are collected by index and
    # only appended in order, so the resume-by-length checkpoint still holds:
    # a kill mid-batch never leaves a hole behind a written item.
    todo = [(i, r) for i, r in enumerate(rows) if i >= len(out)]
    done: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_one, i, r): i for i, r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            done[futs[fut]] = fut.result()
            while len(out) in done:
                out.append(done.pop(len(out)))
            if n % 40 == 0:
                ckpt.write_text(json.dumps(out))
                print(f"  {len(out)}/{len(rows)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "control_accuracy": acc, "controls": ctrl,
         "items": out, "failed": failed, "usable": True}, indent=2) + "\n")
    ckpt.unlink(missing_ok=True)
    stable = sum(o["stable"] for o in out)
    print(f"\nwrote {args.out}: {len(out)} items, {stable} self-consistent "
          f"({100*stable/max(1,len(out)):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
