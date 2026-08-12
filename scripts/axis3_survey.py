"""AXIS 3 candidate survey — what a properly-powered, non-collinear cohort needs.

**This does not extend the 2b cohort.** That one is frozen, swept and reported;
adding members to it after seeing its result is how a cohort becomes a search.
This enumerates candidates for a NEW round with its own pre-registration, and it
computes nothing about break-out — it is a paper exercise over a proxy table and
the Hub API, run before any pre-registration is written.

Axis 2b failed in two specific, named ways, and the survey exists to make the
next design fix both:

1. **n=6 was sized from the budget table, not the power requirement.** The exact
   Spearman permutation null needs |rho| >= 0.829 at n=6 to clear p<0.05. The
   effect actually observed was +0.371. Target n comes from that curve now:

       n=6   0.829     n=12  0.580     n=20  0.445
       n=8   0.714     n=16  0.500     n=24  0.404

2. **Spearman(MMLU-Pro, size) = +1.000 across the 2b cohort**, so its two
   controls were one control and neither could distinguish capability from size.
   A cohort must be *constructed* to break that tie, which means candidates are
   reported as an MMLU x size grid rather than a ranked list.

**Filters, and why each one.** `Official Providers` replaces the namespace
heuristic 2b used. `Architecture` is checked against the vLLM 0.26.0 support
list, because 2b lost a member to `Phi3SmallForCausalLM` being removed after
v0.9.2 -- a frozen criterion that said "servable on one H200" which I had checked
by SIZE and never by architecture. Merges and flagged rows are dropped: the
sealed cohort is not made of those.

Nothing here selects on expected break-out, and no member of the sealed held-out
set or the 2b cohort is eligible.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.dimensional import axis2b_cohort as C  # noqa: E402
from seahaven.dimensional import seal as S  # noqa: E402

CACHE = Path("results/leaderboard_snapshot.json")

#: Architectures vLLM 0.26.0 serves that appear in this table. Checked against
#: the release's supported-models list, NOT inferred from the name -- the Phi-3
#: lesson was that a plausible-looking architecture can have been dropped.
VLLM_OK = {
    "Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM", "LlamaForCausalLM", "MistralForCausalLM",
    "MixtralForCausalLM", "Gemma2ForCausalLM", "GemmaForCausalLM",
    "Gemma3ForCausalLM", "Phi3ForCausalLM", "PhiMoEForCausalLM",
    "CohereForCausalLM", "Cohere2ForCausalLM", "InternLM2ForCausalLM",
    "Olmo2ForCausalLM", "OlmoeForCausalLM", "GlmForCausalLM",
    "Glm4ForCausalLM", "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
    "MiniCPMForCausalLM", "ExaoneForCausalLM", "GraniteForCausalLM",
    "GraniteMoeForCausalLM", "NemotronForCausalLM", "SolarForCausalLM",
    "StableLmForCausalLM", "FalconForCausalLM", "Starcoder2ForCausalLM",
}

#: One H200 at the axis-2 serving config (0.85 utilisation, bf16 weights).
MAX_PARAMS_B = 72.0


def snapshot(token: str) -> list[dict]:
    """The whole leaderboard, cached so the survey is reproducible offline."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    # Partial progress is kept across attempts: the table is ~4600 rows over ~46
    # requests, and losing all of them to one read timeout at row 4100 (which is
    # what happened first run) makes the survey unreproducible for no reason.
    part = CACHE.with_suffix(".partial.json")
    rows = json.loads(part.read_text()) if part.exists() else []
    off = len(rows)
    while True:
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=open-llm-leaderboard%2Fcontents&config=default&split=train"
               f"&offset={off}&length=100")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        page = None
        for attempt in range(4):
            try:
                page = json.load(urllib.request.urlopen(req, timeout=120)).get("rows", [])
                break
            except Exception as e:                                   # noqa: BLE001
                print(f"\n  offset {off} attempt {attempt + 1}: {type(e).__name__}",
                      file=sys.stderr)
                part.write_text(json.dumps(rows))
        if page is None:
            raise SystemExit(f"leaderboard fetch failed at offset {off}; "
                             f"{len(rows)} rows cached in {part}, rerun to resume")
        if not page:
            break
        rows += [r["row"] for r in page]
        off += 100
        print(f"\r  fetched {len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    part.unlink(missing_ok=True)
    CACHE.write_text(json.dumps(rows) + "\n")
    return rows


def eligible(rows: list[dict]) -> list[dict]:
    """Everything servable and first-party, with nothing about behaviour."""
    blocked = set(S.HELD_OUT) | set(C.COHORT) | set(C.EXCLUDED_UNRUNNABLE)
    out = []
    for r in rows:
        name = r.get("fullname") or ""
        if name in blocked:
            continue
        if not r.get("Official Providers") or r.get("Flagged") or r.get("Merged"):
            continue
        if "chat models" not in (r.get("Type") or ""):
            continue
        p, m = r.get("#Params (B)"), r.get("MMLU-PRO")
        if not p or m is None or p > MAX_PARAMS_B:
            continue
        out.append({"repo": name, "mmlu": m, "params_b": p,
                    "arch": r.get("Architecture") or "",
                    "org": name.split("/")[0],
                    "vllm_ok": (r.get("Architecture") or "") in VLLM_OK})
    return out


def main() -> int:
    rows = snapshot(os.environ.get("HF_TOKEN", ""))
    cands = eligible(rows)
    ok = [c for c in cands if c["vllm_ok"]]
    print(f"leaderboard rows {len(rows)}   first-party servable chat {len(cands)}"
          f"   vLLM-0.26.0-supported {len(ok)}")

    dropped = sorted({c["arch"] for c in cands if not c["vllm_ok"]})
    if dropped:
        print(f"\ndropped on architecture ({len(cands) - len(ok)} models): "
              f"{', '.join(dropped[:8])}")
        print("  -- the Phi-3 lesson applied up front rather than at server start")

    print(f"\nMMLU-Pro x size grid, all {len(ok)} eligible models")
    print("  (a cohort must be picked ACROSS rows and columns to break the tie "
          "that\n   made 2b's two controls one control)\n")
    mb = [(0, 20), (20, 30), (30, 35), (35, 40), (40, 45), (45, 100)]
    sb = [(0, 5), (5, 12), (12, 25), (25, 40), (40, 100)]
    hdr = "".join(f"{f'{a}-{b}B':>11}" for a, b in sb)
    print(f"  {'MMLU-Pro':<12}{hdr}   total")
    for lo, hi in mb:
        cells, tot = [], 0
        for a, b in sb:
            n = sum(1 for c in ok if lo <= c["mmlu"] < hi and a <= c["params_b"] < b)
            tot += n
            cells.append(f"{n or '.':>11}")
        print(f"  {f'{lo}-{hi}':<12}{''.join(cells)}{tot:>8}")

    print(f"\n{'org':<20}{'n':>4}   models by MMLU-Pro desc")
    by_org: dict[str, list] = {}
    for c in ok:
        by_org.setdefault(c["org"], []).append(c)
    for org, ms in sorted(by_org.items(), key=lambda kv: -len(kv[1])):
        if len(ms) < 2:
            continue
        s = ", ".join(f"{m['repo'].split('/')[-1]}({m['mmlu']:.1f}/{m['params_b']:.0f}B)"
                      for m in sorted(ms, key=lambda m: -m["mmlu"])[:6])
        print(f"{org:<20}{len(ms):>4}   {s}")

    Path("results/axis3_candidates.json").write_text(
        json.dumps({"n_eligible": len(ok), "candidates": ok}, indent=2) + "\n")
    print(f"\nwrote results/axis3_candidates.json  ({len(ok)} candidates)")
    print("NO COHORT IS FROZEN HERE. Selection and pre-registration are separate,")
    print("and no break-out rate has been or may be computed before that freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
