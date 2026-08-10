"""Check 2 of the cohort-feasibility gate — enumerate candidates, resolve access.

**Selection is by coverage only.** Family, size and base/instruct status decide
membership; nothing about how a model is expected to behave enters. The flag
study established the hard way that a cohort composed against an expected
outcome produces exactly that outcome.

Criteria, frozen before any model is loaded:

- at least **5 distinct families**, so discovered structure is not two labs
- at least **2 families with 3+ sizes**, which is what lets the program partial
  capability against size *within* family
- at least **3 base/instruct pairs**
- target **35-40** so 30+ survive access and loop-test attrition

**Held out and excluded by construction:** the flag study's pinned reserve
(`docs/reserve-cohort-pin.md`) and its ordered bench. Those stay untouched
regardless of what this program does, and mixing them in would spend a
held-out set that was pinned before any of the flag work ran.

Writes `docs/cohort-candidates.md`. No GPU, no inference; the Hub API answers
existence and gating on paper, which is where most attrition happens.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Pinned reserve + bench from the flag study. Never candidates.
EXCLUDED = {
    "01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B", "microsoft/Phi-3.5-mini-instruct",
    "internlm/internlm2_5-7b-chat", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat", "deepseek-ai/deepseek-llm-7b-base",
    "zai-org/glm-4-9b-chat", "stabilityai/stablelm-2-12b-chat",
}

#: (family, size_b, instruct_repo, base_repo | None)
SLATE = [
    # Qwen2.5 — the deepest ladder available, six rungs with bases
    ("Qwen2.5", 0.5, "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5", 1.5, "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B"),
    ("Qwen2.5", 3.0, "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B"),
    ("Qwen2.5", 7.0, "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B"),
    ("Qwen2.5", 14.0, "Qwen/Qwen2.5-14B-Instruct", None),
    # Qwen3 — second ladder, same lab but a different generation
    ("Qwen3", 0.6, "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base"),
    ("Qwen3", 1.7, "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base"),
    ("Qwen3", 4.0, "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Base"),
    ("Qwen3", 8.0, "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"),
    # OLMo-2 — fully open, three rungs with bases
    ("OLMo-2", 1.0, "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B"),
    ("OLMo-2", 7.0, "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-1124-7B"),
    ("OLMo-2", 13.0, "allenai/OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-1124-13B"),
    # Falcon3 — four rungs
    ("Falcon3", 1.0, "tiiuae/Falcon3-1B-Instruct", "tiiuae/Falcon3-1B-Base"),
    ("Falcon3", 3.0, "tiiuae/Falcon3-3B-Instruct", "tiiuae/Falcon3-3B-Base"),
    ("Falcon3", 7.0, "tiiuae/Falcon3-7B-Instruct", "tiiuae/Falcon3-7B-Base"),
    ("Falcon3", 10.0, "tiiuae/Falcon3-10B-Instruct", None),
    # Granite — two rungs
    ("Granite-3.1", 2.0, "ibm-granite/granite-3.1-2b-instruct", None),
    ("Granite-3.1", 8.0, "ibm-granite/granite-3.1-8b-instruct", None),
    # Gemma-2 — gating is per-repo and expected to bite here
    ("Gemma-2", 2.0, "google/gemma-2-2b-it", None),
    ("Gemma-2", 9.0, "google/gemma-2-9b-it", None),
    # Llama — gating expected; included so attrition is measured, not assumed
    ("Llama-3.2", 1.0, "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B"),
    ("Llama-3.2", 3.0, "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B"),
    ("Llama-3.1", 8.0, "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B"),
    # Mistral — one rung, family breadth
    ("Mistral", 7.0, "mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-v0.3"),
]


def hub(repo: str, token: str) -> dict:
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        return {"ok": True, **json.load(urllib.request.urlopen(req, timeout=30))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "err": f"HTTP {e.code}"}
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "err": type(e).__name__}


def can_pull(repo: str, token: str) -> bool:
    """Whether THIS account can actually fetch the weights.

    `gated: manual` describes the repo, not our permission — the token holds
    accepted terms for several gated repos and this project has already served
    them. Reading the flag alone understated the cohort by four checkpoints, so
    the test is an actual file fetch.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError

    try:
        HfApi(token=token).hf_hub_download(
            repo_id=repo, filename="config.json", cache_dir="/tmp/hfprobe")
        return True
    except GatedRepoError:
        return False
    except Exception:                                         # noqa: BLE001
        return False


def main() -> int:
    token = os.environ.get("HF_TOKEN", "")
    rows = []
    for family, size, inst, base in SLATE:
        for repo, kind in ((inst, "instruct"), (base, "base")):
            if repo is None:
                continue
            if repo in EXCLUDED:
                rows.append({"family": family, "size": size, "kind": kind,
                             "repo": repo, "status": "EXCLUDED (pinned reserve)"})
                continue
            m = hub(repo, token)
            if not m["ok"]:
                status = f"MISSING ({m['err']})"
            elif m.get("gated") in ("manual", "auto", True):
                status = ("available (gated, terms accepted)"
                          if can_pull(repo, token)
                          else "BLOCKED (gated, terms not accepted)")
            else:
                status = "available"
            rows.append({"family": family, "size": size, "kind": kind,
                         "repo": repo, "status": status})

    ok = [r for r in rows if r["status"].startswith("available")]
    fams = {r["family"] for r in ok}
    ladders = {f for f in fams if len({r["size"] for r in ok if r["family"] == f}) >= 3}
    pairs = {(r["family"], r["size"]) for r in ok if r["kind"] == "base"} & \
            {(r["family"], r["size"]) for r in ok if r["kind"] == "instruct"}

    print(f"  enumerated {len(rows)}, available on paper {len(ok)}")
    print(f"  families {len(fams)}: {sorted(fams)}")
    print(f"  families with 3+ sizes: {len(ladders)} {sorted(ladders)}")
    print(f"  base/instruct pairs: {len(pairs)}")
    print()
    for r in rows:
        if not r["status"].startswith("available"):
            print(f"    {r['status']:<24} {r['repo']}")

    lines = ["# Cohort candidates — Check 2 of the feasibility gate", "",
             "Selection is by **coverage only**: family, size and base/instruct",
             "status. Nothing about expected behaviour enters, because a cohort",
             "composed against an expected outcome produces that outcome — the",
             "flag study established that the hard way.", "",
             "The flag study's pinned reserve and bench are excluded by",
             "construction and stay untouched.", "",
             f"**On paper: {len(ok)} available of {len(rows)} enumerated.**", "",
             f"- families: **{len(fams)}** (need 5) — {', '.join(sorted(fams))}",
             f"- families with 3+ sizes: **{len(ladders)}** (need 2) — "
             f"{', '.join(sorted(ladders)) or 'none'}",
             f"- base/instruct pairs: **{len(pairs)}** (need 3)", "",
             "Loop-test and determinism attrition are Check 3; these are",
             "paper-availability figures only.", "",
             "| family | size | kind | repo | status |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["family"], x["size"], x["kind"])):
        lines.append(f"| {r['family']} | {r['size']}B | {r['kind']} | "
                     f"`{r['repo']}` | {r['status']} |")
    Path("docs/cohort-candidates.md").write_text("\n".join(lines) + "\n")
    Path("results/cohort_candidates.json").write_text(
        json.dumps({"phase": "gate-zero", "rows": rows}, indent=2) + "\n")
    print("\n  wrote docs/cohort-candidates.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
