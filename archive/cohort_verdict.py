"""Gate zero's verdict — read against the capability bin that carries the load.

**The reframe this verdict is written under.** The program never needed to
explain size; it needed to separate a behavioural axis from capability. Within-
family size ladders hold training fixed and vary size, which is the wrong
contrast for that. **Capability-matched cross-family sets hold capability fixed
and vary training**, which is the discordant-case structure Stage 4 actually
needs. `OLMo-2-7B-Instruct` at 18.58 MMLU-Pro sitting *below*
`Qwen2.5-1.5B-Instruct` at 19.99 — a 4.7x size gap at matched capability across
two labs — is identifying variation in its purest form. So the cohort's job is
capability-matched contrast, and the verdict is read that way.

**Which makes one bin load-bearing.** The 20-25 MMLU-Pro bin holds 7 models
across 4 families, and **5 of those 7, and 3 of the 4 families, sit at 1.5B-3B**.
If attrition clusters at the small end, the bin collapses toward Mistral alone
and the identifying variation is back to a family concentration. So attrition is
read against *that bin's family diversity*, not against ladder completeness.

**The Qwen3 proxy gap is pinned, not repaired.** Eight Qwen3 checkpoints have no
MMLU-Pro number on the leaderboard pinned at `4d32c08` before any lookup. The
ladder therefore participates in non-proxy analysis and is **excluded from the
capability-partialled analysis**, stated as a per-ladder coverage gap. Filling
those numbers from HELM or any other source would be exactly the proxy-shopping
the pin exists to prevent — and it is tempting precisely because Qwen3 is
otherwise the cleanest ladder. Any later suggestion to do so, from anyone, is the
pin being violated.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

BIN_LO, BIN_HI = 20.0, 25.0          # the load-bearing capability bin
SMALL_END = 3.0                       # "small end" of that bin, in billions


def capability(repos: list[str], token: str) -> dict:
    """MMLU-Pro from the leaderboard pinned at 4d32c08. Cached to results/."""
    cache = Path("results/cohort_capability.json")
    if cache.exists():
        return json.loads(cache.read_text())
    want = {r.lower() for r in repos}
    found, off = {}, 0
    while off < 5200 and len(found) < len(want):
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=open-llm-leaderboard%2Fcontents&config=default&split=train"
               f"&offset={off}&length=100")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=30)).get("rows", [])
        except Exception:                                      # noqa: BLE001
            break
        if not rows:
            break
        for r in rows:
            fn = (r["row"].get("fullname") or "").lower()
            if fn in want and fn not in found:
                found[fn] = r["row"].get("MMLU-PRO")
        off += 100
    cache.write_text(json.dumps(found, indent=2) + "\n")
    return found


def main() -> int:
    cands = [r for r in json.loads(Path("results/cohort_candidates.json").read_text())["rows"]
             if r["status"].startswith("available")]
    probed = {json.loads(Path(f).read_text())["repo"]: json.loads(Path(f).read_text())
              for f in glob.glob("results/noise_*.json")}
    cap = capability([r["repo"] for r in cands], os.environ.get("HF_TOKEN", ""))

    survived = [r for r in cands if r["repo"] in probed]
    excluded = [r for r in cands if r["repo"] not in probed]

    # ---- the first line: attrition against the load-bearing bin's small end
    def in_bin(r):
        v = cap.get(r["repo"].lower())
        return isinstance(v, (int, float)) and BIN_LO <= v < BIN_HI

    bin_all = [r for r in cands if in_bin(r)]
    bin_ok = [r for r in bin_all if r["repo"] in probed]
    small_all = [r for r in bin_all if r["size"] <= SMALL_END]
    small_ok = [r for r in small_all if r["repo"] in probed]
    fam_before = {r["family"] for r in bin_all}
    fam_after = {r["family"] for r in bin_ok}

    print("GATE ZERO — VERDICT\n")
    print(f"ATTRITION AGAINST THE {BIN_LO:.0f}-{BIN_HI:.0f} BIN'S SMALL END")
    print(f"  bin before attrition : {len(bin_all)} models, "
          f"{len(fam_before)} families {sorted(fam_before)}")
    print(f"  bin after  attrition : {len(bin_ok)} models, "
          f"{len(fam_after)} families {sorted(fam_after)}")
    print(f"  small end (<={SMALL_END:.0f}B): {len(small_ok)}/{len(small_all)} survived")
    lost = [r["repo"] for r in bin_all if r["repo"] not in probed]
    print(f"  lost from the bin    : {lost or 'none'}")

    print(f"\nCOHORT\n  probed {len(survived)}/{len(cands)}; "
          f"excluded {len(excluded)}")
    for r in excluded:
        print(f"    EXCLUDED  {r['repo']}")
    fams = {r["family"] for r in survived}
    ladders = {f for f in fams
               if len({r["size"] for r in survived if r["family"] == f}) >= 3}
    pairs = {(r["family"], r["size"]) for r in survived if r["kind"] == "base"} & \
            {(r["family"], r["size"]) for r in survived if r["kind"] == "instruct"}
    print(f"  families {len(fams)}, ladders(3+ sizes) {len(ladders)}, "
          f"base/instruct pairs {len(pairs)}")

    # ---- determinism dimension
    det = [r for r in probed.values() if r["deterministic"]]
    noisy = [r for r in probed.values() if not r["deterministic"]]
    print(f"\nDETERMINISM\n  {len(det)}/{len(probed)} fully reproducible")
    if noisy:
        sds = sorted(r["adherence_sd"] for r in noisy)
        print(f"  noisy models' adherence sd: median {sds[len(sds)//2]:.3f}, "
              f"max {max(sds):.3f}")
    print("  A floor is only disqualifying relative to a target effect, and Stage 2")
    print("  profile distances do not exist yet — so this dimension is reported,")
    print("  not adjudicated, and the verdict states which effects survive it.")

    # ---- the pinned proxy gap
    no_cap = [r["repo"] for r in survived
              if not isinstance(cap.get(r["repo"].lower()), (int, float))]
    print(f"\nPINNED-PROXY COVERAGE GAP\n  {len(no_cap)} surviving models have no "
          f"MMLU-Pro on the pinned leaderboard:")
    for r in sorted(no_cap):
        print(f"    {r}")
    print("  These participate in non-proxy analysis and are EXCLUDED from the")
    print("  capability-partialled analysis. Substituting another proxy to rescue")
    print("  them is the proxy-shopping the pin at 4d32c08 forbids.")

    # ---- verdict
    print("\n" + "=" * 68)
    if len(survived) < 30:
        v, why = "NO-GO", f"only {len(survived)} models survived; 30+ required"
    elif len(fam_after) >= 4:
        v, why = ("GO-FULL",
                  f"the load-bearing bin kept {len(fam_after)} families through "
                  f"attrition; the program is MORE identifiable than the original "
                  f"size-ladder design assumed")
    elif len(fam_after) >= 2:
        v, why = ("GO-SCOPED",
                  f"the bin fell from {len(fam_before)} to {len(fam_after)} "
                  f"families; identifying variation is thinner than wanted and the "
                  f"decision is whether the thinned cohort answers the question or "
                  f"waits for gating access on the access-excluded families")
    else:
        v, why = ("NO-GO",
                  f"the load-bearing bin collapsed to {len(fam_after)} family; "
                  f"capability-matched contrast is gone and the cohort cannot "
                  f"identify the construct")
    print(f"VERDICT: {v}\n  {why}")
    print("\n  Two coverage gaps travel with this verdict regardless:")
    print("   1. Qwen3 ladder is excluded from capability-partialled analysis.")
    print(f"   2. The {BIN_LO:.0f}-{BIN_HI:.0f} bin is load-bearing for identification.")
    print("=" * 68)

    Path("results/cohort_verdict.json").write_text(json.dumps({
        "verdict": v, "reason": why, "n_survived": len(survived),
        "n_candidates": len(cands), "families": sorted(fams),
        "ladders": sorted(ladders), "n_pairs": len(pairs),
        "bin": {"lo": BIN_LO, "hi": BIN_HI,
                "families_before": sorted(fam_before),
                "families_after": sorted(fam_after),
                "small_end_survived": len(small_ok),
                "small_end_total": len(small_all), "lost": lost},
        "determinism": {"reproducible": len(det), "probed": len(probed)},
        "proxy_gap": sorted(no_cap),
    }, indent=2) + "\n")
    print("\nwrote results/cohort_verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
