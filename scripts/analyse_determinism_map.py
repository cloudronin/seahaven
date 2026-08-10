"""The determinism map: per-model noise floors, and whether they track size.

Reads the `noise_*.json` files Check 3 produces and answers three things the
gate's verdict needs:

1. **Which models reproduce, and which do not** — the map itself. TRAP 35
   established that `VLLM_BATCH_INVARIANT=1` binds for some models and not
   others, so "the flag is set" stopped being a claim about reproducibility.
2. **Does the noise floor track size?** Two points suggested it might: Llama-8B
   binds, Qwen-0.5B does not. If it holds across the cohort, precision is worst
   exactly where the flag study found every one of its flags and where the
   dimensional program expects its interesting variance. That would change Stage
   2's per-model repeat budget and therefore the program's cost, which is why it
   is settled inside the gate rather than after it.
3. **Which flag-study cohort models were reproducible** — documentation, not
   remediation. The published negative does not move: it rests on rho +0.72 /
   +0.92, a perfectly separating split, and 20-33 point margins, none of which a
   few points of sampling noise reaches, and the cluster bootstraps already
   absorbed this variance. But any future claim leaning on byte-identity or a
   narrow margin should be checkable against whether that specific model
   reproduces. **The corpus is not re-run.**

The floor is reported in the quantity the program consumes — command-level
adherence and the verb distribution — because a Stage 2 distance has to be read
against a model's own floor, and fidelity is not what Stage 2 measures.
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

#: The twelve models the flag study actually scored, by HF repo.
FLAG_COHORT = {
    "Qwen/Qwen3-8B": "Alibaba", "mistralai/Mistral-7B-Instruct-v0.3": "MistralAI",
    "allenai/OLMo-2-1124-13B-Instruct": "AI2",
    "ibm-granite/granite-3.1-8b-instruct": "IBM",
    "tiiuae/Falcon3-10B-Instruct": "TII",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta", "google/gemma-2-9b-it": "Google",
    "Qwen/Qwen2.5-0.5B-Instruct": "AlibabaSmall",
    "ibm-granite/granite-3.1-2b-instruct": "IBMSmall",
    "allenai/OLMo-2-0425-1B-Instruct": "AI2Small",
    "Qwen/Qwen2.5-3B-Instruct": "AlibabaMid",
    "allenai/OLMo-2-1124-7B-Instruct": "AI2Mid",
}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    rows = [json.loads(Path(f).read_text())
            for f in sorted(glob.glob("results/noise_*.json"))]
    if not rows:
        print("no results/noise_*.json — run Check 3 first", file=sys.stderr)
        return 2

    rows.sort(key=lambda r: (r["size_b"], r["repo"]))
    det = [r for r in rows if r["deterministic"]]
    print(f"THE DETERMINISM MAP — {len(rows)} models probed, {len(det)} fully "
          f"reproducible ({len(det)/len(rows):.0%})\n")
    print(f"  {'model':<44}{'size':>6}{'kind':>9}{'ident':>7}{'adh_sd':>8}"
          f"{'verbTV':>8}")
    for r in rows:
        print(f"  {r['repo']:<44}{r['size_b']:>5}B{r['kind']:>9}"
              f"{r['runs_identical_frac']:>7.2f}{r['adherence_sd']:>8.3f}"
              f"{r['verb_tv_mean']:>8.4f}"
              + ("" if r["deterministic"] else "  noisy"))

    # ---- does the floor track size?
    sizes = [r["size_b"] for r in rows]
    lsize = [math.log(s) for s in sizes]
    print("\nDOES THE NOISE FLOOR TRACK SIZE?")
    for label, key in (("runs identical (higher = cleaner)", "runs_identical_frac"),
                       ("adherence sd (lower = cleaner)", "adherence_sd"),
                       ("verb-dist TV  (lower = cleaner)", "verb_tv_mean")):
        vals = [r[key] for r in rows]
        rho = spearman(lsize, vals)
        print(f"  Spearman(log size, {label:<34}) = {rho:+.3f}")

    small = [r for r in rows if r["size_b"] <= 3]
    large = [r for r in rows if r["size_b"] >= 7]
    if small and large:
        print(f"\n  <=3B  ({len(small):>2} models): "
              f"{sum(r['deterministic'] for r in small)}/{len(small)} reproducible, "
              f"median adh_sd {st.median(r['adherence_sd'] for r in small):.3f}")
        print(f"  >=7B  ({len(large):>2} models): "
              f"{sum(r['deterministic'] for r in large)}/{len(large)} reproducible, "
              f"median adh_sd {st.median(r['adherence_sd'] for r in large):.3f}")

    # ---- what the floor means against the effects the program wants to see
    floors = [r["adherence_sd"] for r in rows if not r["deterministic"]]
    if floors:
        print(f"\n  Noisy models' adherence sd: median {st.median(floors):.3f}, "
              f"max {max(floors):.3f}")
        print(f"  Published within-model sd was 3.09; the flag study's smallest "
              f"decisive\n  margin was ~20 points. A floor is only disqualifying "
              f"relative to an effect.")

    # ---- directive 4: which flag-cohort models reproduce. Documentation only.
    print("\nFLAG-STUDY COHORT — which of the twelve reproduce")
    print("  (documentation, not remediation; the corpus is NOT re-run)")
    seen = {r["repo"]: r for r in rows}
    for repo, lab in sorted(FLAG_COHORT.items(), key=lambda kv: kv[1]):
        r = seen.get(repo)
        if r is None:
            print(f"    {lab:<14}{repo:<44} not probed")
        else:
            print(f"    {lab:<14}{repo:<44}"
                  f"{'reproducible' if r['deterministic'] else 'NOISY'}"
                  f"  (adh_sd {r['adherence_sd']:.3f})")

    out = {"phase": "gate-zero", "n_probed": len(rows),
           "n_deterministic": len(det),
           "spearman_logsize_vs_identical":
               spearman(lsize, [r["runs_identical_frac"] for r in rows]),
           "spearman_logsize_vs_adherence_sd":
               spearman(lsize, [r["adherence_sd"] for r in rows]),
           "models": rows}
    Path("results/determinism_map.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/determinism_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
