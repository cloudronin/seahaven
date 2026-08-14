"""Phase A — everything recomputed under the frozen D1 detector.

Decides G1, G2a, G2b and G3 from `prereg-v3`. No new episodes: the same 500
rollouts, the same ground truth, only the *mention* side swapped from a regex
that scores exactly 0.000 on the decisive stratum to `gpt-4.1-mini` under the
frozen prompt and dual-ask rule.

Scores are recomputed from `acts` rather than read from the files, because the
`score` block in each result was produced by the regex and is stale by
construction here.

**G2b is the reason this script exists in this shape.** A detector that merely
adds noise shrinks every correlation in the table, so a fallen verbosity
confound is not evidence of a fix on its own. The between-model spread and the
cross-world correlation are computed under both detectors and reported beside
the length slope, so uniform attenuation is visible rather than flattering.
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyse_v2 import spearman  # noqa: E402
from analyse_v4 import pearson  # noqa: E402


def load(pattern):
    """One row per episode, scored from `acts` under whichever detector wrote them."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        lab = Path(f).stem.split("_")[1]
        world = d["meta"]["world_id"]
        for run in d["runs"]:
            acts = run["acts"]
            perf = [k for k, v in acts.items() if v["performed"]]
            absent = [k for k, v in acts.items() if not v["performed"]]
            if not perf or not absent:
                continue
            om = sum(1 for k in perf if not acts[k]["mentioned"]) / len(perf)
            fab = sum(1 for k in absent if acts[k]["mentioned"]) / len(absent)
            rows.append({"lab": lab, "world": world,
                         "words": len(run["narrative"].split()),
                         "n_performed": len(perf), "omission": om,
                         "fabrication": fab,
                         "n_om": sum(1 for k in perf if not acts[k]["mentioned"]),
                         "n_perf": len(perf),
                         "n_fab": sum(1 for k in absent if acts[k]["mentioned"]),
                         "n_absent": len(absent),
                         "fidelity": 100 * ((1 - om) + (1 - fab)) / 2})
    return rows


def by_model_world(rows):
    """POOLED, matching `score.py`: rates formed once over all observations.

    Averaging per-episode fidelity instead is a mean of ratios, and episodes
    with few performed entities make it badly behaved — it moved the regex
    cross-world rho from 0.964 to 0.679 on identical data. The published metric
    pools, so the baseline must pool or the comparison is between two different
    estimators rather than two detectors.
    """
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        c = agg[(r["lab"], r["world"])]
        c[0] += r["n_om"]; c[1] += r["n_perf"]
        c[2] += r["n_fab"]; c[3] += r["n_absent"]
    return {k: 100 * ((1 - o / n) + (1 - fa / na)) / 2
            for k, (o, n, fa, na) in agg.items()}


def boot_rho(rows, labs, w0, w1, n=4000, seed=19):
    import random
    rng = random.Random(seed)
    per = defaultdict(list)
    for r in rows:
        per[(r["lab"], r["world"])].append(r)
    def pooled(sample):
        o = sum(x["n_om"] for x in sample); n_ = sum(x["n_perf"] for x in sample)
        fa = sum(x["n_fab"] for x in sample); na = sum(x["n_absent"] for x in sample)
        return 100 * ((1 - o / n_) + (1 - fa / na)) / 2
    out = []
    for _ in range(n):
        a = [pooled(rng.choices(per[(l, w0)], k=len(per[(l, w0)]))) for l in labs]
        b = [pooled(rng.choices(per[(l, w1)], k=len(per[(l, w1)]))) for l in labs]
        r = spearman(a, b)
        if r == r:
            out.append(r)
    out.sort()
    return out[len(out) // 2], out[int(.025 * len(out))], out[int(.975 * len(out))]


def main() -> int:
    reg = load("results/x_*.json")
    d1 = load("results/d1/x_*.json")
    if not d1:
        print("results/d1/ is empty — Phase A rescoring has not finished",
              file=sys.stderr)
        return 2
    print(f"regex: {len(reg)} episodes | D1: {len(d1)} episodes\n")

    labs = sorted({r["lab"] for r in reg})
    mr, md = by_model_world(reg), by_model_world(d1)

    # ---- G1: does the ranking change? -------------------------------------
    print("G1 — does swapping the detector reorder the world_v0 table?")
    print(f"  {'lab':<11}{'regex':>9}{'D1':>9}{'shift':>9}")
    for l in labs:
        print(f"  {l:<11}{mr[(l,'world_v0')]:>9.1f}{md[(l,'world_v0')]:>9.1f}"
              f"{md[(l,'world_v0')]-mr[(l,'world_v0')]:>+9.1f}")
    rank_r = [l for _, l in sorted((mr[(l, "world_v0")], l) for l in labs)][::-1]
    rank_d = [l for _, l in sorted((md[(l, "world_v0")], l) for l in labs)][::-1]
    print(f"  regex rank: {rank_r}")
    print(f"  D1    rank: {rank_d}")
    g1 = rank_r != rank_d
    print(f"  G1 {'CONFIRMED' if g1 else 'FALSIFIED'} — ranking "
          f"{'changed' if g1 else 'identical'}\n")

    # ---- G2a / G2b: verbosity, and whether any fall is selective ----------
    print("G2a — does the verbosity confound survive D1?")
    out = {}
    for nm, rows in (("regex", reg), ("D1", d1)):
        mw = [st.mean(x["words"] for x in rows if x["lab"] == l) for l in labs]
        mo = [st.mean(x["omission"] for x in rows if x["lab"] == l) for l in labs]
        pm = by_model_world(rows)
        mf = [st.mean([pm[(l, w)] for w in sorted({r["world"] for r in rows})])
              for l in labs]
        mn = [st.mean(x["n_performed"] for x in rows if x["lab"] == l) for l in labs]
        out[nm] = {
            "r_words_omission": pearson(mw, mo),
            "between_sd": st.pstdev(mf),
            "r_nperf_fidelity": pearson(mn, mf),
        }
        print(f"  {nm:<6} model-level r(words, omission) = "
              f"{out[nm]['r_words_omission']:+.3f}")
    g2a = out["D1"]["r_words_omission"] < -0.35
    print(f"  G2a {'CONFIRMED' if g2a else 'FALSIFIED'} "
          f"(threshold: more negative than −0.35)")

    print("\nG2b — if the confound fell, was the fall SELECTIVE?")
    drop = 1 - out["D1"]["between_sd"] / out["regex"]["between_sd"]
    print(f"  between-model sd: {out['regex']['between_sd']:.2f} -> "
          f"{out['D1']['between_sd']:.2f}  ({drop*100:+.0f}%)")
    worlds = sorted({r["world"] for r in reg})
    rho_d = spearman([md[(l, worlds[0])] for l in labs],
                     [md[(l, worlds[1])] for l in labs])
    rho_r = spearman([mr[(l, worlds[0])] for l in labs],
                     [mr[(l, worlds[1])] for l in labs])
    print(f"  cross-world rho:  {rho_r:.3f} -> {rho_d:.3f}")
    if g2a:
        print("  G2a held, so G2b is not triggered — the confound did not fall.")
    else:
        selective = (drop <= 0.20) and (rho_d >= 0.80)
        print(f"  G2b {'PASSED' if selective else 'FAILED'} — "
              f"{'selective fall, D1 credited with fixing the confound' if selective else 'UNIFORM ATTENUATION: D1 added noise and only looked like a fix'}")

    print(f"\n  r(n_performed, fidelity): {out['regex']['r_nperf_fidelity']:+.3f} "
          f"-> {out['D1']['r_nperf_fidelity']:+.3f}   (G6 target: |r| < 0.20)")

    # ---- G3: does V2 stay marginal? ---------------------------------------
    print("\nG3 — is V2 still a marginal pass under D1?")
    med, lo, hi = boot_rho(d1, labs, worlds[0], worlds[1])
    print(f"  point rho = {rho_d:.3f}")
    print(f"  bootstrap  = {med:.3f} [{lo:.3f}, {hi:.3f}]")
    g3 = (rho_d >= 0.80) and (lo < 0.80)
    print(f"  G3 {'CONFIRMED' if g3 else 'FALSIFIED'} "
          f"(>=0.80 point, lower bound <0.80)")

    Path("results/phaseA.json").write_text(json.dumps({
        "n_episodes": {"regex": len(reg), "d1": len(d1)},
        "world_v0_means": {"regex": {l: mr[(l, "world_v0")] for l in labs},
                           "d1": {l: md[(l, "world_v0")] for l in labs}},
        "rank": {"regex": rank_r, "d1": rank_d},
        "G1_confirmed": bool(g1), "G2a_confirmed": bool(g2a),
        "metrics": out, "cross_world_rho": {"regex": rho_r, "d1": rho_d},
        "G3_confirmed": bool(g3),
        "bootstrap_rho_d1": {"median": med, "ci": [lo, hi]},
    }, indent=2) + "\n")
    print("\nwrote results/phaseA.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
