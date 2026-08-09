"""V2 — does a model's fidelity survive a change of world?

The plan calls this "the criterion most likely to kill the benchmark". If the
same seven models rank differently in a walled garden than in a weather station,
the number is a property of world_v0 and not of the model, and it should be
published as a world-specificity finding rather than submitted as a model score.

**Gate:** Spearman rho >= 0.80 across worlds, and mean absolute difference below
the between-model standard deviation. Both, not either — a high rho with a large
level shift still means the score is not portable.

**Selection rule (spec §2).** Repeats are never filtered on preflight before
averaging: that is selection on an outcome-adjacent criterion, and it is what
made an earlier analysis look cleaner than it was.

The first version of this script then went on to drop whole *models* unless all
three repeats passed in both worlds — which is the same error one level up,
since preflight status is a function of the score. It also left one comparable
model out of seven and made V2 undecidable for a reason that was an artefact of
the rule rather than of the data. The correlation is now computed over every
model from the mean of all repeats, and the preflight pass fraction is reported
beside it as a validity caveat rather than used as a filter.

**Gate -1 is marginal on world_v0, and that is the more important finding.** It
is the only fatal check that ever fails here: 7 of 18 world_v0 repeats against
2 of 18 in world_v2. The lift is positive in both (+8.6 and +10.6), so the
signal is real in both; world_v0 is simply underpowered, with 12.8 of 17
entities varying against world_v2's 17.6 of 20. A benchmark whose validity gate
is a coin-flip at this n cannot carry a per-model leaderboard column whatever
the correlation says.

**Pairing.** Both worlds are played on the same three seeds, so the comparison is
within-seed. Independent draws would add sampling noise to a quantity that is
already estimated from three repeats.
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def spearman(a: list[float], b: list[float]) -> float:
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):          # average ties, or a plateau skews rho
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def load(pattern="results/x_*.json") -> dict:
    by = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        lab = Path(f).stem.split("_")[1]
        world = d.get("meta", {}).get("world_id") or "_".join(Path(f).stem.split("_")[2:4])
        by[(lab, world)].append({
            "ok": bool(d.get("preflight", {}).get("ok")),
            "fidelity": (d.get("score") or {}).get("fidelity"),
            "omission": (d.get("score") or {}).get("omission_rate"),
            "fabrication": (d.get("score") or {}).get("fabrication_rate"),
            "n_runs": d.get("n_runs_completed"),
        })
    return by


def main() -> int:
    by = load()
    if not by:
        print("no results/x_*.json yet — the V2 sweep has not landed", file=sys.stderr)
        return 2

    labs = sorted({lab for lab, _ in by})
    worlds = sorted({w for _, w in by})
    print(f"{'lab':<12}" + "".join(f"{w:>26}" for w in worlds))
    status, means = {}, {}
    for lab in labs:
        cells = []
        for w in worlds:
            reps = by.get((lab, w), [])
            oks = sum(r["ok"] for r in reps)
            vals = [r["fidelity"] for r in reps if r["fidelity"] is not None]
            if not reps:
                cells.append(f"{'—':>26}")
                continue
            if oks == 0:
                status[(lab, w)] = "NON_ELICITABLE"
            elif oks < len(reps):
                status[(lab, w)] = f"UNSTABLE {oks}/{len(reps)}"
            else:
                status[(lab, w)] = "ok"
            # Mean over ALL repeats, not the passing ones.
            if vals:
                means[(lab, w)] = st.mean(vals)
            cells.append(f"{' '.join(f'{v:.1f}' for v in vals):>16}{status[(lab,w)][:9]:>10}")
        print(f"{lab:<12}" + "".join(cells))

    if len(worlds) < 2:
        print("\nonly one world present — V2 cannot be evaluated yet")
        return 0
    w0, w1 = worlds[0], worlds[1]

    # Every model with a score in both worlds. Dropping models on preflight
    # status would select on an outcome-adjacent criterion — see the docstring.
    usable = [l for l in labs if (l, w0) in means and (l, w1) in means]
    dropped = [l for l in labs if l not in usable]
    print(f"\n{len(usable)} models scored in both worlds: {usable}")
    if dropped:
        print(f"  no score in one or both worlds: {dropped}")
    if len(usable) < 3:
        print("\nfewer than 3 comparable models — rho is not interpretable. V2 UNDECIDED.")
        return 0

    a = [means[(l, w0)] for l in usable]
    b = [means[(l, w1)] for l in usable]
    rho = spearman(a, b)
    mad = st.mean(abs(x - y) for x, y in zip(a, b))
    between_sd = st.pstdev(a + b)
    print(f"\n  Spearman rho ({w0} vs {w1}) = {rho:.3f}     gate >= 0.80")
    print(f"  mean |difference|            = {mad:.2f}")
    print(f"  between-model sd             = {between_sd:.2f}   gate: MAD < sd")
    passes = (rho >= 0.80) and (mad < between_sd)
    print(f"\n  V2 verdict: {'PASS' if passes else 'FAIL'}")
    unstable = [l for l in usable
                if status.get((l, w0)) != "ok" or status.get((l, w1)) != "ok"]
    if unstable:
        print(f"\n  CAVEAT: {len(unstable)}/{len(usable)} models had at least one repeat")
        print(f"  fail gate -1: {unstable}. Their means average repeats that")
        print(f"  individually carried no demonstrable signal, so rho here is an")
        print(f"  upper bound on what the instrument currently supports.")
    if not passes:
        print("  On failure (F2) the plan is explicit: publish as a world-specificity")
        print("  finding — a world score, not a model score — and do not submit.")

    # rho over 6-7 models saturates easily: one clear outlier plus a middle
    # cluster separated by less than its own repeat-to-repeat spread can give
    # rho = 1.0 while the ranking inside that cluster is pure noise. Resampling
    # the repeats says how much of the correlation survives that.
    import random
    rng = random.Random(19)
    boot = []
    for _ in range(4000):
        aa, bb = [], []
        for l in usable:
            ra = [r["fidelity"] for r in by[(l, w0)] if r["fidelity"] is not None]
            rb = [r["fidelity"] for r in by[(l, w1)] if r["fidelity"] is not None]
            aa.append(st.mean(rng.choices(ra, k=len(ra))))
            bb.append(st.mean(rng.choices(rb, k=len(rb))))
        r = spearman(aa, bb)
        if r == r:
            boot.append(r)
    boot.sort()
    lo, hi = boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]
    frac = sum(1 for r in boot if r >= 0.80) / len(boot)
    print(f"\n  rho bootstrapped over repeats: median {boot[len(boot)//2]:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"  fraction of resamples clearing the 0.80 gate: {frac:.2f}")

    Path("results/v2_crossworld.json").write_text(json.dumps({
        "rho_boot": {"median": boot[len(boot)//2], "ci": [lo, hi], "frac_ge_080": frac},
        "worlds": [w0, w1], "usable": usable, "dropped": dropped,
        "means": {f"{l}|{w}": v for (l, w), v in means.items()},
        "status": {f"{l}|{w}": s for (l, w), s in status.items()},
        "spearman": rho, "mean_abs_diff": mad, "between_sd": between_sd,
        "gate_pass": passes,
    }, indent=2) + "\n")
    print("\nwrote results/v2_crossworld.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
