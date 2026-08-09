"""Phase B — the split-half reliability of the reference standard itself.

**The quantity, and why the obvious one is wrong.** Pre-registration v2 set the
V1 gate from mean *pairwise* inter-judge kappa (0.736). That is agreement
between individual judges, but the standard in use is the **majority of three**,
and majority voting averages out judge-specific independent error. The aggregate
is therefore more reliable than any member of it, and the ceiling on
agreement-with-the-majority lies above the pairwise figure. v3 §1 records that
as an error in v2's reasoning.

The right quantity is the reliability of the aggregate against itself:

    ceiling = kappa( majority(J1,J2,J3), majority(J4,J5,J6) )

Two **disjoint** triples, each a majority of three, which is exactly the object
V1 scores against. Because the halves are the same size as the standard, no
Spearman–Brown step is licensed — the number is read off directly.

**The partition is fixed by a rule declared before the split** (v3 §2): judge
tags sorted alphabetically, then alternating — indices 0, 2, 4 against 1, 3, 5.
All ten partitions are reported so the pre-declared one can be seen in context,
but it is the one that decides the gate. Choosing the partition afterwards would
be selection on the outcome.

**The gate then follows by arithmetic** (v3 §3), with no discretion:

    gate = largest multiple of 0.05 strictly below ceiling, clamped [0.60, 0.80]
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.detectors import DETECTORS  # noqa: E402
from seahaven.fidelity.parse_detector import parse_mentioned  # noqa: E402


def kappa(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def boot_ci(x, y, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    ks = [k for _ in range(n)
          for idx in [rng.integers(0, len(y), len(y))]
          for k in [kappa(x[idx], y[idx])] if not np.isnan(k)]
    if not ks:
        return float("nan"), float("nan")
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def gate_from(ceiling: float) -> float:
    """v3 §3, applied mechanically."""
    g = np.floor(ceiling / 0.05 - 1e-9) * 0.05
    return float(min(0.80, max(0.60, round(g, 2))))


def main() -> int:
    judges = {}
    for f in sorted(glob.glob("results/v1b_judge_*.json")):
        d = json.loads(Path(f).read_text())
        tag = Path(f).stem.replace("v1b_judge_", "")
        if not d.get("usable"):
            print(f"  {tag}: FAILED CONTROLS — excluded")
            continue
        judges[tag] = d
        print(f"  {tag:<11} controls {d['control_accuracy']:.2f}  model={d.get('model')}")

    def key(i):
        return f'{i["stratum"]}|{i["entity"]}|{i["narrative"][:120]}'

    tables = {t: {key(i): i for i in d["items"]
                  if i.get("stable") and i.get("judge") is not None}
              for t, d in judges.items()}
    common = sorted(set.intersection(*(set(t) for t in tables.values())))
    tags = sorted(tables)
    print(f"\n{len(tags)} judges, {len(common)} items labelled stably by all")
    if len(tags) < 6:
        print(f"\nNeed 6 judges for two disjoint triples; have {len(tags)}. "
              f"Phase B incomplete — no ceiling reported.", file=sys.stderr)
        return 1

    lab = {t: np.array([int(tables[t][k]["judge"]) for k in common]) for t in tags}

    def majority(group):
        return (np.vstack([lab[t] for t in group]).sum(0) > len(group) / 2).astype(int)

    # Pre-declared partition: alphabetical, alternating.
    A = [tags[0], tags[2], tags[4]]
    B = [tags[1], tags[3], tags[5]]
    ceiling = kappa(majority(A), majority(B))
    lo, hi = boot_ci(majority(A), majority(B))
    print(f"\nPRE-DECLARED SPLIT (alphabetical, alternating)")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  ceiling = kappa(majority A, majority B) = {ceiling:.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print(f"\nall 10 partitions (context only — the pre-declared one decides):")
    alls = []
    for tri in itertools.combinations(tags, 3):
        rest = tuple(t for t in tags if t not in tri)
        if tri > rest:
            continue
        k = kappa(majority(list(tri)), majority(list(rest)))
        alls.append(k)
        mark = "  <- pre-declared" if sorted(tri) == sorted(A) or sorted(tri) == sorted(B) else ""
        print(f"  {'+'.join(tri):<34} vs {'+'.join(rest):<34} {k:.3f}{mark}")
    print(f"  median over partitions: {np.median(alls):.3f}   "
          f"range [{min(alls):.3f}, {max(alls):.3f}]")

    pairwise = [kappa(lab[a], lab[b]) for a, b in itertools.combinations(tags, 2)]
    print(f"\n  mean PAIRWISE inter-judge kappa: {np.mean(pairwise):.3f}   "
          f"<- what v2 wrongly used")
    print(f"  aggregate split-half ceiling:    {ceiling:.3f}   <- the right quantity")
    print(f"  difference: {ceiling - np.mean(pairwise):+.3f}")

    gate = gate_from(ceiling)
    print(f"\nGATE, by v3 §3 arithmetic: {gate:.2f}")
    if ceiling < 0.65:
        print("  K1 FIRES — ceiling below 0.65. The construct is not reliably")
        print("  measurable by an LLM panel. Publishable as a methodological result.")
    elif gate >= 0.80:
        print("  v2's lowering to 0.70 was NOT justified. The gate returns to 0.80")
        print("  and v3 §1 stands as the correction of that error.")

    # D1 and the string detectors against the full 6-judge majority.
    cons = majority(tags)
    rows = [tables[tags[0]][k] for k in common]
    votes = np.array([[DETECTORS["name_only"](r["narrative"], r["entity"]),
                       DETECTORS["relation_aware"](r["narrative"], r["entity"]),
                       parse_mentioned(r["narrative"], r["entity"])]
                      for r in rows], dtype=int)
    d1 = None
    p = Path("results/v1b_llm_detector_labels.json")
    if p.exists():
        m = json.loads(p.read_text())
        d1 = np.array([int(m[k]) for k in common]) if all(k in m for k in common) else None

    print(f"\nDETECTORS vs the 6-judge majority (gate {gate:.2f}, per stratum)")
    strata = sorted({r["stratum"] for r in rows})
    names = ["name-only", "relation-aware", "parse"] + (["D1 gpt-4.1-mini"] if d1 is not None else [])
    cols = [votes[:, 0], votes[:, 1], votes[:, 2]] + ([d1] if d1 is not None else [])
    print(f"  {'stratum':<16}{'n':>5}" + "".join(f"{n:>18}" for n in names))
    per = {}
    for s in strata:
        idx = np.array([i for i, r in enumerate(rows) if r["stratum"] == s])
        cells = "".join(f"{kappa(c[idx], cons[idx]):>18.3f}" for c in cols)
        per[s] = [float(kappa(c[idx], cons[idx])) for c in cols]
        print(f"  {s:<16}{len(idx):>5}{cells}")
    if d1 is not None:
        ok = all(per[s][-1] >= gate for s in strata)
        print(f"\n  V1 verdict for D1 at gate {gate:.2f}: {'PASS' if ok else 'FAIL'}")

    Path("results/ceiling.json").write_text(json.dumps({
        "judges": tags, "n_items": len(common),
        "partition_A": A, "partition_B": B,
        "ceiling": ceiling, "ceiling_ci": [lo, hi],
        "all_partitions_median": float(np.median(alls)),
        "all_partitions_range": [float(min(alls)), float(max(alls))],
        "mean_pairwise": float(np.mean(pairwise)),
        "gate": gate, "per_stratum": per, "detectors": names,
    }, indent=2) + "\n")
    print("\nwrote results/ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
