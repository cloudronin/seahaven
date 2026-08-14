"""V1 rebuild — is an LLM judge measuring the construct, or measuring its own lab?

**The finding this is built to test.** gpt-4.1-mini used as a detector scored
kappa 0.864 against GPT-5.2's labels, far ahead of every regex (best 0.72). But
both are OpenAI, so shared blind spots inflate agreement and the number may be
agreement-with-OpenAI rather than accuracy.

**So the decisive quantity is judge-to-judge agreement across labs**, not any
judge's agreement with a detector. Three judges from three labs (OpenAI,
Alibaba, Google) answer the same 806 items:

- if cross-lab agreement is comparable to within-lab, LLM judging is measuring
  something real and V1 can use it as the second instrument;
- if cross-lab is markedly lower, the 0.864 was self-agreement, and V1 remains
  unvalidated no matter how good that number looked.

**A majority vote is only reported if the judges agree.** Pooling three judges
that disagree produces a consensus that belongs to no one and hides the
disagreement inside a number that looks authoritative.

Strata are never pooled: `main`, `disagreement` and `fabrication` are separate
estimates, and the fabrication stratum is the one two of P4's four branches put
the raidex column on.
"""

from __future__ import annotations

import glob
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


def boot_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    ks = [k for _ in range(n)
          for idx in [rng.integers(0, len(y), len(y))]
          for k in [kappa(x[idx], y[idx])] if not np.isnan(k)]
    if not ks:
        return float("nan"), float("nan")
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def load_judges():
    out = {}
    for f in sorted(glob.glob("results/v1b_judge_*.json")):
        d = json.loads(Path(f).read_text())
        tag = Path(f).stem.replace("v1b_judge_", "")
        if not d.get("usable"):
            print(f"  {tag}: FAILED CONTROLS ({d.get('control_accuracy')}) — excluded")
            continue
        print(f"  {tag}: control accuracy {d['control_accuracy']:.2f}, "
              f"{len(d['items'])} items, model={d.get('model')}")
        out[tag] = d["items"]
    return out


def main() -> int:
    print("judges")
    judges = load_judges()
    if len(judges) < 2:
        print("\nneed at least two judges to measure cross-lab agreement",
              file=sys.stderr)
        return 2

    # Align on items all judges labelled stably. An item one judge could not
    # decide is not evidence about the others.
    def key(i):
        return f'{i["stratum"]}|{i["entity"]}|{i["narrative"][:120]}'

    tables = {t: {key(i): i for i in items if i.get("stable")
                  and i.get("judge") is not None}
              for t, items in judges.items()}
    common = set.intersection(*(set(t) for t in tables.values()))
    common = sorted(common)
    print(f"\n{len(common)} items labelled stably by all {len(tables)} judges")

    tags = sorted(tables)
    lab = {t: np.array([int(tables[t][k]["judge"]) for k in common]) for t in tags}

    print("\nJUDGE-TO-JUDGE AGREEMENT — the number that decides V1")
    print(f"  {'pair':<26}{'kappa':>8}   95% CI          %claim each")
    cross = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            k = kappa(lab[a], lab[b])
            lo, hi = boot_ci(lab[a], lab[b])
            cross.append(k)
            print(f"  {a + ' vs ' + b:<26}{k:>8.3f}   [{lo:+.3f}, {hi:+.3f}]"
                  f"   {100*lab[a].mean():.0f}% / {100*lab[b].mean():.0f}%")
    print(f"\n  mean cross-lab judge agreement: {np.mean(cross):.3f}")
    print(f"  reference — gpt-4.1-mini vs GPT-5.2, same lab: 0.864")

    # Consensus only where the judges actually agree.
    stack = np.vstack([lab[t] for t in tags])
    unanimous = (stack.sum(0) == 0) | (stack.sum(0) == len(tags))
    print(f"  judges unanimous on {unanimous.mean()*100:.0f}% of items")
    consensus = (stack.sum(0) > len(tags) / 2).astype(int)

    rows = [{"entity": tables[tags[0]][k]["entity"],
             "narrative": tables[tags[0]][k]["narrative"],
             "stratum": tables[tags[0]][k]["stratum"]} for k in common]
    votes = np.array([[DETECTORS["name_only"](r["narrative"], r["entity"]),
                       DETECTORS["relation_aware"](r["narrative"], r["entity"]),
                       parse_mentioned(r["narrative"], r["entity"])]
                      for r in rows], dtype=int)

    print("\nDETECTOR vs each judge, and vs consensus (V1 gate = 0.80)")
    names = ["name-only", "relation-aware", "parse"]
    hdr = "".join(f"{t:>12}" for t in tags)
    print(f"  {'detector':<18}{hdr}{'consensus':>12}")
    for j, nm in enumerate(names):
        cells = "".join(f"{kappa(votes[:, j], lab[t]):>12.3f}" for t in tags)
        print(f"  {nm:<18}{cells}{kappa(votes[:, j], consensus):>12.3f}")
    maj3 = (votes.sum(1) >= 2).astype(int)
    cells = "".join(f"{kappa(maj3, lab[t]):>12.3f}" for t in tags)
    print(f"  {'majority of 3':<18}{cells}{kappa(maj3, consensus):>12.3f}")

    print("\nPER STRATUM against consensus — never pooled")
    strata = sorted({r["stratum"] for r in rows})
    print(f"  {'stratum':<16}{'n':>5}{'%claim':>8}" + "".join(f"{n:>16}" for n in names))
    verdict = {}
    for s in strata:
        idx = np.array([i for i, r in enumerate(rows) if r["stratum"] == s])
        cells = "".join(f"{kappa(votes[idx, j], consensus[idx]):>16.3f}"
                        for j in range(3))
        best = max(kappa(votes[idx, j], consensus[idx]) for j in range(3))
        verdict[s] = best
        print(f"  {s:<16}{len(idx):>5}{100*consensus[idx].mean():>7.0f}%{cells}")
    print(f"\n  best detector per stratum: "
          f"{ {s: round(v, 3) for s, v in verdict.items()} }")
    passed = all(v >= 0.80 for v in verdict.values())
    print(f"  V1 verdict: {'PASS' if passed else 'FAIL'}  (gate 0.80 on every stratum)")

    Path("results/v1b_summary.json").write_text(json.dumps({
        "judges": tags, "n_common": len(common),
        "cross_lab_kappa": {f"{tags[i]}|{tags[j]}": kappa(lab[tags[i]], lab[tags[j]])
                            for i in range(len(tags)) for j in range(i + 1, len(tags))},
        "mean_cross_lab": float(np.mean(cross)),
        "unanimous_fraction": float(unanimous.mean()),
        "detector_vs_consensus": {names[j]: kappa(votes[:, j], consensus)
                                  for j in range(3)},
        "per_stratum_best": verdict, "gate_pass": bool(passed),
    }, indent=2) + "\n")
    print("\nwrote results/v1b_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
