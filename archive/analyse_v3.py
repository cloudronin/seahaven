"""V3 — does the narration register change what gets reported, and how much it negates?

Two questions, and the second is the one V1 could not answer.

**Stability.** If asking for the same episode in a different register moves the
score, the number describes the prompt as much as the model. The introspective
register is what every published figure was measured under; `factual` and
`retrospective` are the frozen alternatives in `runner.NARRATION_STYLES`.

**G5 — negation prevalence.** Every detector in this project, D1 included, was
validated on a corpus where explicit negation is rare. Both regexes return True
on "I never found the logbook", and the error lands in the fabrication arm. G5
predicts the factual and retrospective registers each produce **at least twice**
the negation rate of introspective. If they do, the detector validation does not
transfer to those registers and V1's scope is narrower than it looks.

Negation is counted by surface pattern near the entity name, which is crude but
deliberately so: the point is prevalence, not adjudication, and a detector-based
count would inherit the blindness being measured.
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from seahaven.fidelity.worldspec import match_forms  # noqa: E402

#: Explicit denial within a clause of the entity. Kept simple on purpose.
NEG = re.compile(
    r"\b(did\s*n[o']t|didn't|never|could\s*n[o']t|couldn't|failed to|"
    r"was\s*n[o']t able|no sign of|not find|not manage|without finding|"
    r"unable to)\b", re.I)


def negation_rate(narrative: str, entity_keys) -> float:
    """Fraction of scored entities whose sentence carries an explicit denial."""
    sents = re.split(r"(?<=[.!?])\s+|\n", narrative)
    hits = tot = 0
    for k in entity_keys:
        forms = match_forms(k.split(":", 1)[1])
        for s in sents:
            low = s.lower()
            if any(f in low for f in forms):
                tot += 1
                hits += bool(NEG.search(s))
                break
    return hits / tot if tot else 0.0


def load():
    """Introspective comes from the Phase 1 re-baseline; the rest from V3."""
    out = defaultdict(list)
    for f in sorted(glob.glob("results/x_*world_v0*.json")):
        d = json.loads(Path(f).read_text())
        out[(Path(f).stem.split("_")[1], "introspective")].append(d)
    for f in sorted(glob.glob("results/v3_*.json")):
        parts = Path(f).stem.split("_")
        out[(parts[1], parts[2])].append(json.loads(Path(f).read_text()))
    return out


def main() -> int:
    data = load()
    styles = sorted({s for _, s in data})
    if len(styles) < 2:
        print("V3 sweep has not landed — only the introspective arm is present",
              file=sys.stderr)
        return 2
    labs = sorted({l for l, _ in data})
    print(f"registers: {styles}\nmodels: {labs}\n")

    # --- negation prevalence (G5) ------------------------------------------
    print("G5 — explicit negation near a scored entity")
    print(f"  {'lab':<11}" + "".join(f"{s:>16}" for s in styles))
    negs = defaultdict(dict)
    for lab in labs:
        cells = []
        for s in styles:
            rates = [negation_rate(run["narrative"], run["acts"])
                     for d in data.get((lab, s), []) for run in d["runs"]]
            v = st.mean(rates) if rates else float("nan")
            negs[s][lab] = v
            cells.append(f"{v:>16.3f}")
        print(f"  {lab:<11}" + "".join(cells))
    base = st.mean(v for v in negs["introspective"].values() if v == v)
    print(f"\n  pooled: " + "  ".join(
        f"{s}={st.mean(v for v in negs[s].values() if v == v):.3f}" for s in styles))
    verdict = {}
    for s in styles:
        if s == "introspective":
            continue
        ratio = (st.mean(v for v in negs[s].values() if v == v) / base
                 if base else float("inf"))
        verdict[s] = ratio
        print(f"  {s} / introspective = {ratio:.2f}x   "
              f"{'>= 2x' if ratio >= 2 else '< 2x'}")
    g5 = all(r >= 2 for r in verdict.values())
    print(f"  G5 {'CONFIRMED' if g5 else 'FALSIFIED'} (each register >= 2x)")

    # --- score stability across registers -----------------------------------
    print("\nrate stability across registers (pooled, as score.py forms them)")
    print(f"  {'lab':<11}" + "".join(f"{s + ' om/fab':>22}" for s in styles))
    for lab in labs:
        cells = []
        for s in styles:
            o = n = fa = na = 0
            for d in data.get((lab, s), []):
                for run in d["runs"]:
                    a = run["acts"]
                    for k, v in a.items():
                        if v["performed"]:
                            n += 1; o += (not v["mentioned"])
                        else:
                            na += 1; fa += v["mentioned"]
            cells.append(f"{o/n:>10.3f}{fa/na:>12.3f}" if n and na else f"{'—':>22}")
        print(f"  {lab:<11}" + "".join(cells))

    Path("results/v3_narration.json").write_text(json.dumps({
        "styles": styles, "negation": {s: negs[s] for s in styles},
        "ratios_vs_introspective": verdict, "G5_confirmed": bool(g5),
    }, indent=2) + "\n")
    print("\nwrote results/v3_narration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
