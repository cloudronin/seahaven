"""Judge and score the test–retest run, then answer one question.

    Can the fidelity score rank models?

Run against the artifact from `scripts/gpu_job13`:

    python scripts/score_fidelity_repeats.py results/fidelity_repeats.json

Judging happens here rather than on the GPU so the judge can be swapped, or the
whole thing re-scored under the strict act/result reading, without repaying for
generation.

**Scored twice, deliberately.** Judge and regex disagree by up to 0.70 on
`taking` in the existing data, and a reliability verdict that depends on which
instrument was used is not a verdict. Both are reported; if they disagree on
*publishability*, that is the finding.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.analysis.mention_judge import MentionJudge  # noqa: E402
from seahaven.fidelity.runner import ACT_CLASSES, _REGEX  # noqa: E402
from seahaven.fidelity.score import ActOutcome, reliability, score  # noqa: E402


def outcomes_for(rep: dict, judge: MentionJudge | None) -> list[ActOutcome]:
    import re

    out = []
    for run in rep["runs"]:
        nar = run["narrative"]
        for act in ACT_CLASSES:
            performed = run["performed"][act]
            if judge is None:
                mentioned = bool(re.search(_REGEX[act], nar.lower()))
            else:
                mentioned = judge.judge(nar, ACT_CLASSES[act]["description"]).mentioned
            out.append(ActOutcome(act, performed, mentioned))
    return out


def main(path: str) -> int:
    data = [r for r in json.loads(Path(path).read_text()) if r["status"] == "ok"]
    judge = MentionJudge().load()

    results = {}
    for arm, j in (("judge", judge), ("regex", None)):
        by_model: dict[str, list[float]] = {}
        detail: dict[str, list] = {}
        for r in data:
            for rep in r["repeats"]:
                s = score(outcomes_for(rep, j))
                detail.setdefault(r["lab"], []).append(s.as_dict())
                if s.fidelity is not None:
                    by_model.setdefault(r["lab"], []).append(s.fidelity)
        results[arm] = {"scores": by_model, "reliability": reliability(by_model),
                        "detail": detail}
        print(f"  {arm} scored", flush=True)

    print("\n=== FIDELITY, three repeats per checkpoint ===")
    for arm in ("judge", "regex"):
        sc = results[arm]["scores"]; rel = results[arm]["reliability"]
        print(f"\n--- {arm}")
        print(f"  {'lab':<12}{'repeat 1':>10}{'repeat 2':>10}{'repeat 3':>10}{'mean':>9}{'sd':>8}")
        for lab, v in sc.items():
            cells = "".join(f"{x:>10.1f}" for x in v)
            print(f"  {lab:<12}{cells}{st.mean(v):>9.1f}{(st.pstdev(v) if len(v)>1 else 0):>8.2f}")
        print(f"  within-model sd {rel.get('within_sd')}   between-model sd "
              f"{rel.get('between_sd')}   share_between {rel.get('share_between')}")
        print(f"  -> {'PUBLISHABLE per-model' if rel.get('publishable') else 'NOT publishable per-model'}")

    agree = (results["judge"]["reliability"].get("publishable")
             == results["regex"]["reliability"].get("publishable"))
    print(f"\n  instruments agree on publishability: {agree}")
    if not agree:
        print("  -> the verdict depends on the instrument, so there is no verdict yet.")

    Path("results/fidelity_reliability.json").write_text(
        json.dumps(results, indent=2) + "\n")
    print("\n  wrote results/fidelity_reliability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "results/fidelity_repeats.json"))
