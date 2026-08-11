"""Read the three arms. **The control decides whether the treatment is readable.**

Arms A and B are identical configuration (4096). Arm C is the treatment (8192).

The previous run of this check compared 4096 against 8192 with no same-config
baseline and returned "differs on one seed of two". That is uninterpretable: one
identical and one differing is the signature of ambient nondeterminism, not of a
systematic config effect, and nothing in that design could tell the two apart.

**TRAP 32 in the research log is this exact error, committed before**: a
byte-identity regression run ahead of the determinism control, where the control
then refuted the hypothesis outright. So A-vs-B is evaluated first, and if it
fails, A-vs-C is not reported as a result at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

SEEDS = [5150, 7301]


def sig(path: str) -> str:
    d = json.loads(open(path).read())
    cmds = [c for r in d["runs"] for c in r.get("commands", [])]
    return hashlib.sha256(json.dumps(cmds, sort_keys=True).encode()).hexdigest()


def load(arm: str, ctx: int) -> dict:
    out = {}
    for s in SEEDS:
        p = f"/tmp/results/ctx{arm}{ctx}_{s}.json"
        out[s] = sig(p) if os.path.exists(p) else None
    return out


def main() -> int:
    A, B, C = load("A", 4096), load("B", 4096), load("C", 8192)
    print(f"  {'seed':<8}{'A(4096)':<20}{'B(4096)':<20}{'C(8192)':<20}")
    for s in SEEDS:
        print(f"  {s:<8}{str(A[s])[:16]:<20}{str(B[s])[:16]:<20}{str(C[s])[:16]:<20}")

    have = all(A[s] and B[s] and C[s] for s in SEEDS)
    if not have:
        print("\n  MISSING ARM — cannot read")
        verdict, control, treat = "INCONCLUSIVE", None, None
    else:
        control = all(A[s] == B[s] for s in SEEDS)
        treat = all(A[s] == C[s] for s in SEEDS)
        print(f"\n  CONTROL   4096 vs 4096: "
              f"{'REPRODUCIBLE' if control else 'NOT REPRODUCIBLE'}")
        if not control:
            print("  -> this harness is nondeterministic run-to-run at this "
                  "concurrency, so ANY 4096-vs-8192 difference is ambient.")
            print("  -> the context-length question cannot be answered by "
                  "comparing single runs; it needs serial cells or a working "
                  "batch invariance first.")
            verdict = "INCONCLUSIVE"
        else:
            print(f"  TREATMENT 4096 vs 8192: "
                  f"{'IDENTICAL' if treat else 'DIFFERS'}")
            verdict = "CLEAN" if treat else "CONFOUNDED"
            print("  -> " + ("floors transfer; the E-sweep is clean"
                             if treat else
                             "context length changes generation even far below "
                             "the limit; the floors were measured at 4096 and "
                             "need a caveat plus re-measurement at 8192"))

    # Which seeds moved, when the control itself is unstable — this is what
    # distinguishes "one flaky seed" from "nothing reproduces".
    if have and not control:
        for s in SEEDS:
            print(f"    seed {s}: A==B {A[s] == B[s]}   A==C {A[s] == C[s]}")

    json.dump({"phase": "exploration", "verdict": verdict,
               "control_reproducible": control, "treatment_identical": treat,
               "model": "meta-llama/Llama-3.1-8B-Instruct",
               "arms": {"A": 4096, "B": 4096, "C": 8192},
               "signatures": {"A": A, "B": B, "C": C}},
              open("/tmp/results/ctx_noop_check.json", "w"), indent=2)
    print(f"\n  VERDICT: {verdict}")
    return 0 if verdict == "CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
