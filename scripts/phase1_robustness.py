"""PHASE 1c — the two kill criteria, read against the right uncertainty.

**A single-seed number is not a finding, and neither is a spread over one draw
of models.** `phase1_bend.py` reports the instrument's output at the pinned
seed. This script asks the two questions that decide KP-1 and KP-4, each against
the uncertainty that actually governs it:

* **KP-1 — does junk-masking generalise?** The pre-registered statistic is the
  *between-model spread delta*: removing the `other` bin should WIDEN the spread,
  as it did on both burned smoke pairs (+0.073, +0.053). The governing
  uncertainty is not subsampling — it is **which models are in the set**. So the
  delta is bootstrapped over models, not over seeds. Subsampling noise is
  removed first by averaging each model's bend over `SEEDS`, because the
  per-model direction was seed-luck: it swung 2/13 to 9/13 across twelve seeds.

* **KP-4 — is the bend just capability?** Spearman against the pinned MMLU-Pro,
  a permutation test on the correlation, and then the partialling KP-4 actually
  asks for: regress the bend on capability and check whether the **residual**
  spread still clears the within-model null floor. A correlation alone does not
  fire KP-4; a residual that falls inside the noise floor does.

**Only 9 of the 15 usable models carry a proxy**, because the seal forced all
nine uncovered models into exploration by design. The capability read is thin and
says so rather than being quietly reported as n=15.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase1_bend import SEED, load_cells, profile, strip_junk  # noqa: E402
from smoke_state_conditioned import bend  # noqa: E402

from seahaven.dimensional import seal as S  # noqa: E402

SEEDS = tuple(range(SEED, SEED + 12))
N_BOOT = 5000
N_PERM = 20000


def spearman(a, b) -> float:
    k = len(a)
    rank = lambda v: [sorted(v).index(t) for t in v]  # noqa: E731
    ra, rb = rank(a), rank(b)
    return 1 - 6 * sum((p - q) ** 2 for p, q in zip(ra, rb)) / (k * (k * k - 1))


def collect() -> dict:
    """Seed-averaged bends per model, so the reads below are not seed artefacts."""
    out = {}
    for i in range(len(S.EXPLORATION)):
        p = profile(i)
        if not (p and p.get("usable")):
            continue
        eps = load_cells(i)
        leg = strip_junk(eps)
        rec = {"null": p["full"]["null_p95"], "junk": p["junk_rate"],
               "paired": p["paired"],
               "bend": st.mean([bend(eps, p["full"]["n"], random.Random(s + i))
                                for s in SEEDS])}
        if p["paired"]:
            nc = p["n_common"]
            rec["full_c"] = st.mean([bend(eps, nc, random.Random(s + i))
                                     for s in SEEDS])
            rec["legal_c"] = st.mean([bend(leg, nc, random.Random(s + i))
                                      for s in SEEDS])
        out[p["repo"]] = rec
    return out


def kp1(rows: dict, rng: random.Random) -> dict:
    """Bootstrap the spread delta over MODELS — the uncertainty that governs it."""
    names = [r for r, v in rows.items() if v["paired"]]
    delta = lambda sel: (st.pstdev([rows[r]["legal_c"] for r in sel])  # noqa: E731
                         - st.pstdev([rows[r]["full_c"] for r in sel]))
    obs = delta(names)
    boots = []
    for _ in range(N_BOOT):
        sel = [rng.choice(names) for _ in names]
        if len(set(sel)) > 1:
            boots.append(delta(sel))
    boots.sort()
    lo, hi = boots[int(.025 * len(boots))], boots[int(.975 * len(boots))]
    unburned = [r for r in names if r not in S.BURNED]
    return {"n_models": len(names), "delta": obs, "ci95": [lo, hi],
            "frac_positive": sum(1 for b in boots if b > 0) / len(boots),
            "delta_unburned_only": delta(unburned),
            "n_unburned": len(unburned),
            "excludes_zero": bool(lo > 0),
            "burned_pair_effect_at_discovery": [0.073, 0.053],
            "fires": bool(lo <= 0)}


def kp4(rows: dict, rng: random.Random) -> dict:
    """Correlation, permutation test, then the partialling KP-4 asks for."""
    cov = [(r, S.COHORT[r][3], v["bend"]) for r, v in rows.items()
           if S.COHORT[r][3] is not None]
    x = [c[1] for c in cov]
    y = [c[2] for c in cov]
    n = len(cov)
    rho = spearman(y, x)

    perm = []
    for _ in range(N_PERM):
        s = y[:]
        rng.shuffle(s)
        perm.append(abs(spearman(s, x)))
    pval = sum(1 for t in perm if t >= abs(rho)) / len(perm)

    mx, my = st.mean(x), st.mean(y)
    b1 = (sum((a - mx) * (c - my) for a, c in zip(x, y))
          / sum((a - mx) ** 2 for a in x))
    b0 = my - b1 * mx
    res = [c - (b0 + b1 * a) for a, c in zip(x, y)]
    r2 = 1 - sum(e * e for e in res) / sum((c - my) ** 2 for c in y)
    floor = st.median([rows[c[0]]["null"] for c in cov])
    resid_spread = st.pstdev(res)

    sizes = [S.COHORT[r][1] for r in rows]
    return {"n_covered": n, "n_total": len(rows),
            "spearman_capability": rho, "permutation_p": pval,
            "spearman_size": spearman([v["bend"] for v in rows.values()], sizes),
            "spearman_junk": spearman([v["bend"] for v in rows.values()],
                                      [v["junk"] for v in rows.values()]),
            "r2_capability": r2,
            "raw_spread": st.pstdev(y), "residual_spread": resid_spread,
            "null_floor": floor,
            "residual_clears_floor": bool(resid_spread > floor),
            "fires": bool(resid_spread <= floor),
            "residuals": {c[0]: e for c, e in zip(cov, res)}}


def main() -> int:
    S.assert_sealed()
    rows = collect()
    rng = random.Random(99)
    k1, k4 = kp1(rows, rng), kp4(rows, rng)

    print("PHASE 1c ROBUSTNESS — the two kill criteria")
    print(f"seal {S.SEAL_HASH[:16]}   {len(rows)} usable models   "
          f"bends averaged over {len(SEEDS)} seeds\n")

    print("KP-1 — does junk-masking generalise beyond the burned pairs?")
    print(f"  spread delta = {k1['delta']:+.4f}   "
          f"bootstrap over models 95% CI [{k1['ci95'][0]:+.4f}, "
          f"{k1['ci95'][1]:+.4f}]  ({k1['n_models']} models)")
    print(f"  at discovery on the burned pairs: +0.073, +0.053")
    print(f"  unburned models only ({k1['n_unburned']}): "
          f"{k1['delta_unburned_only']:+.4f}")
    print(f"  -> KP-1 {'FIRES' if k1['fires'] else 'does not fire'}: CI "
          f"{'includes' if k1['fires'] else 'excludes'} zero\n")

    print("KP-4 — is the bend capability in disguise?")
    print(f"  Spearman(bend, MMLU-Pro) = {k4['spearman_capability']:+.3f}  "
          f"n={k4['n_covered']}/{k4['n_total']}  perm p = "
          f"{k4['permutation_p']:.4f}")
    print(f"  Spearman(bend, size_B)   = {k4['spearman_size']:+.3f}   "
          f"Spearman(bend, junk%) = {k4['spearman_junk']:+.3f}")
    print(f"  partialling capability:  raw spread {k4['raw_spread']:.4f} -> "
          f"residual {k4['residual_spread']:.4f}  (R^2 {k4['r2_capability']:.3f})")
    print(f"  within-model null floor  {k4['null_floor']:.4f}")
    print(f"  -> KP-4 {'FIRES' if k4['fires'] else 'does not fire'}: residual "
          f"spread {'falls inside' if k4['fires'] else 'clears'} the noise floor")

    out = {"phase": "exploration", "seal": S.SEAL_HASH, "seeds": list(SEEDS),
           "models": rows, "KP1": k1, "KP4": k4}
    Path("results/phase1_robustness.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("\nwrote results/phase1_robustness.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
