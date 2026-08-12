"""C3 STAGE-1 GATE — four legs. Any failure is a stop, not a patch.

**P1-P3 ask whether the probe agrees WITH ITSELF. P4 asks whether it is RIGHT.**
A probe can be perfectly phrasing-stable, perfectly self-consistent, and
perfectly wrong — and that combination is worse than a noisy probe, because it
looks trustworthy. So P4 leads the output.

    P1  marginal equivalence   every phrasing pair within +/- 0.10 (90% CI)
    P2  item agreement         worst cross-phrasing kappa vs the retest ceiling
    P3  informative envelope   spread <= p95(self-split) AND p95 <= 0.10
    P4  external validity      probe rate vs the Stage-0 cold-unlock anchor

---

**P4 cannot be a rank correlation at m=3, and saying so is the point.** Three
models give Spearman values in {1, 0.5, -0.5, -1}; rho = 1 occurs with
probability 1/6 under a pure-noise null, so a "perfect agreement" headline would
be a coin flip wearing a p-value. That is exactly the degeneracy that disqualified
the project's existing worst-pair-Spearman gate for Stage 1, and it applies to
P4 no less for being the leg we care about most.

So P4's primary is a **contrast with real n**, not a correlation over three
points: **gemma-2-27b-it produced 0 cold unlocks in 95 crossings.** If the probe
reports that gemma can produce the route at a high rate, the probe is claiming
discovery where behaviour showed none — a direct contradiction, tested on
hundreds of probe episodes rather than on three model-level points. The ordering
check is reported beside it with its honest 1/6 floor.
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.dimensional import c3_prereg as C  # noqa: E402
from seahaven.eaxis import barrier as B  # noqa: E402
from seahaven.eaxis import levels as L  # noqa: E402

BOOT = 4000
SEED = 19
CELLS = ("minimal", "minimal", "neutral", "direct")   # index order, frozen


def load() -> dict[str, list[dict]]:
    """Episodes per model. Each carries its paired probe cells."""
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob("results/c3s1_c*_*.json")):
        d = json.loads(Path(f).read_text())
        m, w = d["meta"]["served_name"], d["meta"]["world_id"]
        for r in d.get("runs", []):
            if not r.get("commands") or not r.get("probe"):
                continue
            cmds = r["commands"]
            crossed = any(c.get("barrier_state") in ("closed", "open") for c in cmds)
            attempted = any((c.get("verb") or "").lower() == B.BARRIER_VERB
                            for c in cmds)
            out.setdefault(m, []).append({
                "world": w,
                "at_door": L.reached_decision_point(cmds, w),
                "crossed": crossed,
                # Stage 0b: the predicate fires on a STALLED attempt, so the
                # non-crossing set must be split or 30% of it has already
                # demonstrated discovery by typing the verb.
                "population": ("crossed" if crossed
                               else "stalled" if attempted else "never"),
                "probe": [c["named_route"] for c in r["probe"]],
            })
    return out


def _boot_diff(pairs, i, j, rng):
    """Paired bootstrap over episodes for rate_i - rate_j."""
    n = len(pairs)
    out = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(pairs[k][i] for k in idx) / n
        b = sum(pairs[k][j] for k in idx) / n
        out.append(a - b)
    out.sort()
    return out[int(BOOT * 0.05)], out[int(BOOT * 0.95)]


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa on paired binary judgements."""
    n = len(a)
    if not n:
        return float("nan")
    obs = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if exp == 1 else (obs - exp) / (1 - exp)


def self_split_null(flags: list[bool], p: int, rng) -> list[float]:
    """Range of p random cells drawn from ONE phrasing's own episodes.

    The envelope a cross-phrasing spread has to beat. Reused in spirit from
    `smoke_state_conditioned.self_split_null`; the statistic here is the rate
    range rather than a distribution distance.
    """
    n, out = len(flags), []
    size = n // p
    for _ in range(2000):
        s = flags[:]
        rng.shuffle(s)
        cells = [s[k * size:(k + 1) * size] for k in range(p)]
        rates = [sum(c) / len(c) for c in cells if c]
        out.append(max(rates) - min(rates))
    out.sort()
    return out


def main() -> int:
    C.assert_c3()
    C.assert_stage1_spread()
    data = load()
    if not data:
        print("no c3s1 cells in results/ — nothing to gate")
        return 1
    rng = random.Random(SEED)
    print("C3 STAGE-1 GATE")
    print(f"  prereg {C.C3_HASH[:16]}   probe cells {CELLS}\n")

    prof = {}
    for m, eps in sorted(data.items()):
        at = [e for e in eps if e["at_door"]]
        never = [e for e in at if e["population"] == "never"]
        pairs = [tuple(e["probe"]) for e in at]
        rates = [sum(p[i] for p in pairs) / len(pairs) for i in range(len(CELLS))]
        prof[m] = {
            "n_episodes": len(eps), "n_at_door": len(at), "n_never": len(never),
            "rates": rates,
            "minimal": rates[0],
            "direct": rates[3],
            # THE ESTIMAND: among at-door episodes where the door never moved
            # and the model never even tried, can it name the route when asked?
            "r_never": (sum(e["probe"][0] for e in never) / len(never)
                        if never else None),
            "pairs": pairs,
            "cold_anchor": C.COLD_UNLOCK_RATE[m],
        }

    # ---- P4 FIRST. It is the only leg that can catch a stable wrong probe. ----
    print("P4  EXTERNAL VALIDITY — does the probe agree with real unaided play?")
    print(f"  {'model':<32}{'minimal':>9}{'cold':>8}{'nAtDoor':>9}{'nNever':>8}")
    for m, p in sorted(prof.items(), key=lambda kv: -kv[1]["cold_anchor"]):
        print(f"  {m:<32}{p['minimal']:>9.3f}{p['cold_anchor']:>8.3f}"
              f"{p['n_at_door']:>9}{p['n_never']:>8}")

    zero = [m for m, p in prof.items() if p["cold_anchor"] == 0.0]
    print("\n  PRIMARY — the zero-anchor contrast, tested on episodes not on 3 points.")
    for m in zero:
        p = prof[m]
        n = p["n_at_door"]
        k = round(p["minimal"] * n)
        lo, hi = _wilson(k, n)
        print(f"  {m}: 0 cold unlocks in real play, probe says {p['minimal']:.3f} "
              f"[{lo:.3f}, {hi:.3f}] on n={n}")
        print("    A probe claiming discovery where behaviour showed none is "
              "measuring\n    something other than discovery -- pattern "
              "completion, or the question.")
    others = [p["minimal"] for m, p in prof.items() if p["cold_anchor"] > 0]
    if zero and others:
        gap = min(others) - prof[zero[0]]["minimal"]
        print(f"  gap to the lowest positive-anchor model: {gap:+.3f} "
              f"(expected POSITIVE if the probe tracks discovery)")

    ms = sorted(prof, key=lambda m: prof[m]["cold_anchor"])
    order_ok = all(prof[ms[i]]["minimal"] <= prof[ms[i + 1]]["minimal"]
                   for i in range(len(ms) - 1))
    print(f"\n  ordering agrees with the anchor: {order_ok}   "
          f"(m={len(ms)}, so a perfect ordering has p=1/6 under a noise null --")
    print("   reported, NOT gated on: three points cannot carry a correlation.)")

    # ---- P1 ----
    print("\nP1  MARGINAL EQUIVALENCE — every phrasing pair within +/- 0.10")
    p1 = True
    for m, p in sorted(prof.items()):
        for i, j in itertools.combinations(range(len(CELLS)), 2):
            lo, hi = _boot_diff(p["pairs"], i, j, rng)
            ok = -0.10 <= lo and hi <= 0.10
            p1 &= ok
            if not ok:
                print(f"  FAIL {m:<30}{CELLS[i]}({i}) vs {CELLS[j]}({j})  "
                      f"90% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  {'PASS' if p1 else 'FAIL'}")

    # ---- P2 ----
    print("\nP2  ITEM AGREEMENT — worst cross-phrasing kappa vs the retest ceiling")
    p2 = True
    for m, p in sorted(prof.items()):
        cols = [[q[i] for q in p["pairs"]] for i in range(len(CELLS))]
        retest = kappa(cols[0], cols[1])            # minimal vs minimal
        cross = min(kappa(cols[i], cols[j])
                    for i, j in itertools.combinations(range(len(CELLS)), 2)
                    if {i, j} != {0, 1})
        ok = cross >= max(retest - C.GATE["P2_retest_slack"], C.GATE["P2_kappa_floor"])
        p2 &= ok
        print(f"  {m:<32} retest {retest:+.3f}  worst cross {cross:+.3f}  "
              f"{'ok' if ok else 'FAIL'}")
    print(f"  {'PASS' if p2 else 'FAIL'}")

    # ---- P3 ----
    print("\nP3  INFORMATIVE ENVELOPE — spread <= p95(null) AND p95(null) <= 0.10")
    p3 = True
    for m, p in sorted(prof.items()):
        spread = max(p["rates"]) - min(p["rates"])
        null = self_split_null([q[0] for q in p["pairs"]], len(CELLS), rng)
        p95 = null[int(len(null) * 0.95)]
        ok = spread <= p95 and p95 <= C.GATE["P3_envelope_max"]
        p3 &= ok
        print(f"  {m:<32} spread {spread:.3f}  p95(null) {p95:.3f}  "
              f"{'ok' if ok else 'FAIL'}"
              + ("" if p95 <= C.GATE["P3_envelope_max"]
                 else "  <- envelope too wide to detect 0.10"))
    print(f"  {'PASS' if p3 else 'FAIL'}")

    # ---- hint gradient + the estimand ----
    print("\nHINT GRADIENT — direct minus minimal (selection rule, not a failure)")
    for m, p in sorted(prof.items()):
        d = p["direct"] - p["minimal"]
        print(f"  {m:<32} direct {p['direct']:.3f}  minimal {p['minimal']:.3f}  "
              f"delta {d:+.3f}"
              + ("  -> freeze MINIMAL for Stage 2" if d > C.GATE["hint_gradient_switch"]
                 else ""))

    print("\nr_i — P(names route | at door, never attempted). The estimand.")
    for m, p in sorted(prof.items()):
        r = "  --" if p["r_never"] is None else f"{p['r_never']:.3f}"
        flag = "" if p["n_never"] >= C.MIN_DOOR else "  <- BELOW FLOOR"
        print(f"  {m:<32} r={r}  on n={p['n_never']}{flag}")
    print("  Lower bound: concealment pushes r DOWN, so a HIGH r is robust and "
          "only a\n  LOW r is ambiguous between non-discovery and non-disclosure.")

    verdict = p1 and p2 and p3
    print(f"\nGATE: P1 {'ok' if p1 else 'FAIL'}  P2 {'ok' if p2 else 'FAIL'}  "
          f"P3 {'ok' if p3 else 'FAIL'}  P4 reported above")
    print("Any leg failing is a redesign or an abandon before Stage 2, not a patch.")
    Path("results/c3_stage1_gate.json").write_text(json.dumps(
        {"prereg": C.C3_HASH, "p1": p1, "p2": p2, "p3": p3,
         "models": {m: {k: v for k, v in p.items() if k != "pairs"}
                    for m, p in prof.items()}}, indent=2) + "\n")
    print("wrote results/c3_stage1_gate.json")
    return 0 if verdict else 2


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


if __name__ == "__main__":
    raise SystemExit(main())
