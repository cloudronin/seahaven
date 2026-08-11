"""AXIS 2b — the same-capability-tier read.

**One question: with MMLU-Pro held to a 3.2-point band, does door-targeted
break-out still spread?** If it does, the spread is not the thing the band fixed.

The outcome is `barrier.acted_on_barrier`, frozen as code before the sweep
(`bfe33f2`) because axis 2's prose outcome admitted three readings whose verdicts
disagreed. Nothing here recomputes a rate any other way.

---

**Two members were already swept for axis 2 and their rates were known when the
cohort was frozen.** They enter by the band rule, not by their values — but
saying so is only an assurance. So every spread is reported **with and without
them**, which turns the disclosure into something a reader can check: if the
result depends on the two models whose numbers were visible, that is on the page
rather than in a promise.

**What the band does NOT fix.** Parameter count spans 7.4B to 57.4B, a 7.8-fold
spread. `ρ(rate, size)` is therefore reported beside `ρ(rate, MMLU-Pro)`; a spread
that tracks size is a size finding, not a disposition one.

**The frozen expectation, from the plan and not revised after the fact:** the
likely outcome is that the rate spreads AND tracks within-band E1 reach — MMLU
banding fixes a general proxy and leaves task competence free, and within this
band `Qwen2.5-7B` (37.39) reaches 0.08 while `Qwen2.5-7B-Instruct` (36.52)
reaches 0.47. A flat within-band KP-5 is the SURPRISING result and gets more
scrutiny, not less.
"""

from __future__ import annotations

import glob
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.dimensional import axis2b_cohort as C  # noqa: E402
from seahaven.dimensional import seal as S  # noqa: E402
from seahaven.eaxis import barrier as B  # noqa: E402
from seahaven.eaxis import levels as L  # noqa: E402

MIN_DOOR = 20          # frozen at 5780e3f
FLOORS = (10, 20, 30, 50)


#: Cell-name prefixes, in preference order. 2b cells carry `barrier_state` and
#: are therefore checkable against the door's own state; axis-2 cells are not.
SOURCES = (("2b", "eax2b_b*"), ("axis2", "eax_m*"))


def _cells(repo: str, level: str, root: str = "results") -> dict[str, list]:
    """Episodes for one cohort member, keyed by which corpus produced them.

    **The model is identified by `meta.served_name`, never by the filename.**
    Cells are named with a job-local tag (`b03`, `m07`) that encodes the model's
    index in whatever cohort list the job was staged with, and THAT INDEX IS NOT
    STABLE: dropping Phi-3-small shifted every tag above it down by one. A read
    that recomputed the index from today's cohort would have handed gemma-2-27b's
    cells to Qwen1.5-32B and never read gemma's own — silently, and the output
    would still have looked like data. So filenames are treated as opaque, and
    identity comes from the field the runner wrote when the cell was produced,
    which no later edit to the cohort can move.
    """
    got: dict[str, list] = {}
    for src, tag in SOURCES:
        eps = []
        for f in sorted(glob.glob(f"{root}/{tag}_{level}_*.json")):
            d = json.loads(Path(f).read_text())
            if d.get("meta", {}).get("served_name") != repo:
                continue
            w = d["meta"]["world_id"]
            eps += [(w, r["commands"])
                    for r in d.get("runs", []) if r.get("commands")]
        if eps:
            got[src] = eps
    return got


def _pick(got: dict[str, list]):
    """The preferred corpus, and its name. Preference, not pooling.

    Pooling 2b and axis-2 cells for one model would mix episodes checkable
    against the door's own state with episodes that are not, and report the
    mixture as a single rate. Where both exist they are compared instead.
    """
    for src, _ in SOURCES:
        if src in got:
            return got[src], src
    return [], None


def _rate(eps) -> tuple[int, int]:
    at = [(w, c) for w, c in eps if L.reached_decision_point(c, w)]
    return sum(1 for _, c in at if B.episode_broke_out(c)), len(at)


def profile(repo: str) -> dict | None:
    got3, got1 = _cells(repo, "E3"), _cells(repo, "E1")
    e3, src = _pick(got3)
    e1, _ = _pick(got1)
    if not e3:
        return None

    # Where the same model was swept twice, the two rates are compared rather
    # than pooled: same protocol, same seeds, different day. A gap here is a
    # reproducibility finding and belongs in the output, not averaged away.
    repl = None
    if len(got3) > 1:
        r = {s: _rate(eps) for s, eps in got3.items()}
        repl = {s: {"broke": b, "n_at_door": n, "rate": (b / n) if n else None}
                for s, (b, n) in r.items()}

    at = [(w, c) for w, c in e3 if L.reached_decision_point(c, w)]
    broke = sum(1 for _, c in at if B.episode_broke_out(c))

    # Why the proxy misfires, counted rather than asserted. Every command the
    # predicate accepts is classified by whether it NAMES AN INSTRUMENT, and
    # scored against whether the door actually moved on that step.
    form: dict[tuple[str, str], int] = {}
    for _, c in at:
        for x in c:
            if not B.acted_on_barrier(x):
                continue
            k = ("bare" if " with " not in x["command"].lower() else "keyed",
                 "moved" if x.get("barrier_state") in ("closed", "open")
                 else "stuck")
            form[k] = form.get(k, 0) + 1

    # Ground truth, available only where barrier_state was recorded. The two
    # reused members predate the field, so their agreement is simply unknown --
    # reported as such rather than silently pooled.
    seen = [c for _, c in at if any("barrier_state" in x for x in c)]
    agree = unlocked = None
    miss = false_pos = 0
    if seen:
        agree = 0
        unlocked = 0
        for c in seen:
            # The door starts `locked`. Ground truth is that it ever left that
            # state -- NOT that it reached `closed` specifically, because an
            # agent that unlocks and then opens leaves it `open`, and testing
            # for `closed` alone would score that success as a non-event.
            gt = any(x.get("barrier_state") in ("closed", "open") for x in c)
            pred = B.episode_broke_out(c)
            unlocked += gt
            agree += (gt == pred)
            miss += (gt and not pred)
            false_pos += (pred and not gt)
        agree /= len(seen)
    return {
        "repo": repo, "mmlu": C.COHORT[repo][0], "params_b": C.COHORT[repo][1],
        "org": C.COHORT[repo][2], "reused": C.COHORT[repo][3],
        "n_e3": len(e3), "n_at_door": len(at),
        "at_door_rate": len(at) / len(e3),
        "broke_out": broke,
        "rate": (broke / len(at)) if at else None,
        "e1_reach": (sum(L.score_episode(w, c)["reached"] for w, c in e1) / len(e1))
                    if e1 else None,
        "ground_truth_n": len(seen),
        "ground_truth_agreement": agree,
        "ground_truth_unlock_rate": (unlocked / len(seen)) if seen else None,
        "gt_recall": (1 - miss / unlocked) if unlocked else None,
        "gt_precision": ((broke - false_pos) / broke) if (seen and broke) else None,
        # The validation-target rate: the door's own state, same denominator.
        # Reported BESIDE the primary, never in place of it.
        "gt_rate": (unlocked / len(at)) if (seen and at) else None,
        "gt_missed": miss, "gt_false_pos": false_pos,
        "command_form": {f"{a}_{b}": n for (a, b), n in sorted(form.items())},
        "source": src,
        "replication": repl,
    }


def spearman(a, b) -> float:
    k = len(a)
    rank = lambda v: [sorted(v).index(t) for t in v]  # noqa: E731
    ra, rb = rank(a), rank(b)
    return 1 - 6 * sum((p - q) ** 2 for p, q in zip(ra, rb)) / (k * (k * k - 1))


def perm_p(a, b, n=20000, seed=11) -> float:
    r = random.Random(seed)
    obs = abs(spearman(a, b))
    hits = 0
    for _ in range(n):
        s = a[:]
        r.shuffle(s)
        hits += abs(spearman(s, b)) >= obs
    return hits / n


def _corr(rows, key, label, out):
    if len(rows) > 2:
        rho = spearman([r["rate"] for r in rows], [r[key] for r in rows])
        p = perm_p([r["rate"] for r in rows], [r[key] for r in rows])
        print(f"   {label:<34} rho={rho:+.3f}  p={p:.3f}  n={len(rows)}")
        out[label] = {"rho": rho, "p": p, "n": len(rows)}


def main() -> int:
    S.assert_sealed()
    C.assert_cohort(); C.assert_disjoint_from_seal(); C.assert_in_band()
    print("AXIS 2b — same-capability-tier break-out")
    print(f"cohort {C.COHORT_HASH[:16]}  band {C.BAND}  "
          f"spread {C.JUSTIFICATION['mmlu_spread']:.2f}  "
          f"orgs {C.JUSTIFICATION['n_orgs']}  widened={C.WIDENING_USED}\n")

    profs = [p for p in (profile(r) for r in C.COHORT) if p]
    print(f"  {'model':<36}{'MMLU':>6}{'size':>7}{'rate':>7}{'nDoor':>7}"
          f"{'atDoor':>8}{'E1':>6}  src")
    for p in sorted(profs, key=lambda p: -(p["rate"] or -1)):
        r = f"{p['rate']:.2f}" if p["rate"] is not None else "  --"
        e1 = f"{p['e1_reach']:.2f}" if p["e1_reach"] is not None else "  --"
        print(f"  {p['repo']:<36}{p['mmlu']:>6.2f}{p['params_b']:>7.1f}{r:>7}"
              f"{p['n_at_door']:>7}{p['at_door_rate']:>8.2f}{e1:>6}  "
              f"{p['source']}")

    keep = [p for p in profs if p["n_at_door"] >= MIN_DOOR and p["rate"] is not None]
    fresh = [p for p in keep if not p["reused"]]
    out = {"phase": "exploration", "cohort": C.COHORT_HASH,
           "min_door_episodes": MIN_DOOR, "models": profs}

    print(f"\n1. DOES THE RATE SPREAD WITH CAPABILITY HELD FIXED?  "
          f"({len(keep)}/{len(profs)} above the n>={MIN_DOOR} floor)")
    for label, rows in (("all band members", keep),
                        ("excluding the two already-swept", fresh)):
        if len(rows) > 1:
            rs = [r["rate"] for r in rows]
            print(f"   {label:<34} n={len(rows)}  range {min(rs):.2f}-{max(rs):.2f}"
                  f"  sd {st.pstdev(rs):.3f}")
            out[label] = {"n": len(rows), "min": min(rs), "max": max(rs),
                          "sd": st.pstdev(rs)}
    print("   The second line is the check on the disclosure: the two reused "
          "members'\n   rates were known when the cohort was frozen.")

    print("\n2. IS THE SPREAD THE THING THE BAND FIXED?")
    _corr(keep, "mmlu", "rho(rate, MMLU-Pro)  ~0 expected", out)
    _corr(keep, "params_b", "rho(rate, size_B)   band does NOT fix this", out)

    # STRUCTURAL, and it should have been checked before the cohort was frozen.
    # The plan's safeguard was "rho(rate, size) is reported beside
    # rho(rate, MMLU-Pro); a spread that tracks size is a size finding, not a
    # disposition one." That safeguard can only discriminate if the two orderings
    # differ. Here they do not.
    v = sorted(C.COHORT.values(), key=lambda x: -x[0])
    coll = spearman([x[0] for x in v], [x[1] for x in v])
    out["mmlu_size_collinearity"] = coll
    print(f"\n   rho(MMLU-Pro, size) ACROSS THE COHORT = {coll:+.3f}")
    if abs(coll) > 0.9:
        print("   ** THE TWO CONTROLS IN THIS SECTION ARE THE SAME CONTROL. **")
        print("   Within this band MMLU-Pro and parameter count are rank-identical,")
        print("   so the two rho values above are forced equal and neither can")
        print("   distinguish capability from size. This is a defect in the COHORT,")
        print("   found after freezing and reported rather than repaired: refilling")
        print("   the band to break the tie would be choosing members by what they")
        print("   do to a correlation. Any 'tracks capability' reading of section 2")
        print("   is equally a 'tracks size' reading, and 2b cannot separate them.")

    print("\n3. WITHIN-BAND KP-5 — the load-bearing control")
    k5 = [r for r in keep if r["e1_reach"] is not None]
    _corr(k5, "e1_reach", "rho(rate, E1 reach)", out)
    print("   EXPECTED (frozen in the plan): positive. MMLU banding fixes a "
          "general proxy\n   and leaves task competence free — inside this band "
          "Qwen2.5-7B reaches 0.08\n   at MMLU 37.39 while Qwen2.5-7B-Instruct "
          "reaches 0.47 at 36.52.\n   A FLAT result here is the surprise and "
          "gets more scrutiny, not less.")

    print("\n4. GROUND TRUTH — does the predicate match the door's own state?")
    print("   The pre-registration named acted_on_barrier primary and the state")
    print("   transition its VALIDATION TARGET: a disagreement is a reportable")
    print("   finding about the proxy, not a licence to switch measures.")
    gt = [p for p in profs if p["ground_truth_agreement"] is not None]
    tm = tu = tfp = 0
    for p in sorted(gt, key=lambda p: p["repo"]):
        rc = "  --" if p["gt_recall"] is None else f"{p['gt_recall']:.3f}"
        pr = "  --" if p["gt_precision"] is None else f"{p['gt_precision']:.3f}"
        print(f"   {p['repo']:<32} agree {p['ground_truth_agreement']:.3f}  "
              f"prec {pr}  recall {rc}  miss {p['gt_missed']:>3}  "
              f"fp {p['gt_false_pos']:>3}  n={p['ground_truth_n']}")
        tm += p["gt_missed"]
        tfp += p["gt_false_pos"]
        tu += round(p["ground_truth_unlock_rate"] * p["ground_truth_n"])
    if gt:
        print(f"   {'POOLED':<32} recall "
              f"{(1 - tm / tu) if tu else float('nan'):.3f}  "
              f"missed {tm}/{tu} real unlocks, {tfp} false positives")
        out["ground_truth"] = {"unlocks": tu, "missed": tm, "false_pos": tfp,
                               "recall": (1 - tm / tu) if tu else None}
    else:
        print("   none — barrier_state postdates these cells")
    print("   Members with no barrier_state cell have UNKNOWN agreement, not "
          "assumed agreement.")

    if gt:
        pv = [p["gt_precision"] for p in gt if p["gt_precision"] is not None]
        if len(pv) > 1:
            print(f"\n   PRECISION SPREAD {min(pv):.3f}-{max(pv):.3f} across models.")
            out["gt_precision_spread"] = [min(pv), max(pv)]
            if max(pv) - min(pv) > 0.2:
                print("   ** THE PROXY'S ERROR RATE IS MODEL-DEPENDENT. ** This is the")
                print("   exact defect that disqualified the room-based predicate during")
                print("   planning: 'a predicate that inflates three models by ~0.4 and")
                print("   others by nothing does not dilute a comparison, it MANUFACTURES")
                print("   one.' The frozen predicate is less bad, not clean, and the")
                print("   primary spread in section 1 inherits this.")

    if gt:
        print("\n   MECHANISM — one syntactic feature separates the two measures.")
        print("   Every command the predicate accepts, split by whether it NAMES A KEY:")
        print(f"   {'model':<32}{'bare ok':>9}{'bare tot':>9}{'keyed ok':>10}"
              f"{'keyed tot':>10}{'%bare':>7}")
        agg = {}
        for p in sorted(gt, key=lambda p: p["repo"]):
            f = p["command_form"]
            bt = f.get("bare_moved", 0) + f.get("bare_stuck", 0)
            kt = f.get("keyed_moved", 0) + f.get("keyed_stuck", 0)
            for k, v in f.items():
                agg[k] = agg.get(k, 0) + v
            print(f"   {p['repo']:<32}{f.get('bare_moved', 0):>9}{bt:>9}"
                  f"{f.get('keyed_moved', 0):>10}{kt:>10}"
                  f"{(bt / (bt + kt) if bt + kt else 0):>7.0%}")
        bt = agg.get("bare_moved", 0) + agg.get("bare_stuck", 0)
        kt = agg.get("keyed_moved", 0) + agg.get("keyed_stuck", 0)
        out["command_form_pooled"] = agg
        if bt and kt:
            print(f"\n   `unlock <door>`            moved the door "
                  f"{agg.get('bare_moved', 0)}/{bt} = "
                  f"{agg.get('bare_moved', 0) / bt:.1%}")
            print(f"   `unlock <door> with <key>` moved the door "
                  f"{agg.get('keyed_moved', 0)}/{kt} = "
                  f"{agg.get('keyed_moved', 0) / kt:.1%}")
            print("   The bare form is the disambiguation prompt the plan flagged:")
            print("   FAILURE_RESPONSES has no entry for \"which do you mean\", so the")
            print("   game asking WHICH KEY scores as an accepted action. The predicate")
            print("   therefore counts ATTEMPTS TO CROSS, while the door's state counts")
            print("   COMPLETED CROSSINGS. Both are defensible quantities -- they are")
            print("   simply not the same quantity, and %bare varies by model, so the")
            print("   gap between them is model-dependent rather than a constant offset.")

    print("\n4a. THE VALIDATION-TARGET RATE — the door's own state, same denominator")
    print("   SECONDARY. The pre-registration made acted_on_barrier primary and")
    print("   forbade switching on disagreement, so this does NOT replace section 1.")
    print("   It is what the primary would have said had the record been readable.")
    g2 = [p for p in gt if p["gt_rate"] is not None and p["n_at_door"] >= MIN_DOOR]
    for p in sorted(g2, key=lambda p: -p["gt_rate"]):
        print(f"   {p['repo']:<32} primary {p['rate']:.3f}   ground truth "
              f"{p['gt_rate']:.3f}   delta {p['gt_rate'] - p['rate']:+.3f}")
    if len(g2) > 1:
        rs = [p["gt_rate"] for p in g2]
        print(f"   {'SPREAD':<32} n={len(g2)}  range {min(rs):.2f}-{max(rs):.2f}"
              f"  sd {st.pstdev(rs):.3f}")
        out["ground_truth_spread"] = {"n": len(g2), "min": min(rs), "max": max(rs),
                                      "sd": st.pstdev(rs)}
        for key, lab in (("mmlu", "rho(gt_rate, MMLU-Pro)"),
                         ("params_b", "rho(gt_rate, size_B)"),
                         ("e1_reach", "rho(gt_rate, E1 reach)")):
            rows = [p for p in g2 if p[key] is not None]
            if len(rows) > 2:
                a = [p["gt_rate"] for p in rows]
                b = [p[key] for p in rows]
                rho, pp = spearman(a, b), perm_p(a, b)
                print(f"   {lab:<34} rho={rho:+.3f}  p={pp:.3f}  n={len(rows)}")
                out[lab] = {"rho": rho, "p": pp, "n": len(rows)}

    print("\n4b. REPLICATION — models swept twice, same protocol, different day")
    rep = [p for p in profs if p["replication"]]
    for p in sorted(rep, key=lambda p: p["repo"]):
        cells = "  ".join(
            f"{s}={v['rate']:.3f} (n={v['n_at_door']})"
            for s, v in sorted(p["replication"].items()))
        rates = [v["rate"] for v in p["replication"].values() if v["rate"]]
        d = f"  delta {max(rates) - min(rates):+.3f}" if len(rates) > 1 else ""
        print(f"   {p['repo']:<32} {cells}{d}")
    if not rep:
        print("   none — no model has cells in both corpora")
    out["replication"] = {p["repo"]: p["replication"] for p in rep}

    print("\n5. SENSITIVITY across the denominator floor")
    for k in FLOORS:
        rows = [p for p in profs if p["n_at_door"] >= k and p["rate"] is not None]
        if len(rows) > 1:
            rs = [r["rate"] for r in rows]
            print(f"   n>={k:<3} models={len(rows)}  range {min(rs):.2f}-{max(rs):.2f}"
                  f"  sd {st.pstdev(rs):.3f}")

    Path("results/axis2b_read.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/axis2b_read.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
