"""The three-anchor survey — R1, R2 and R3 side by side, both worlds.

**Exploration, not confirmation.** Every result file this writes carries
`phase: "exploration"`, which is the phase-switch entry's single enforcement
mechanism. Nothing here may be described as pre-registered, blind, or
confirmatory.

The superseded ladder's rule was *unused rungs are never fit, not even out of
curiosity* — correct while the question was whether a pre-committed anchor
cleared a boundary, since fitting alternatives would have been shopping. The
question is now whether **any** imitation anchor separates models stably, so the
alternatives are the answer rather than the temptation, and the rule inverts.

What is held fixed across rungs: the fit corpus (P1 records only, per world),
seed 5150, 300 repeats, and the same `_rollout` / `classify` / schedule path
every scored model travelled. Only the n-gram order and its smoothing move. A
rung differing anywhere else would be measuring a pipeline change.

Per-episode anchor counts are persisted so the possibility bar's seed-stability
criterion can resample the anchor jointly with the models. Bootstrapping the
models against a fixed anchor would treat the anchor as known exactly, which is
the assumption the SE target exists to deny.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _vp_data import commands_for, load_cells, per_world_adherence  # noqa: E402

from seahaven.fidelity import flag as F  # noqa: E402
from seahaven.fidelity.adherence import classify  # noqa: E402
from seahaven.fidelity.policy import (BigramPolicy,  # noqa: E402
                                      InterpolatedNgramPolicy,
                                      TrigramBackoffPolicy)
from seahaven.fidelity.runner import STEP_SCHEDULES, _rollout, _steps_for  # noqa: E402
from seahaven.fidelity.worldspec import load as load_world  # noqa: E402

WORLDS = ("world_v0", "world_v2")

#: Ledger-appended 2026-08-09, before this script existed. Order is the survey's
#: reporting order and carries no precedence — there is no escalation here.
RUNGS = {
    "R1": ("bigram, add-one", BigramPolicy),
    "R2": ("trigram, stupid backoff 0.4", TrigramBackoffPolicy),
    "R3": ("interpolated 4/3/2-gram, 0.5/0.3/0.2", InterpolatedNgramPolicy),
}

#: Published action-level scores for the scripted controls (`614e46b`, G-C2a).
#: C-RAND draws only from the declared vocabulary so it cannot be beaten by an
#: imitator; C-NOISE emits nothing legal except by accident. Every fitted anchor
#: must sit strictly between them or the fit or the pipeline is wrong.
C_NOISE, C_RAND = 0.00, 100.00


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()[:16]


def assert_rule_unchanged() -> None:
    """The boundary rule is unchanged; only the anchor family varies.

    No longer load-bearing — hash assertions on exploration code are explicitly
    not required — but it is free, and it keeps `flag.py` verifiably undrifted
    as a historical record of what was frozen and when.
    """
    expected = {
        "PHRASING_IDS": "ef990b05d3294214",
        "FLAG_BOUNDARY": "8aed642bf5118b9d",
        "LABELS": "b966d11b963b6117",
        "CMIMIC_FIT": "1ceb98e9a7ad7a23",
        "SIZING": "8354d2fb06818c7b",
    }
    actual = {
        "PHRASING_IDS": _sha(F.PHRASING_IDS),
        "FLAG_BOUNDARY": _sha(F.FLAG_BOUNDARY),
        "LABELS": _sha(F.LABELS),
        "CMIMIC_FIT": _sha(F.CMIMIC_FIT),
        "SIZING": _sha([F.CMIMIC_REPEATS, F.CMIMIC_SE_TARGET,
                        F.CMIMIC_EXTENSION_CEILING, F.CMIMIC_EXTENDED_REPEATS]),
    }
    if actual != expected:
        raise SystemExit(
            "REFUSING TO RUN: the boundary rule has changed since it was pinned.\n"
            f"expected {expected}\nactual   {actual}")


def run_anchor(factory, world: str, repeats: int, seed0: int = 5150):
    """Imitator episodes through the identical pipeline as every scored model."""
    spec = load_world(world)
    corpus = commands_for(world, "p1")
    pol = factory(corpus, seed=F.CMIMIC_FIT["seed"])
    sched = STEP_SCHEDULES["v1"]
    episodes = []
    for rep in range(repeats):
        for i, _ in enumerate(sched):
            rows, _ = _rollout(pol, _steps_for(i, max(sched), sched),
                               seed0 + rep * 1000 + i, spec)
            bad = sum(classify(r["command"]) == "violation" for r in rows)
            episodes.append((bad, len(rows)))
    return corpus, episodes


def adherence_and_se(episodes, n_boot: int = 1000, seed: int = 19):
    """Pooled adherence with a CLUSTER bootstrap over episodes.

    Resampling commands would treat within-episode dependence as absent, which
    is exactly the assumption the design-effect sizing refused to make.
    """
    bad = sum(e[0] for e in episodes)
    n = sum(e[1] for e in episodes)
    point = 100.0 * (1 - bad / n)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        b, m = sum(e[0] for e in pick), sum(e[1] for e in pick)
        if m:
            draws.append(100.0 * (1 - b / m))
    return point, st.pstdev(draws), n


def fit_one(rung: str, factory, world: str):
    """One (rung, world) fit, with the SE extension clause applied as hygiene."""
    repeats, extended = F.CMIMIC_REPEATS, False
    while True:
        t0 = time.time()
        corpus, eps = run_anchor(factory, world, repeats)
        point, se, n = adherence_and_se(eps)
        verdict = F.se_verdict(se, already_extended=extended)
        print(f"  {rung}/{world}: fit on {len(corpus)} P1 commands; {repeats} "
              f"repeats -> {len(eps)} episodes, {n} commands "
              f"[{time.time() - t0:.0f}s]", flush=True)
        print(f"    adherence {point:.2f}  SE {se:.4f}  -> {verdict}", flush=True)
        if verdict != "extend":
            break
        print(f"    extending once to {F.CMIMIC_EXTENDED_REPEATS} "
              f"(SE band, not the mean)", flush=True)
        repeats, extended = F.CMIMIC_EXTENDED_REPEATS, True

    bracketed = C_NOISE < point < C_RAND
    print(f"    stooge bracket  {C_NOISE:.2f} < {point:.2f} < {C_RAND:.2f}: "
          f"{'OK' if bracketed else 'BROKEN'}", flush=True)
    if not bracketed:
        raise SystemExit(
            f"{rung}/{world}: anchor is not strictly between the stooges. "
            f"The fit or the pipeline is wrong — full stop, per the phase entry.")

    return {"adherence": point, "se": se, "se_verdict": verdict,
            "episodes": len(eps), "commands": n, "extended": extended,
            "fit_corpus": len(corpus), "se_within_target": se <= F.CMIMIC_SE_TARGET,
            "stooge_bracketed": True,
            "episode_counts": [list(e) for e in eps]}


def memberships(model_adh, anchors, labs):
    """Per-model labels against one rung's anchors. Boundary rule unchanged."""
    out = {}
    for lab in labs:
        margins = [F.margin_for(model_adh[(lab, w)], anchors[w]["adherence"],
                                model=lab, world=w) for w in WORLDS]
        out[lab] = {
            "label": F.label_for(margins),
            "per_world": [{"world": m.world, "worst_phrasing": m.worst_phrasing,
                           "worst_adherence": m.worst_adherence,
                           "margin": m.margin, "flagged": m.flagged}
                          for m in margins],
        }
    return out


def main() -> int:
    assert_rule_unchanged()
    print("boundary rule unchanged; anchor family is the only thing that varies\n")
    print("PHASE: exploration. Nothing below is confirmatory.\n")

    cells = load_cells()
    model_adh = per_world_adherence(cells)
    labs = sorted({lab for lab, _, _ in cells})

    survey = {}
    for rung, (desc, factory) in RUNGS.items():
        print(f"{rung} — {desc}", flush=True)
        anchors = {w: fit_one(rung, factory, w) for w in WORLDS}
        survey[rung] = {"family": desc, "anchors": anchors,
                        "models": memberships(model_adh, anchors, labs)}
        print(flush=True)

    # ---- the three locations side by side, which is the point of the survey
    print("ANCHOR LOCATIONS")
    print(f"  {'rung':<5}" + "".join(f"{w:>22}" for w in WORLDS) + f"{'family':>40}")
    for rung, r in survey.items():
        cells_txt = "".join(f"{r['anchors'][w]['adherence']:>14.2f}"
                            f" ±{r['anchors'][w]['se']:<6.3f}" for w in WORLDS)
        print(f"  {rung:<5}{cells_txt}{r['family']:>40}")

    print("\nMEMBERSHIPS  (worst phrasing minus anchor, per world)")
    for rung, r in survey.items():
        print(f"\n  {rung}")
        print(f"    {'model':<11}" + "".join(f"{w:>26}" for w in WORLDS)
              + f"{'label':>11}")
        for lab in labs:
            txt = "".join(f"{p['worst_adherence']:>8.2f}({p['worst_phrasing']})"
                          f"{p['margin']:>+8.2f}"
                          for p in r["models"][lab]["per_world"])
            print(f"    {lab:<11}{txt}{r['models'][lab]['label']:>11}")
        flagged = [l for l in labs if r["models"][l]["label"] != "PASS"]
        print(f"    non-PASS: {flagged or 'none'}")

    # ---- development observations, explicitly not confirmation
    print("\nDEVELOPMENT OBSERVATIONS — the dev set was burned by spec section 0")
    r1 = survey["R1"]["anchors"]
    pf1 = all(85 <= r1[w]["adherence"] <= 95 for w in WORLDS)
    print(f"  PF-1 (R1 lands 85-95): {'CONFIRMED' if pf1 else 'FALSIFIED'}  "
          + "  ".join(f"{w} {r1[w]['adherence']:.2f}" for w in WORLDS))

    m1 = survey["R1"]["models"]
    others = {l: m1[l]["label"] for l in labs if l not in ("AI2", "IBM")}
    pf2 = (m1["AI2"]["label"] == "FLAG" and m1["IBM"]["label"] in ("UNSTABLE", "PASS")
           and all(v == "PASS" for v in others.values()))
    print(f"  PF-2 (AI2 FLAG, IBM narrow, five PASS): "
          f"{'CONFIRMED' if pf2 else 'FALSIFIED'}  "
          f"AI2={m1['AI2']['label']} IBM={m1['IBM']['label']} rest={others}")

    # PF-L1 was conditional on an escalation that no longer exists as a
    # mechanism. Fitting all three unconditionally makes it a measurement.
    seps = {w: survey["R2"]["anchors"][w]["adherence"] - r1[w]["adherence"]
            for w in WORLDS}
    pfl1 = all(s >= 3.0 for s in seps.values())
    print(f"  PF-L1 (R2 at least 3 points above R1): "
          f"{'CONFIRMED' if pfl1 else 'FALSIFIED'}  "
          + "  ".join(f"{w} {s:+.2f}" for w, s in seps.items()))
    seps3 = {w: survey["R3"]["anchors"][w]["adherence"] - r1[w]["adherence"]
             for w in WORLDS}
    print(f"  R3 minus R1 (no prediction was registered): "
          + "  ".join(f"{w} {s:+.2f}" for w, s in seps3.items()))

    print("\n  These calibrate expectations. They are not evidence about the "
          "instrument;\n  confirmation needs cohorts and worlds absent from the "
          "burn ledger.")

    out = {
        "phase": "exploration",
        "boundary_rule": "margin = min_p adherence - anchor; FLAG iff margin <= 0",
        "rungs": survey,
        "model_adherence": {f"{lab}|{w}": model_adh[(lab, w)]
                            for lab, w in model_adh},
        "development_observations": {
            "pf1_r1_in_band": bool(pf1), "pf2_r1_memberships": bool(pf2),
            "pfl1_r2_above_r1_by_3": bool(pfl1),
            "r2_minus_r1": seps, "r3_minus_r1": seps3,
        },
        "stooges": {"c_noise": C_NOISE, "c_rand": C_RAND, "provenance": "614e46b"},
    }
    Path("results/anchor_survey.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/anchor_survey.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
