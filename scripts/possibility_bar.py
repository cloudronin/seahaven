"""The possibility bar — read each surveyed anchor against the four criteria.

Written against `docs/research-log.md`'s PHASE SWITCH entry, which fixed the
criteria **before** any rung was fit:

1. Non-trivial split — at least two models on the minority side.
2. Cross-world — memberships identical on v0 and v2.
3. Seed-stable — no model's side flips under episode bootstrap at 90 percent.
4. Phrasing-robust — memberships unchanged under leave-one-phrasing-out.

Only a candidate clearing all four earns one of the two reserve looks.

**Two things about how the criteria interact, predicted before reading the
numbers.** Criterion 4 is one-directional: leave-one-out takes the minimum over
four phrasings, and min-over-4 >= min-over-5, so margins can only rise and flags
can only disappear. And a rising anchor buys criterion 1 by moving the line up
into the pack, but every model it newly flags is by construction *near* the
line, which is exactly where 2 and 3 fail. Expect the binding constraint to
migrate from 1 to 2 and 3 as the anchor climbs.

**The bootstrap resamples the anchor too.** Holding it fixed while resampling
models would treat the anchor as known exactly — the assumption the SE target
exists to deny, and the one that would make criterion 3 look far stronger than
it is.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _vp_data import episodes_for  # noqa: E402

from seahaven.fidelity.flag import FLAG_BOUNDARY, PHRASING_IDS  # noqa: E402

WORLDS = ("world_v0", "world_v2")
N_BOOT = 2000
CONFIDENCE = 0.90


def _adherence(eps) -> float:
    bad = sum(e[0] for e in eps)
    n = sum(e[1] for e in eps)
    return 100.0 * (1 - bad / n) if n else float("nan")


def _resample(eps, rng):
    return [eps[rng.randrange(len(eps))] for _ in eps]


def flagged_set(models: dict, world: str) -> frozenset:
    return frozenset(lab for lab, m in models.items()
                     for p in m["per_world"] if p["world"] == world and p["flagged"])


# --- criterion 1 ------------------------------------------------------------

def criterion_1(models: dict) -> tuple[bool, str]:
    """At least two models on the minority side of the FLAG/PASS split."""
    labels = {lab: m["label"] for lab, m in models.items()}
    flag = [l for l, v in labels.items() if v == "FLAG"]
    npass = [l for l, v in labels.items() if v == "PASS"]
    unstable = [l for l, v in labels.items() if v == "UNSTABLE"]
    minority = flag if len(flag) <= len(npass) else npass
    ok = len(minority) >= 2
    note = (f"FLAG={len(flag)} PASS={len(npass)} UNSTABLE={len(unstable)}; "
            f"minority side has {len(minority)} ({sorted(minority)})")
    return ok, note


# --- criterion 2 ------------------------------------------------------------

def criterion_2(models: dict) -> tuple[bool, str]:
    """Memberships identical on v0 and v2 — equivalently, nothing UNSTABLE."""
    a, b = (flagged_set(models, w) for w in WORLDS)
    ok = a == b
    note = (f"{WORLDS[0]} flags {sorted(a) or 'none'}; "
            f"{WORLDS[1]} flags {sorted(b) or 'none'}"
            + ("" if ok else f"; differ on {sorted(a ^ b)}"))
    return ok, note


# --- criterion 3 ------------------------------------------------------------

def criterion_3(models: dict, anchors: dict, model_eps, seed: int = 4111):
    """No model's side flips under a joint episode bootstrap at 90 percent.

    A side is stable when the margin's 90 percent interval excludes zero. Both
    the model's episodes and the anchor's are resampled in the same draw, so the
    interval carries the anchor's own uncertainty rather than pretending it away.
    """
    rng = random.Random(seed)
    labs = sorted(models)
    margins = {(lab, w): [] for lab in labs for w in WORLDS}

    anchor_eps = {w: [tuple(e) for e in anchors[w]["episode_counts"]] for w in WORLDS}

    for _ in range(N_BOOT):
        for w in WORLDS:
            a = _adherence(_resample(anchor_eps[w], rng))
            for lab in labs:
                worst = min(_adherence(_resample(model_eps[(lab, p, w)], rng))
                            for p in PHRASING_IDS)
                margins[(lab, w)].append(worst - a)

    lo_q, hi_q = (1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2
    detail, ok = {}, True
    for lab in labs:
        for w in WORLDS:
            d = sorted(margins[(lab, w)])
            lo, hi = d[int(lo_q * len(d))], d[int(hi_q * len(d))]
            stable = (lo > FLAG_BOUNDARY) or (hi <= FLAG_BOUNDARY)
            share = sum(1 for x in d if (x <= FLAG_BOUNDARY)) / len(d)
            detail[f"{lab}|{w}"] = {"ci90": [lo, hi], "stable": stable,
                                    "flag_share": share}
            ok &= stable
    unstable = [k for k, v in detail.items() if not v["stable"]]
    note = ("all sides stable" if ok
            else f"{len(unstable)} cell(s) straddle the boundary: {unstable}")
    return ok, note, detail


# --- criterion 4 ------------------------------------------------------------

def criterion_4(models: dict, anchors: dict, per_phrasing) -> tuple[bool, str, dict]:
    """Memberships unchanged when each phrasing is dropped in turn.

    Margins can only rise under leave-one-out, so a flag can only disappear. A
    model that flags on the strength of a single phrasing is exactly what this
    removes, and that is the criterion working rather than failing.
    """
    base = {w: flagged_set(models, w) for w in WORLDS}
    changed, detail = [], {}
    for dropped in PHRASING_IDS:
        kept = [p for p in PHRASING_IDS if p != dropped]
        for w in WORLDS:
            got = frozenset(
                lab for lab in models
                if min(per_phrasing[f"{lab}|{w}"][p] for p in kept)
                - anchors[w]["adherence"] <= FLAG_BOUNDARY)
            detail[f"drop_{dropped}|{w}"] = sorted(got)
            if got != base[w]:
                changed.append(f"drop {dropped} on {w}: "
                               f"{sorted(base[w])} -> {sorted(got)}")
    ok = not changed
    return ok, ("membership survives every leave-one-out" if ok
                else "; ".join(changed)), detail


def main() -> int:
    survey = json.loads(Path("results/anchor_survey.json").read_text())
    if survey.get("phase") != "exploration":
        raise SystemExit("survey file is not labelled exploration — refusing")

    model_eps = episodes_for()
    per_phrasing = survey["model_adherence"]

    print("POSSIBILITY BAR — criteria fixed in the PHASE SWITCH entry, before "
          "any rung was fit")
    print("PHASE: exploration. Clearing the bar earns one reserve look, "
          "not a claim.\n")

    out = {"phase": "exploration", "confidence": CONFIDENCE, "n_boot": N_BOOT,
           "rungs": {}}

    for rung, r in survey["rungs"].items():
        models, anchors = r["models"], r["anchors"]
        loc = "  ".join(f"{w} {anchors[w]['adherence']:.2f}" for w in WORLDS)
        print(f"{rung} — {r['family']}   anchor: {loc}")

        c1, n1 = criterion_1(models)
        c2, n2 = criterion_2(models)
        c3, n3, d3 = criterion_3(models, anchors, model_eps)
        c4, n4, d4 = criterion_4(models, anchors, per_phrasing)

        for i, (ok, note) in enumerate([(c1, n1), (c2, n2), (c3, n3), (c4, n4)], 1):
            print(f"  {i}. {'PASS' if ok else 'FAIL'}  {note}")

        cleared = all([c1, c2, c3, c4])
        print(f"  => {rung} {'CLEARS the bar' if cleared else 'does NOT clear'}\n")

        out["rungs"][rung] = {
            "anchor": {w: anchors[w]["adherence"] for w in WORLDS},
            "labels": {lab: m["label"] for lab, m in models.items()},
            "criteria": {"non_trivial_split": {"pass": c1, "note": n1},
                         "cross_world": {"pass": c2, "note": n2},
                         "seed_stable": {"pass": c3, "note": n3, "detail": d3},
                         "phrasing_robust": {"pass": c4, "note": n4, "detail": d4}},
            "clears_bar": cleared,
        }

    cleared = [k for k, v in out["rungs"].items() if v["clears_bar"]]
    print(f"CLEARED: {cleared or 'none'}")
    if not cleared:
        print("  No reserve look is earned. The reserve stays unburned, which is\n"
              "  the point of having pinned it before the survey ran.")

    Path("results/possibility_bar.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/possibility_bar.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
