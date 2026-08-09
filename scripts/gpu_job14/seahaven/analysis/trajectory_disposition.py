"""Read disposition off a trajectory, and compare it to what the battery says.

**The question.** In the closed-loop run, distinct command sequences stayed 8/8
and trajectory spread rose while probe-fingerprint spread halved. The agents kept
*doing* different things while their measured dispositions converged. Two
readings point opposite ways:

    the battery misses what differs      → instrument problem, fixable
    disposition and behaviour decouple   → a finding about the setup

The test: build a disposition vector from what the agent actually **did**, on the
same axes the battery probes, and ask which representation separates characters
we know to be different.

**Construction.** Each axis is a rate in [0, 1] read off ground truth, not off
the agent's self-report — inventory comes from the world's fact set, so carrying
and dropping are measured rather than claimed. Rates are chosen so that the two
poles of each probe axis map to high and low values:

| axis | trajectory measure | high means |
|---|---|---|
| risk | share of moves into a room not yet seen | tries the uncertain thing |
| social | drops per (drops + takes) | gives things away |
| curio | distinct rooms visited / rooms reachable | goes looking |
| commit | mean run-length of consecutive same-verb actions | finishes what it starts |
| pace | share of steps that change the room | acts rather than deliberates |
| report | share of commands the parser rejected | acts without checking |

`report` is the weakest analogy and is reported but excluded from the headline
comparison; the others are direct.

**Comparison.** Both representations are scored the same way — mean pairwise
distance within a character against between characters — so the ratio is
comparable even though the underlying units differ. A representation that
separates characters has ratio well above 1.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

# world_v0 has six rooms; used to normalise the curiosity rate.
N_ROOMS = 6

AXES = ("risk", "social", "curio", "commit", "pace", "report")
HEADLINE_AXES = ("risk", "social", "curio", "commit", "pace")


def disposition_from_trajectory(ep: list[dict]) -> dict[str, float]:
    """Six rates in [0,1], read from what happened rather than what was said."""
    if not ep:
        return {a: 0.0 for a in AXES}

    seen: set[str] = set()
    novel_moves = moves = 0
    for r in ep:
        if r["room_before"]:
            seen.add(r["room_before"])
        if r["verb"] == "go":
            moves += 1
            if r["room_after"] and r["room_after"] not in seen:
                novel_moves += 1
        if r["room_after"]:
            seen.add(r["room_after"])

    verbs = Counter(r["verb"] for r in ep)
    drops, takes = verbs.get("drop", 0), verbs.get("take", 0)

    # Mean length of consecutive same-verb stretches: how long the agent stays
    # with a line of activity before switching.
    runs, cur = [], 1
    for a, b in zip(ep, ep[1:]):
        if a["verb"] == b["verb"]:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    mean_run = sum(runs) / len(runs)

    changed_room = sum(1 for r in ep if r["room_before"] != r["room_after"])

    return {
        "risk":   novel_moves / moves if moves else 0.0,
        "social": drops / (drops + takes) if (drops + takes) else 0.5,
        "curio":  len(seen) / N_ROOMS,
        # Normalised against the episode so it lands in [0,1] like the others.
        "commit": min(1.0, (mean_run - 1) / max(1, len(ep) / 4)),
        "pace":   changed_room / len(ep),
        "report": sum(1 for r in ep if r["rejected"]) / len(ep),
    }


def dist_disposition(a: dict, b: dict, axes=HEADLINE_AXES) -> float:
    return sum((a[k] - b[k]) ** 2 for k in axes) / len(axes)


def dist_fingerprint(a: dict, b: dict) -> float:
    ks = sorted(set(a) & set(b))
    return sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k])) for k in ks) / len(ks)


def between_within(items, labels, dist) -> dict:
    within, between = [], []
    for (i, x), (j, y) in itertools.combinations(list(enumerate(items)), 2):
        (within if labels[i] == labels[j] else between).append(dist(x, y))
    mw = sum(within) / len(within) if within else 0.0
    mb = sum(between) / len(between) if between else 0.0

    # Zero within-character distance means PERFECT separation, not none. Twice in
    # this project a `ratio = None -> None or 0` path printed a confident label
    # that inverted the result, once declaring a strong effect dead. Encode the
    # degenerate case explicitly rather than letting a falsy value decide.
    if mw > 1e-9:
        ratio, kind = mb / mw, "finite"
    elif mb > 1e-9:
        ratio, kind = float("inf"), "perfect_separation_zero_within"
    else:
        ratio, kind = None, "degenerate_both_zero"

    return {
        "within": round(mw, 6), "between": round(mb, 6),
        "ratio": (round(ratio, 3) if ratio not in (None, float("inf")) else ratio),
        "ratio_kind": kind,
        "n_within": len(within), "n_between": len(between),
    }


def analyse(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    labels = d["labels"]
    trajs = d["trajectories"]
    fps = d["probe_fingerprints"]

    disp = [disposition_from_trajectory(ep) for ep in trajs]
    traj_bw = between_within(disp, labels, dist_disposition)
    probe_bw = between_within(fps, labels, dist_fingerprint)

    # Per-axis means by character: does the trajectory measure move in the
    # direction the assigned character implies? This is the check that the axes
    # mean what they claim, independent of any separation statistic.
    by_char: dict[str, dict[str, float]] = {}
    for name in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == name]
        by_char[name] = {a: round(sum(disp[i][a] for i in idx) / len(idx), 3)
                         for a in AXES}

    def separates(bw) -> bool | None:
        """True / False / None-for-undecidable. Never coerce None to 0."""
        if bw["ratio_kind"] == "perfect_separation_zero_within":
            return True
        if bw["ratio_kind"] == "degenerate_both_zero":
            return None
        return bw["ratio"] > 1.3

    t_sep, p_sep = separates(traj_bw), separates(probe_bw)
    if t_sep is None or p_sep is None:
        verdict = "UNDECIDABLE"
    elif t_sep and not p_sep:
        verdict = "BATTERY_IS_THE_WEAK_LINK"
    elif t_sep and p_sep:
        verdict = "BOTH_SEPARATE"
    elif not t_sep and not p_sep:
        verdict = "NEITHER_SEPARATES"
    else:
        verdict = "PROBE_ONLY"

    return {
        "n_runs": len(labels),
        "characters": sorted(set(labels)),
        "trajectory_disposition": traj_bw,
        "probe_fingerprint": probe_bw,
        "per_character_trajectory_axes": by_char,
        "verdict": verdict,
        "interpretation": {
            "BATTERY_IS_THE_WEAK_LINK":
                "Trajectories separate characters the probe battery cannot. The "
                "convergence measured across every distillation experiment may be "
                "an instrument artefact rather than a fact about the agents — the "
                "battery is not capturing what actually differs between them.",
            "BOTH_SEPARATE":
                "Both representations separate characters, so the battery is "
                "working. Convergence measured under distillation is then a real "
                "property of the weights, not a measurement failure.",
            "NEITHER_SEPARATES":
                "Neither trajectories nor probes separate characters we know to "
                "be behaviourally distinct. That points at the sample or the "
                "axes, and this analysis cannot resolve the original question.",
            "PROBE_ONLY":
                "The battery separates characters that trajectories do not — the "
                "reverse of the concern, and a sign the probe measures stated "
                "disposition with no behavioural correlate.",
            "UNDECIDABLE":
                "One representation had zero distance both within and between "
                "characters, so no separation statistic is defined. Report the "
                "raw within/between numbers rather than a verdict.",
        }[verdict],
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(analyse(Path(sys.argv[1])), indent=2))
