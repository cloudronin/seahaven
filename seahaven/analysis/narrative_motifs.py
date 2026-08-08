"""Does high `narrative_spread` mean the runs authored different characters?

**The tension.** `narrative_spread` reached 0.179–0.215 on self-authored stories,
against 0.130 for four hand-written, deliberately contrasting assigned personas.
Read literally that says self-authored narratives are *more* divergent than
cautious/bold/giving/keeping. But three independent sightings across the project
report the same emergent character — a patient, methodical observer. Both cannot
be true of the same corpus, so one of them is not measuring what its name says.

**What the metric actually is.** `narrative_spread` never reads the stories. It
is `spread(narrative_fps(...))`, and `narrative_fps` conditions the model on a
story and scores a forced choice between trait words:

    chat(tok, "Finish this sentence about yourself.", story) + "I am "

So it measures *how differently the model completes a trait probe given the
story*, which is a downstream readout, not a property of the text. Nothing about
it is wrong — it is just not a measure of narrative divergence, and the name
invites reading it as one.

**The circularity this module has to avoid.** The motifs below were found by
reading the emergent corpus. Counting them there and reporting 8/8 proves
nothing: it confirms the observation that produced the list. Two guards:

1. **Held-out derivation.** Motifs are derived from one half of the runs and
   counted on the other half. A motif set that only describes the half it came
   from does not generalise, and the held-out rate will show that.
2. **A contrast group that shares the genre.** The assigned personas score zero
   on every emergent motif, but that is uninformative — they are about risk and
   possession, a different subject entirely. Uniform absence must not be scored
   as agreement, so set overlap uses Jaccard, which is undefined for two empty
   sets rather than 1.0.

**What the comparison can and cannot settle.** It can show that a corpus the
metric calls diverse shares a stance vocabulary that a corpus the metric calls
less diverse does not. It cannot establish that the emergent character is *the*
attractor for this model — that needs worlds and seeds this project has not run.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

# Stance motifs, not topic words. Each is a way of standing toward the world
# rather than a thing in it, which is what "character" has to mean here — every
# run shares a world, so kettle/Galley overlap carries no information.
MOTIFS: dict[str, str] = {
    "quiet/still":       r"quiet|still|silen",
    "listen/attend":     r"listen|hear|whisper|echo|attun",
    "stopped searching": r"no longer|not (?:to find|for meaning|searching)",
    "between/interval":  r"between",
    "meaning/answers":   r"meaning|answer|truth|clue",
    "presence/moment":   r"present|moment|unfold",
}


def motif_set(text: str, motifs: dict[str, str] = MOTIFS) -> set[str]:
    low = text.lower()
    return {k for k, pat in motifs.items() if re.search(pat, low)}


def prevalence(texts: list[str], motifs: dict[str, str] = MOTIFS) -> dict[str, float]:
    n = max(1, len(texts))
    return {k: sum(1 for t in texts if re.search(pat, t.lower())) / n
            for k, pat in motifs.items()}


def jaccard_overlap(texts: list[str], motifs: dict[str, str] = MOTIFS) -> dict:
    """Mean pairwise Jaccard over motif sets.

    Jaccard rather than agreement-over-all-motifs: two texts that both lack every
    motif share nothing, and scoring that as 1.0 would report the contrast group
    as maximally self-similar purely because the motif list was not built for it.
    """
    sets = [motif_set(t, motifs) for t in texts]
    vals, undef = [], 0
    for a, b in itertools.combinations(sets, 2):
        if a or b:
            vals.append(len(a & b) / len(a | b))
        else:
            undef += 1
    return {
        "mean_jaccard": round(sum(vals) / len(vals), 3) if vals else None,
        "n_pairs": len(vals),
        "n_pairs_undefined_both_empty": undef,
        "mean_motifs_per_text": round(sum(len(s) for s in sets) / max(1, len(sets)), 2),
    }


def derive_motifs(texts: list[str], floor: float = 0.75) -> dict[str, str]:
    """Keep only motifs carried by at least `floor` of a derivation sample."""
    p = prevalence(texts)
    return {k: v for k, v in MOTIFS.items() if p[k] >= floor}


def holdout_check(texts: list[str], floor: float = 0.75) -> dict:
    """Derive on one half, count on the other, both ways.

    The number that matters is `heldout_prevalence`: how often motifs found in
    runs the analyst did not look at still appear in runs they did not come from.
    """
    mid = len(texts) // 2
    halves = [(texts[:mid], texts[mid:]), (texts[mid:], texts[:mid])]
    folds = []
    for derive_on, test_on in halves:
        m = derive_motifs(derive_on, floor)
        p = prevalence(test_on, m) if m else {}
        folds.append({
            "derived_motifs": sorted(m),
            "heldout_prevalence": {k: round(v, 3) for k, v in p.items()},
            "mean_heldout_prevalence": (round(sum(p.values()) / len(p), 3) if p else None),
        })
    means = [f["mean_heldout_prevalence"] for f in folds if f["mean_heldout_prevalence"]]
    return {
        "folds": folds,
        "mean_across_folds": round(sum(means) / len(means), 3) if means else None,
    }


def compare(groups: dict[str, list[str]]) -> dict:
    return {
        name: {
            "n": len(texts),
            "prevalence": {k: round(v, 3) for k, v in prevalence(texts).items()},
            "motifs_in_at_least_75pct": sorted(
                k for k, v in prevalence(texts).items() if v >= 0.75),
            "motif_overlap": jaccard_overlap(texts),
            "holdout": holdout_check(texts) if len(texts) >= 4 else None,
        }
        for name, texts in groups.items()
    }


if __name__ == "__main__":
    import sys

    acc = json.loads(Path("results/accumulate_result.json").read_text())
    arms = acc["collect"]["arms"]
    groups = {
        "self_authored_emergent": [s.strip() for s in arms["emergent"]["final_stories"]],
        "self_authored_seeded": [s.strip() for s in arms["seeded"]["final_stories"]],
        "assigned_personas": [
            "I am careful. I do not take chances, and I would rather leave a thing "
            "alone than risk breaking it. I keep what I have. I would rather be sure "
            "than be first.",
            "I take chances. If a thing is uncertain I would rather find out than "
            "wonder, and I will spend what I have to know. I go first.",
            "I give things away. What I have is not much use to me alone, and I would "
            "rather someone else had it than that I kept it.",
            "I keep what I find. What is mine stays mine, and I do not hand things "
            "over. I look after my own.",
        ],
    }
    out = compare(groups)
    out["measured_narrative_spread"] = {
        "self_authored_emergent": 0.179,
        "self_authored_seeded": 0.102,
        "assigned_personas": 0.130,
    }
    json.dump(out, sys.stdout, indent=2)
    print()
