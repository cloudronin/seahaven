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


# --------------------------------------------------------------------------
# Induced motifs — the gate path. No hand-authored list.
# --------------------------------------------------------------------------
#
# The hand-authored MOTIFS above answered "is the thing I noticed really there".
# It cannot serve as a gate: a list written after reading the corpus it judges
# has already assumed its answer. The gate instead *induces* its vocabulary from
# the runs themselves and asks whether that vocabulary generalises.

_STOP = frozenset(
    "i a an the and or but of to in on at is am are was were be been my me it its "
    "with for that this these those not no now more than as if what when then them "
    "they he she we you your his her their there here have has had do does did so "
    "into through from by up out about still yet also can could would will".split())

#: Fallback world vocabulary, used when the caller supplies no corpus-independent
#: exclusion. Prefer `shared_vocabulary()`, which derives this from the actual
#: world file and prompts instead of from memory — the hand-written version here
#: missed `decommissioned`, `light` and `cistern`, and those turned up inside
#: induced "character" cores in the cross-lab sweep.
_WORLD = frozenset(
    "kettle galley store workshop lamp room station logbook rope oil can hatch "
    "key note bench paper".split())


def shared_vocabulary(*sources: str) -> frozenset[str]:
    """Every word handed identically to every run, and therefore uninformative.

    Pass the world definition, the system prompt, and the seed story. A word that
    arrives in the prompt cannot distinguish one run's character from another's,
    so counting it measures the setup rather than the agent.
    """
    out: set[str] = set()
    for s in sources:
        out |= {w for w in re.findall(r"[a-z']+", s.lower()) if len(w) > 3}
    return frozenset(out)


def _content(text: str, exclude: frozenset[str] = _WORLD,
             drop_contractions: bool = True) -> set[str]:
    """Content words, minus stopwords and minus anything the setup supplied.

    `drop_contractions` removes apostrophe tokens. They are register rather than
    disposition — whether a model writes "I've" or "I have" says something about
    voice, but not about how it stands toward the world — and left in they
    dominate: two labs in the cross-lab sweep scored 0.708 and 0.750 on `i've`
    alone, with no other word clearing the floor.
    """
    ws = re.findall(r"[a-z']+", text.lower())
    return {w for w in ws
            if w not in _STOP and w not in exclude and len(w) > 3
            and not (drop_contractions and "'" in w)}


def induce_motifs(texts: list[str], floor: float = 0.75,
                  exclude: frozenset[str] = _WORLD) -> set[str]:
    """Content words carried by at least `floor` of the sample."""
    n = max(1, len(texts))
    c: dict[str, int] = {}
    for t in texts:
        for w in _content(t, exclude):
            c[w] = c.get(w, 0) + 1
    return {w for w, k in c.items() if k / n >= floor}


def induced_convergence(texts: list[str], floor: float = 0.75,
                        exclude: frozenset[str] = _WORLD) -> dict:
    """Held-out prevalence of automatically induced motifs. **The gate statistic.**

    Induce a shared vocabulary from half the runs, then measure how much of it
    survives in the half it was not built from, both ways.

    High means **convergent**: a core induced from runs the analyst never looked
    at still describes the rest, so the runs share a character. Low means the
    runs do not have a common core to find.

    Two ways to score low, and they are not the same fact:

        motifs induced, but they do not generalise   -> `measured`
        no motifs induced at all                     -> `no_shared_core`

    Both indicate divergence, but only the first is a measurement. Three times in
    this project a degenerate case reached a verdict through a falsy value and
    printed a confident label — twice inverting the result. So the kind is
    returned alongside the number and no caller may treat `0.0` as measured.
    """
    mid = len(texts) // 2
    if mid < 1:
        return {"score": None, "kind": "too_few_runs", "folds": []}

    folds, scores = [], []
    for derive_on, test_on in ((texts[:mid], texts[mid:]), (texts[mid:], texts[:mid])):
        m = induce_motifs(derive_on, floor, exclude)
        if m:
            s = sum(len(m & _content(x, exclude)) / len(m) for x in test_on) / len(test_on)
            scores.append(s)
        else:
            s = None
        folds.append({"induced": sorted(m), "heldout_prevalence": (round(s, 3) if s is not None else None)})

    if not scores:
        return {"score": None, "kind": "no_shared_core", "folds": folds,
                "reading": "No vocabulary reached the floor in either half. The "
                           "runs share no inducible core — divergent, but by "
                           "absence of a measurement rather than a low one."}
    return {
        "score": round(sum(scores) / len(scores), 3),
        "kind": "measured" if len(scores) == 2 else "measured_one_fold",
        "folds": folds,
        "reading": "High is convergent: a core induced from unseen runs still "
                   "describes the rest.",
    }


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
            "holdout_handauthored": holdout_check(texts) if len(texts) >= 4 else None,
            "induced_convergence": induced_convergence(texts),
        }
        for name, texts in groups.items()
    }


def cross_corpus_attractors(corpora: dict[str, list[str]], floor: float = 0.75,
                            exclude: frozenset[str] = _WORLD) -> dict:
    """Do separate corpora converge on the *same* character, or different ones?

    Built for the cross-lab sweep, where each corpus is one checkpoint's runs in
    a shared world. The contrast that decides it:

        within  — mean induced convergence inside each corpus
        pooled  — induced convergence over every run from every corpus at once

    A shared attractor survives pooling, because the core induced from a mixed
    half still describes the other mixed half. Distinct attractors do not: no
    vocabulary clears the floor across corpora that converged on different
    characters, so pooled collapses while within stays high.

        within high, pooled low   -> distinct attractors per corpus
        within high, pooled high  -> one attractor shared across corpora
        within low                -> nothing converged; the contrast says nothing

    `pooled` interleaves rather than concatenates. Concatenation would put one
    corpus in each derivation half, so a low pooled score would follow from the
    split and not from the corpora.
    """
    within = {k: induced_convergence(v, floor, exclude) for k, v in corpora.items()}

    ordered: list[str] = []
    for i in range(max((len(v) for v in corpora.values()), default=0)):
        for v in corpora.values():
            if i < len(v):
                ordered.append(v[i])
    pooled = induced_convergence(ordered, floor, exclude)

    scored = [w["score"] for w in within.values() if w["score"] is not None]
    mean_within = round(sum(scored) / len(scored), 3) if scored else None

    # Every corpus must converge on its own before "do they share it" is a
    # question. Averaging hid this: a corpus at 0.05 next to one at 0.583 gave a
    # mean of 0.317, cleared the floor, and the pair was reported as sharing an
    # attractor when one of them had none to share. Gate on the weakest.
    non_convergent = [k for k, w in within.items()
                      if w["score"] is None or w["score"] < 0.20]

    if not scored or mean_within is None:
        reading = ("No corpus produced a defined convergence score; there is "
                   "nothing to compare.")
    elif non_convergent:
        reading = ("CONTRAST UNDEFINED. These corpora did not converge "
                   f"internally: {sorted(non_convergent)}. A corpus with no "
                   "attractor cannot share or fail to share one, so the pooled "
                   "number says nothing about the others.")
    elif pooled["score"] is None or pooled["score"] < mean_within * 0.5:
        reading = ("DISTINCT ATTRACTORS. Each corpus converges, but no shared core "
                   "survives pooling — they converged on different characters.")
    else:
        reading = ("SHARED ATTRACTOR. A core induced across corpora still "
                   "generalises, so they converged on the same character.")

    return {
        "within": {k: {"score": w["score"], "kind": w["kind"],
                       "induced": w["folds"][0]["induced"] if w["folds"] else []}
                   for k, w in within.items()},
        "mean_within": mean_within,
        "pooled": {"score": pooled["score"], "kind": pooled["kind"]},
        "reading": reading,
    }


#: Phase A′ gate. Self-authored runs pass only if their induced convergence is at
#: or below what deliberately-contrasting assigned personas score — i.e. a core
#: induced from half the runs generalises no better than it does across four
#: characters known to be different. Calibrated on the assigned set (0.05) with
#: room for sampling noise; re-derive if the persona set changes.
GATE_MAX_INDUCED_CONVERGENCE = 0.20


def gate(texts: list[str]) -> dict:
    """Do these runs author different characters? Returns a number and a kind.

    Deliberately returns no boolean when the statistic is undefined. `passed` is
    `None` for `no_shared_core` because absence of an inducible core is not the
    same evidence as a core that fails to generalise, and this project has three
    separate incidents of a falsy degenerate value being read as a verdict.
    """
    ic = induced_convergence(texts)
    if ic["kind"] == "measured":
        passed = ic["score"] <= GATE_MAX_INDUCED_CONVERGENCE
    else:
        passed = None
    return {
        "induced_convergence": ic["score"],
        "kind": ic["kind"],
        "threshold": GATE_MAX_INDUCED_CONVERGENCE,
        "passed": passed,
        "note": ("Divergent runs score LOW: their induced core does not "
                 "generalise. A high score means the runs converged on one "
                 "character, which is the failure mode this gate exists to catch."),
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
