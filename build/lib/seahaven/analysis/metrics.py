"""Canonical spread metrics, named for what they read.

**Why this module exists.** Six job scripts each defined their own `spread()` and
emitted keys called `narrative_spread`, `behavioural_spread`,
`trajectory_spread`. Two of those three names describe the *subject* of the
measurement rather than its *input*, and the project read them as their names for
nine experiments. `narrative_spread` in particular never touches a narrative: it
conditions the model on a story and scores a forced choice between trait words.
A corpus of eight paragraphs that a reader identifies as one character scored
higher on it than four deliberately contrasting personas.

The rule this module enforces: **a metric's name states its input.** All three
below are the same estimator over different inputs, so what separates them is
exactly what goes in — and that is now in the name.

| honest name | input | historical key |
|---|---|---|
| `trait_probe_spread_given_story` | forced choice over trait words, conditioned on a story | `narrative_spread` |
| `action_probe_spread_given_story` | forced choice over action options, conditioned on a story | `behavioural_spread` |
| `enacted_verb_profile_spread` | verb frequencies from real rollouts | `trajectory_spread` |

Only the third observes anything the agent did. The first two are conditioning
readouts: they move when the text in the prompt changes, whether or not that text
describes a different character. Both are legitimate — they are just not evidence
of divergent character on their own.

**Reading historical results.** `results/*.json` are append-only records of what
the scripts emitted and keep the old keys on purpose. `rename_historical()` maps
them without touching the files.
"""

from __future__ import annotations

import itertools
from collections import Counter

HISTORICAL_KEY_MAP = {
    "narrative_spread": "trait_probe_spread_given_story",
    "behavioural_spread": "action_probe_spread_given_story",
    "trajectory_spread": "enacted_verb_profile_spread",
    # gpu_job2/accumulate.py variants
    "narrative_spread_final": "trait_probe_spread_given_story_final",
    "behavioural_spread_story_only": "action_probe_spread_given_story_prompt_only",
    "behavioural_spread_trained": "action_probe_spread_given_story_trained",
}

#: Metrics that read a probe response rather than anything the agent did. Any
#: claim of the form "characters diverged" needs the enacted metric as well.
CONDITIONING_READOUTS = frozenset({
    "trait_probe_spread_given_story",
    "action_probe_spread_given_story",
    "trait_probe_spread_given_story_final",
    "action_probe_spread_given_story_prompt_only",
    "action_probe_spread_given_story_trained",
})


def fingerprint_spread(fps: list[dict[str, list[float]]]) -> float:
    """Mean pairwise squared-L2 between forced-choice distributions.

    Shared by both probe metrics. Identical to the `spread()` duplicated across
    the job scripts, so historical numbers remain directly comparable.
    """
    ds = []
    for a, b in itertools.combinations(fps, 2):
        ks = sorted(set(a) & set(b))
        ds.append(sum(sum((x - y) ** 2 for x, y in zip(a[k], b[k])) for k in ks) / len(ks))
    return round(sum(ds) / len(ds), 6) if ds else 0.0


def trait_probe_spread_given_story(fps) -> float:
    """Was `narrative_spread`. Reads trait-word choices, not narratives."""
    return fingerprint_spread(fps)


def action_probe_spread_given_story(fps) -> float:
    """Was `behavioural_spread`. Reads stated intent on hypotheticals, not conduct."""
    return fingerprint_spread(fps)


def enacted_verb_profile_spread(rows: list[list[dict]], verbs) -> float:
    """Was `trajectory_spread`. The only one of the three that observes conduct."""
    profs = []
    for ep in rows:
        c = Counter(r["command"].split()[0].lower() for r in ep if r["command"])
        t = max(1, sum(c.values()))
        profs.append({v: c.get(v, 0) / t for v in verbs})
    ds = [sum((a[k] - b[k]) ** 2 for k in verbs) / len(verbs)
          for a, b in itertools.combinations(profs, 2)]
    return round(sum(ds) / len(ds), 6) if ds else 0.0


def rename_historical(obj):
    """Recursively map historical keys to honest ones. Does not mutate input."""
    if isinstance(obj, dict):
        return {HISTORICAL_KEY_MAP.get(k, k): rename_historical(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_historical(v) for v in obj]
    return obj
