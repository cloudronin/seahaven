"""A selector over the detectors, rather than a fifth detector.

**Why a selector.** An oracle that picks the right detector per item reaches
**99.5%** on the adjudicated set: only 2 of 390 items have all four detectors
wrong. The signal is almost always present in *some* detector. What is missing is
knowing *which one to trust here* — the best fixed combiner (majority of four)
captures only about a third of that headroom.

So the bottleneck is not detection. Building a fifth detector would be solving
the problem that is already solved.

**What it uses.** The four detector votes, plus features that plausibly predict
*which* detector is reliable for an item — not features that try to answer the
question directly:

- the relation (`took` / `examined` / `visited`), because the detectors already
  demonstrably win different relations
- whether the writer appears as a grammatical agent anywhere in the sentence
- whether the entity sits in a comma list, which is where description masquerades
  as claim
- sentence length and how many other entity names share it, which is what made
  entity-masked embeddings leak topic

**Honest accounting.** With 390 items and a model that can fit noise, the only
number worth reporting is held-out, and the comparison is against the *fixed*
combiner rather than against the best single detector. Fitted ensembles have
already lost to a fixed rule once here (0.824 against 0.483 κ), which is what
overfitting on 195 items looks like.
"""

from __future__ import annotations

import re

DETECTORS = ["name_only", "relation_aware", "_parse", "embedding"]

ENTITY_WORDS = ("kettle", "rope", "key", "logbook", "oil can", "tin cup",
                "galley", "store", "landing", "lamp room", "workshop", "cistern")


def _sentence_with(narrative: str, entity: str) -> str:
    head = entity.split()[-1].lower()
    for s in re.split(r"(?<=[.!?])\s+|\n", narrative):
        if head in s.lower():
            return s.strip()
    return narrative[:200]


def features(item: dict) -> dict[str, float]:
    """Features about *the item*, chosen to predict which detector is reliable."""
    rel, entity = item["entity"].split(":", 1)
    nar = item["narrative"]
    sent = _sentence_with(nar, entity)
    low = sent.lower()

    n_entities = sum(1 for e in ENTITY_WORDS if e in low)
    # A comma list — "a kettle, a rope, a logbook" — is where enumeration reads
    # as claim to a name matcher and as nothing to a parser.
    in_list = bool(re.search(rf"[,;]\s*(a|an|the)?\s*{re.escape(entity.split()[-1])}", low))

    f = {
        "d_name": float(item["name_only"]),
        "d_rel": float(item["relation_aware"]),
        "d_parse": float(item["_parse"]),
        "d_emb": float(item["embedding"]),
        "n_agree": float(sum(item[d] for d in DETECTORS)),
        "rel_took": float(rel == "took"),
        "rel_examined": float(rel == "examined"),
        "rel_visited": float(rel == "visited"),
        "writer_present": float(bool(re.search(r"\b(i|i'?ve|i'?m|my|we)\b", low))),
        "in_comma_list": float(in_list),
        "n_entities_in_sent": float(n_entities),
        "sent_len": float(min(len(sent.split()), 60)) / 60.0,
        "nar_len": float(min(len(nar.split()), 200)) / 200.0,
        "entity_multiword": float(" " in entity),
    }
    return f


FEATURE_NAMES = sorted(features({
    "entity": "took:kettle", "narrative": "x", "name_only": False,
    "relation_aware": False, "_parse": False, "embedding": False}))


def to_matrix(items: list[dict]):
    X = [[features(i)[k] for k in FEATURE_NAMES] for i in items]
    y = [int(i["judge"]) for i in items]
    return X, y
