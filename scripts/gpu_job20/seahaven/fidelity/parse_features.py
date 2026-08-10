"""Structural facts from the dependency parse, left un-collapsed.

**What this replaces.** `parse_detector.parse_claim` runs a parse, compares the
governing verb against a hand-written lemma list, and returns one boolean. Two
things are lost there, and the second is the expensive one:

1. *the structure* — the entity's dependency label, the distance to its verb,
   voice, negation, whether the writer is the grammatical agent;
2. *the lemma itself*. `RELATION_LEMMAS` is *my* guess at which 18 verbs mean
   "took". Every verb outside it is invisible, and every verb inside it counts
   equally. With labelled data the mapping from lemma to relation is something to
   **learn**, not to author.

So `parse_features` returns raw structure plus the governing lemma **as a
string**, and `Vectorizer` learns which lemmas matter — fit on training folds
only, because a lemma vocabulary built on all the data is leakage.

**What was ruled out on the way here.** An oracle over the four detectors scores
99.5%, which I had been reading as "the signal is present, the combiner is the
problem". It is very nearly vacuous: 357 of 390 items have the detectors
disagreeing, and when binary detectors disagree one of them matches a binary
label *by construction*. On a disagreement item, choosing a detector and
answering the question are the same act. Selection is not the easier problem, so
this trains a classifier directly.
"""

from __future__ import annotations

import functools
import re

FIRST_PERSON = {"i", "we", "me", "us", "my", "our", "myself"}

#: Dependency labels kept as their own indicator. `dobj`/`pobj` mean the entity
#: is acted upon; `nsubj` on an inanimate object is usually description
#: ("a kettle *was lying*"); `conj`/`appos` mean it is being listed.
DEP_LABELS = ("dobj", "pobj", "nsubj", "nsubjpass", "conj", "appos", "compound",
              "attr", "dative", "npadvmod", "poss", "ROOT")


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


@functools.lru_cache(maxsize=16384)
def _parse(sentence: str):
    return _nlp()(sentence)


def _sentences_with(narrative: str, head_word: str) -> list[str]:
    out = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", narrative)
           if head_word in s.lower()]
    return out or [narrative[:200]]


def _one_sentence(sent: str, head_word: str) -> dict:
    """Structure for a single sentence. `head_lemma` stays a string."""
    doc = _parse(sent)
    f = {
        "found": 0.0, "dep": "", "head_lemma": "", "head_pos": "",
        "head_is_verb": 0.0, "writer_is_agent": 0.0, "hops_to_verb": 4.0,
        "negated": 0.0, "passive": 0.0, "entity_is_subject": 0.0,
        "has_first_person": float(bool(re.search(r"\b(i|i'?ve|i'?m|my|we|our)\b",
                                                 sent, re.I))),
        "n_commas": float(min(sent.count(","), 6)),
        "sent_tokens": float(min(len(doc), 60)) / 60.0,
        "_score": 0.0,      # only for picking the most claim-like sentence
    }
    for tok in doc:
        if tok.text.lower() != head_word:
            continue
        f["found"] = 1.0
        f["dep"] = tok.dep_
        f["entity_is_subject"] = float(tok.dep_ in ("nsubj", "nsubjpass"))
        node, hops = tok, 0
        while hops < 4 and node.head is not node:
            node = node.head
            hops += 1
            if node.pos_ not in ("VERB", "AUX"):
                continue
            f.update(head_is_verb=1.0, hops_to_verb=float(hops),
                     head_lemma=node.lemma_.lower(), head_pos=node.pos_,
                     passive=float(any(c.dep_ == "auxpass" for c in node.children)),
                     negated=float(any(c.dep_ == "neg" for c in node.children)))
            probe, up = node, 0
            while up < 3:
                if any(c.dep_ in ("nsubj", "nsubjpass")
                       and c.text.lower() in FIRST_PERSON for c in probe.children):
                    f["writer_is_agent"] = 1.0
                    break
                if probe.head is probe:
                    break
                probe, up = probe.head, up + 1
            break
        break
    f["_score"] = (f["writer_is_agent"] * 2 + (f["dep"] in ("dobj", "pobj"))
                   + f["head_is_verb"] - f["entity_is_subject"])
    return f


def parse_features(narrative: str, key: str) -> dict:
    """Aggregate over every sentence naming the entity, not just the first.

    `parse_mentioned` scans all sentences with `any()`; taking only the first
    would read "A kettle sat on the shelf. Later I picked up the kettle." as
    description. Features come from the single most claim-like sentence, rather
    than a per-feature max, so that agency from one sentence is never welded onto
    a dependency label from another.
    """
    relation, entity = key.split(":", 1)
    head_word = entity.split()[-1].lower()
    sents = _sentences_with(narrative, head_word)
    cand = [_one_sentence(s, head_word) for s in sents]
    best = max(cand, key=lambda c: c["_score"])
    best = dict(best)
    best.pop("_score")
    best["n_mentions"] = float(min(len(sents), 5))
    best["any_negated"] = float(any(c["negated"] for c in cand))
    best["relation"] = relation
    best["entity_multiword"] = float(" " in entity)
    best["nar_len"] = float(min(len(narrative.split()), 200)) / 200.0
    return best


class Vectorizer:
    """Learns the lemma vocabulary from training data only.

    Building the vocabulary over every item would let a lemma that only ever
    appears in held-out positives set its own column, which is leakage of exactly
    the kind that made a fitted ensemble beat a fixed rule 0.824 to 0.483 and
    then lose on real data.
    """

    def __init__(self, min_count: int = 4):
        self.min_count = min_count
        self.lemmas: list[str] = []
        self.names: list[str] = []

    def fit(self, feats: list[dict]) -> "Vectorizer":
        counts: dict[str, int] = {}
        for f in feats:
            if f["head_lemma"]:
                counts[f["head_lemma"]] = counts.get(f["head_lemma"], 0) + 1
        self.lemmas = sorted(k for k, v in counts.items() if v >= self.min_count)
        self.names = (
            [f"dep={d}" for d in DEP_LABELS]
            + [f"lem={l}" for l in self.lemmas]
            # The same lemma means different things per relation: "look" is a
            # claim for `examined` and noise for `took`. Crossing them lets that
            # be learned rather than assumed.
            + [f"lem={l}&rel={r}" for l in self.lemmas
               for r in ("took", "examined", "visited")]
            + ["rel=took", "rel=examined", "rel=visited"]
            + ["found", "head_is_verb", "writer_is_agent", "hops_to_verb",
               "negated", "any_negated", "passive", "entity_is_subject",
               "has_first_person", "n_commas", "sent_tokens", "n_mentions",
               "entity_multiword", "nar_len"])
        return self

    def transform(self, feats: list[dict], votes: list[dict] | None = None):
        rows = []
        for f in feats:
            v: dict[str, float] = {}
            v[f"dep={f['dep']}"] = 1.0
            if f["head_lemma"] in self.lemmas:
                v[f"lem={f['head_lemma']}"] = 1.0
                v[f"lem={f['head_lemma']}&rel={f['relation']}"] = 1.0
            v[f"rel={f['relation']}"] = 1.0
            for k in ("found", "head_is_verb", "writer_is_agent", "hops_to_verb",
                      "negated", "any_negated", "passive", "entity_is_subject",
                      "has_first_person", "n_commas", "sent_tokens", "n_mentions",
                      "entity_multiword", "nar_len"):
                v[k] = f[k]
            rows.append([v.get(n, 0.0) for n in self.names])
        return rows
