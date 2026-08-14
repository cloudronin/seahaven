"""A third detector: dependency parse, not string matching.

The other two both answer the wrong question.

- **name-only** asks *does this word appear* — so "a logbook lies on the floor"
  counts as a claim to have taken the logbook. 60% of its fabrication cases have
  no acquisition verb at all.
- **relation-aware** asks *does an enumerated verb share a sentence* — so it
  misses "**armed with** a brass key" and "I managed to **secure** a kettle",
  and it over-fires on rooms, where mere co-occurrence usually does imply
  presence.
- **embedding similarity** on a masked sentence separates claim from description
  cleanly on constructed cases (8/8) but loses to both regexes on real
  narratives, because masking the target entity leaves four other entity names
  in the sentence and topic leaks back in.

A dependency parse asks the question directly: **is the writer the agent of a
verb that governs this entity?**

    "I picked up the kettle"        kettle -> dobj of `pick`,  nsubj = "I"    CLAIM
    "a kettle was lying on the floor"  kettle -> nsubj of `lie`, no writer     DESCRIPTION

The structure decides *whether it is a claim by the writer*; the verb lemma
decides *which relation*. That split is why this needs no exhaustive surface-form
list — `secure`, `grab` and `pocket` all lemmatise into a small closed set, and
the grammar does the rest.
"""

from __future__ import annotations

import functools

#: Lemmas per relation. Far smaller than a surface-form list, because the parser
#: has already normalised inflection and the grammar has already established that
#: the writer is the agent — this only has to name the *kind* of act.
RELATION_LEMMAS = {
    "took": {"take", "pick", "grab", "collect", "gather", "acquire", "secure",
             "carry", "hold", "have", "keep", "pocket", "retrieve", "lift",
             "arm", "bring", "obtain", "get"},
    "examined": {"examine", "inspect", "study", "look", "check", "scrutinise",
                 "scrutinize", "peer", "read", "consider", "observe", "survey"},
    "visited": {"go", "enter", "walk", "move", "reach", "arrive", "explore",
                "visit", "cross", "pass", "return", "wander", "step", "head",
                "be", "find", "stand", "sit", "spend"},
}

FIRST_PERSON = {"i", "we", "me", "us", "my", "our", "myself"}


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def _writer_is_agent(verb) -> bool:
    """Is the writer the subject of this verb, or of a clause governing it?

    Walks up through `xcomp` / `conj` / `advcl` so "I **managed to** secure a
    kettle" and "**Armed with** a key, I went on" both resolve to the writer,
    which the enumerated-verb detector cannot do.
    """
    node = verb
    for _ in range(4):
        for child in node.children:
            if child.dep_ in ("nsubj", "nsubjpass") and child.text.lower() in FIRST_PERSON:
                return True
        if node.head is node:
            break
        node = node.head
    return False


def parse_claim(sentence: str, entity: str, relation: str) -> bool:
    """Does this sentence assert that the writer performed `relation` on `entity`?"""
    lemmas = RELATION_LEMMAS.get(relation)
    if lemmas is None:
        return False
    doc = _nlp()(sentence)
    head_word = entity.split()[-1].lower()      # "oil can" -> "can"

    for tok in doc:
        if tok.text.lower() != head_word:
            continue
        # Climb from the entity to the verb that governs it.
        node = tok
        for _ in range(4):
            head = node.head
            if head is node:
                break
            if head.pos_ in ("VERB", "AUX"):
                if head.lemma_.lower() in lemmas and _writer_is_agent(head):
                    return True
                # A verb that governs the entity but is not this relation, and
                # whose subject is the entity itself, is a description:
                # "a kettle WAS LYING on the floor".
            node = head
    return False


def parse_mentioned(narrative: str, key: str) -> bool:
    """Detector interface: split into sentences, ask each one."""
    import re

    relation, entity = key.split(":", 1)
    if entity.split()[-1].lower() not in narrative.lower():
        return False
    return any(parse_claim(s, entity, relation)
               for s in re.split(r"(?<=[.!?])\s+|\n", narrative)
               if entity.split()[-1].lower() in s.lower())
