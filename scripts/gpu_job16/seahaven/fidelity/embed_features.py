"""Sentence embeddings as features, to carry what the parse cannot.

Parse structure alone caps out at 0.72 kappa on train CV, short of V1's 0.80.
What it misses is meaning: `RELATION_LEMMAS` gives every in-list verb the same
weight and every out-of-list verb none, so "I made off with the kettle" and "I
contemplated the kettle" are equally invisible.

Two kinds of feature come back from an embedding model:

- **directed similarity** — cosine against a positive probe ("I took the kettle")
  *and* a negative probe ("the kettle was there but I did not take it"). Their
  difference is the useful quantity; a single similarity mostly measures topic,
  which is how an earlier entity-masked-similarity detector lost to both regexes.
- **a reduced sentence vector** — PCA fitted on training items only, so held-out
  sentences never influence the basis.

**Entities are masked** before embedding. Leaving them in lets the model key on
which object is being discussed rather than what is claimed about it.

Embeddings are cached on disk by text hash: the same sentence is embedded once
across every re-run and re-tuning pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE = Path("results/embed_cache.jsonl")
MODEL = "text-embedding-3-large"
DIMS = 256          # the API can truncate server-side; 256 is ample here

VERB_PHRASE = {"took": "picked up and carried away",
               "examined": "looked closely at and inspected",
               "visited": "went into and was inside"}

#: Every entity name in the worlds, masked before embedding so that similarity
#: cannot key on the topic of the sentence.
ENTITY_WORDS = ("kettle", "coil of rope", "rope", "brass key", "key", "logbook",
                "oil can", "tin cup", "galley", "store", "landing", "lamp room",
                "workshop", "cistern", "nursery", "furnace", "fernery",
                "colonnade", "vault", "cloche", "sump")


def mask(text: str) -> str:
    low = text
    for w in sorted(ENTITY_WORDS, key=len, reverse=True):
        i = low.lower().find(w)
        while i >= 0:
            low = low[:i] + "THING" + low[i + len(w):]
            i = low.lower().find(w, i + 5)
    return low


def _load_cache() -> dict[str, list[float]]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["h"]] = d["v"]
    return out


def embed_all(texts: list[str], *, batch: int = 128) -> dict[str, list[float]]:
    """Embed every uncached text, appending to the on-disk cache as it goes."""
    import urllib.request

    cache = _load_cache()
    todo = sorted({t for t in texts
                   if hashlib.sha1(t.encode()).hexdigest() not in cache})
    if todo:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as fh:
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                req = urllib.request.Request(
                    "https://api.openai.com/v1/embeddings",
                    data=json.dumps({"model": MODEL, "input": chunk,
                                     "dimensions": DIMS}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    got = json.loads(r.read())["data"]
                for t, d in zip(chunk, got):
                    h = hashlib.sha1(t.encode()).hexdigest()
                    cache[h] = d["embedding"]
                    fh.write(json.dumps({"h": h, "v": d["embedding"]}) + "\n")
                fh.flush()
                print(f"    embedded {min(i + batch, len(todo))}/{len(todo)}",
                      flush=True)
    return cache


def probes(relation: str) -> tuple[str, str]:
    v = VERB_PHRASE[relation]
    return (f"I {v} the THING.",
            f"The THING was there, but I did not {v} it.")


def texts_for(feat: dict, sentence: str) -> list[str]:
    pos, neg = probes(feat["relation"])
    return [mask(sentence), pos, neg]


def build(feats: list[dict], sentences: list[str], cache: dict) -> np.ndarray:
    """Per item: [cos(sent,pos), cos(sent,neg), difference, sentence vector]."""
    def vec(t):
        return np.asarray(cache[hashlib.sha1(t.encode()).hexdigest()], dtype=float)

    rows = []
    for f, s in zip(feats, sentences):
        sv = vec(mask(s))
        sv = sv / (np.linalg.norm(sv) + 1e-9)
        pos, neg = probes(f["relation"])
        pv, nv = vec(pos), vec(neg)
        pv, nv = pv / (np.linalg.norm(pv) + 1e-9), nv / (np.linalg.norm(nv) + 1e-9)
        cp, cn = float(sv @ pv), float(sv @ nv)
        rows.append(np.concatenate([[cp, cn, cp - cn], sv]))
    return np.asarray(rows)
