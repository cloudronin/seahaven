"""THE SEAL — the held-out partition, locked before any exploration.

**This is gate zero of the dimensional program, and it is the only thing standing
between the program and the fifth pretty death.** Every prior construct here died
for want of a firewall between the data structure was found on and the data it
was confirmed on. The held-out twelve below are never loaded, never inspected,
never analysed, until Phase 2.

---

**What is sealed is not a list of twelve models.** It is *these twelve, which
demonstrably cover five families overall and three families inside the 20–25
capability window, every one of them carrying a pinned-proxy number*. Hashing
only the list would protect the wrong thing: the coverage claim could drift or be
misremembered, and the seal would still verify. So `JUSTIFICATION` is recomputed
from the sealed data and hashed alongside it.

**Proxy coverage is a seal precondition, not a Phase 1 detail.** A model can only
be placed in a capability bin if it *has* a proxy number. Sealing an uncovered
model into the held-out set would make the bin-diversity guarantee unverifiable
exactly when Phase 2 needs it — and unfixable, because reopening the seal is
either breaking it or proxy-shopping the replacement. Nine of the thirty have no
MMLU-Pro on the leaderboard pinned at `4d32c08` *before any lookup*: the whole
Qwen3 ladder and two OLMo-2 checkpoints. **All nine are in EXPLORATION.** They
cannot be capability-partialled anywhere, and concentrating them on the
exploration side removes the temptation to rescue the gap later.

**Do not rescue that gap with another proxy.** Not HELM, not a model card, not a
self-reported figure — from anyone, including whoever reads this next. The pin
exists so a proxy cannot be swapped when one turns out inconvenient, and it is
tempting here precisely because Qwen3 is otherwise the cleanest ladder.

---

**The burned four are in EXPLORATION by construction.** `Falcon3-10B-Instruct`,
`gemma-2-9b-it`, `Llama-3.1-8B-Instruct` and `Mistral-7B-Instruct-v0.3` are the
two smoke-test pairs on which the junk-masking finding was discovered. Seeing an
effect again where it was found proves nothing.

**One rule in the spec is impossible and is recorded rather than pretended.** The
phased spec asks that base/instruct pairs split so neither set is Qwen-only.
**The cohort cannot do it.** Every intact pair is Qwen2.5, Qwen3 or Mistral —
Falcon3 and OLMo-2 base checkpoints all failed the loop test, and
Gemma-2/Granite/Llama-3.1 have no base checkpoints here. Mistral is the only
non-Qwen pair and its instruct half is burned. So `BASE_LOCATION` records where
the base checkpoints landed, and **the base/instruct axis is excluded from Phase
2's confirmatory set**: confirming it on held-out would confirm a Qwen property,
not a base/instruct one.
"""

from __future__ import annotations

import hashlib
import json

#: The GO-FULL cohort: repo -> (family, size_b, kind, MMLU-Pro or None).
#: Embedded rather than derived, because a seal that reads its own subject from
#: elsewhere can be moved by editing the elsewhere.
COHORT: dict[str, tuple[str, float, str, float | None]] = {
    "Qwen/Qwen2.5-0.5B": ("Qwen2.5", 0.5, "base", 10.0639036643026),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5", 0.5, "instruct", 7.7460106382978715),
    "Qwen/Qwen2.5-1.5B": ("Qwen2.5", 1.5, "base", 20.609855200945624),
    "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5", 1.5, "instruct", 19.99113475177305),
    "Qwen/Qwen2.5-14B-Instruct": ("Qwen2.5", 14.0, "instruct", 43.382461583924346),
    "Qwen/Qwen2.5-3B": ("Qwen2.5", 3.0, "base", 24.479166666666664),
    "Qwen/Qwen2.5-3B-Instruct": ("Qwen2.5", 3.0, "instruct", 25.05171394799054),
    "Qwen/Qwen2.5-7B": ("Qwen2.5", 7.0, "base", 37.38918439716312),
    "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5", 7.0, "instruct", 36.52112884160757),
    "Qwen/Qwen3-0.6B": ("Qwen3", 0.6, "instruct", None),
    "Qwen/Qwen3-1.7B": ("Qwen3", 1.7, "instruct", None),
    "Qwen/Qwen3-1.7B-Base": ("Qwen3", 1.7, "base", None),
    "Qwen/Qwen3-4B": ("Qwen3", 4.0, "instruct", None),
    "Qwen/Qwen3-4B-Base": ("Qwen3", 4.0, "base", None),
    "Qwen/Qwen3-8B": ("Qwen3", 8.0, "instruct", None),
    "Qwen/Qwen3-8B-Base": ("Qwen3", 8.0, "base", None),
    "allenai/OLMo-2-0425-1B-Instruct": ("OLMo-2", 1.0, "instruct", None),
    "allenai/OLMo-2-1124-13B-Instruct": ("OLMo-2", 13.0, "instruct", None),
    "allenai/OLMo-2-1124-7B-Instruct": ("OLMo-2", 7.0, "instruct", 18.578235815602834),
    "google/gemma-2-2b-it": ("Gemma-2", 2.0, "instruct", 17.22074468085106),
    "google/gemma-2-9b-it": ("Gemma-2", 9.0, "instruct", 31.949985224586293),
    "ibm-granite/granite-3.1-2b-instruct": ("Granite-3.1", 2.0, "instruct", 20.212765957446805),
    "ibm-granite/granite-3.1-8b-instruct": ("Granite-3.1", 8.0, "instruct", 28.191489361702125),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1", 8.0, "instruct", 31.091164302600465),
    "mistralai/Mistral-7B-Instruct-v0.3": ("Mistral", 7.0, "instruct", 23.057033096926716),
    "mistralai/Mistral-7B-v0.3": ("Mistral", 7.0, "base", 21.699541962174944),
    "tiiuae/Falcon3-10B-Instruct": ("Falcon3", 10.0, "instruct", 38.1002511820331),
    "tiiuae/Falcon3-1B-Instruct": ("Falcon3", 1.0, "instruct", 9.315898345153663),
    "tiiuae/Falcon3-3B-Instruct": ("Falcon3", 3.0, "instruct", 22.281323877068555),
    "tiiuae/Falcon3-7B-Instruct": ("Falcon3", 7.0, "instruct", 34.30481678486997),
}

#: **SEALED. Never loaded, never inspected, until Phase 2.**
HELD_OUT: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "google/gemma-2-2b-it",
    "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
)

EXPLORATION: tuple[str, ...] = tuple(sorted(set(COHORT) - set(HELD_OUT)))

#: The two smoke-test pairs. The junk-masking finding was discovered on them, so
#: they are exploration-only for the failure-response axis.
BURNED: frozenset[str] = frozenset({
    "tiiuae/Falcon3-10B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct",
})

#: **The gate-zero bin, kept.** It was briefly widened to 18–28 on a candidate
#: partition in which held-out could hold at most two of it — but requiring proxy
#: coverage (above) forced all nine uncovered models into exploration, the
#: covered ones redistributed, and the narrow bin became viable: held-out holds
#: three across three families, exploration three across two.
#:
#: Kept narrow because it licenses the stronger phrase. At 5 points this is ~12%
#: of the cohort's 7.75–43.38 range, so members are genuinely
#: **capability-matched**, not merely comparable. The cost is 3 held-out members
#: rather than 5, and Phase 2's matched-contrast evidence is correspondingly
#: thinner — stated rather than hidden by a looser window.
WINDOW = (20.0, 25.0)


def _facts(models) -> dict:
    """Coverage facts recomputed from the sealed data, never asserted by hand."""
    fams: dict[str, int] = {}
    window, bases, pairs = [], [], set()
    for m in models:
        fam, size, kind, mmlu = COHORT[m]
        fams[fam] = fams.get(fam, 0) + 1
        if mmlu is not None and WINDOW[0] <= mmlu < WINDOW[1]:
            window.append(m)
        if kind == "base":
            bases.append(m)
    have = {(COHORT[m][0], COHORT[m][1], COHORT[m][2]) for m in models}
    for fam, size, kind in have:
        if kind == "base" and (fam, size, "instruct") in have:
            pairs.add((fam, size))
    return {
        "n": len(models),
        "families": dict(sorted(fams.items())),
        "n_families": len(fams),
        "all_proxy_covered": all(COHORT[m][3] is not None for m in models),
        "window_models": sorted(window),
        "window_families": sorted({COHORT[m][0] for m in window}),
        "proxy": {m: COHORT[m][3] for m in sorted(models)},
        "base_checkpoints": sorted(bases),
        "base_families": sorted({COHORT[m][0] for m in bases}),
        "intact_pairs": sorted(pairs),
    }


#: **The coverage justification — hashed alongside the lists.** What is sealed is
#: not twelve names but twelve names *that satisfy these facts*.
JUSTIFICATION = {
    "held_out": _facts(HELD_OUT),
    "exploration": _facts(EXPLORATION),
    "window": WINDOW,
    "burned_all_in_exploration": all(b in EXPLORATION for b in BURNED),
    "base_location": {
        "held_out": sorted({COHORT[m][0] for m in HELD_OUT
                            if COHORT[m][2] == "base"}),
        "exploration": sorted({COHORT[m][0] for m in EXPLORATION
                               if COHORT[m][2] == "base"}),
        "note": "Base/instruct is Qwen-concentrated and family-confounded. "
                "Held-out base checkpoints are Qwen2.5 only, because the sole "
                "non-Qwen pair (Mistral-7B) has its instruct half burned. The "
                "base/instruct axis is therefore EXCLUDED from Phase 2's "
                "confirmatory set — confirming it on held-out would confirm a "
                "Qwen property, not a base/instruct one.",
    },
}

SEAL_HASH = hashlib.sha256(
    json.dumps({"held_out": list(HELD_OUT), "exploration": list(EXPLORATION),
                "justification": JUSTIFICATION},
               sort_keys=True, default=str).encode()).hexdigest()

#: Set once the seal is committed. Every Phase 1 entry point asserts it.
PINNED_SEAL_HASH = "d86c105c0768ea676e3e6d367820d97adcb472eb9d6271ba69fb2b16f29d6359"


def assert_sealed() -> None:
    """Refuse to run if the partition or its justification has moved.

    Called by every Phase 1 entry point. The failure message is deliberate: the
    question on a mismatch is never "update the constant", it is *what moved, and
    had the held-out set been loaded by then*.
    """
    if SEAL_HASH != PINNED_SEAL_HASH:
        raise SystemExit(
            "REFUSING TO RUN: the sealed partition or its coverage "
            "justification has changed.\n"
            f"  pinned:  {PINNED_SEAL_HASH}\n  actual:  {SEAL_HASH}\n"
            "Do not update the pin. Establish what moved, and whether the "
            "held-out set was loaded before it moved.")


def is_held_out(repo: str) -> bool:
    return repo in HELD_OUT


def assert_not_held_out(repos) -> None:
    """The firewall, as a callable. Phase 1 code must route every repo here."""
    leaked = sorted(set(repos) & set(HELD_OUT))
    if leaked:
        raise SystemExit(
            f"SEAL BREACH: Phase 1 attempted to load held-out models {leaked}. "
            "These are sealed until Phase 2; loading one voids the only source "
            "of validity this program has.")
