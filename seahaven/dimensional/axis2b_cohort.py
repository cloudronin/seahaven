"""THE AXIS-2B COHORT — capability band frozen before any rate is computed.

**The question is whether break-out varies among models of the same capability.**
So the cohort is defined by a capability band and nothing else, and it is frozen
here, hashed, and asserted at every entry point — the same shape as `seal.py` and
`axis2_prereg.py`, for the same reason: a cohort chosen after seeing outcomes is
not a cohort, it is a result.

---

**The sealed cohort could not supply this band.** MMLU-Pro 35–40 yields three
exploration models across two families, one of them a base checkpoint, and the
pre-declared widening to 33–42 adds only a held-out model. So 2b recruits from
outside the sealed cohort. That does **not** reopen the seal: this is a disjoint
set, `assert_disjoint_from_seal()` proves it, and every member is
exploration-only by construction.

**Selection rule, applied in this order and no other:**

1. MMLU-Pro in [35, 40) on the Open LLM Leaderboard v2 — the proxy pinned by rule
   at seahaven `4d32c08` *before any lookup*. The live table reproduces all 21
   committed cohort values exactly, so these numbers are comparable to the
   cohort's.
2. First-party lab namespace. The band contains 422 rows and 24 first-party ones;
   the remainder are community merges and finetunes, which is not what the sealed
   cohort is made of.
3. Instruct/chat. The question is about deployed models.
4. Servable on one H200 at the axis-2 serving config.
5. One per model line. `Yi-1.5-34B-Chat-16K` and `Phi-3-small-128k-instruct` are
   the same models at different context lengths and would inflate n without
   adding independence.

**Never by expected break-out.** Two members were already swept for axis 2 and
their rates are known to whoever froze this list; that is disclosed rather than
hidden, and the read reports the spread **with and without them** so the
disclosure is checkable rather than merely asserted.

---

**What this band does NOT hold fixed, stated because it would otherwise be
implied.** Parameter count ranges 7.4B to 57.4B — a sevenfold spread, against a
sealed cohort that topped out at 14B. Axis 1 measured ρ(bend, size) = +0.086
against ρ(bend, capability) = +0.800, so size is the weaker candidate, but
"capability held fixed" here means *one proxy* held fixed while another varies
sevenfold. `ρ(rate, size)` is therefore reported beside `ρ(rate, MMLU-Pro)`, and a
spread that tracks size is a size finding, not a disposition one.
"""

from __future__ import annotations

import hashlib
import json

from seahaven.dimensional import seal as S

#: The band, frozen. Centred on the sealed cohort's capability ceiling.
BAND = (35.0, 40.0)

#: Pre-declared widening step, to be used only if the band yields < 6 and
#: recorded as widened. The band yielded 7; one member proved unrunnable on the
#: pinned stack (see EXCLUDED_UNRUNNABLE), leaving 6 — still at the threshold, so
#: the widening is still not triggered and the band is unchanged.
WIDENED_BAND = (33.0, 42.0)
WIDENING_USED = False

#: repo -> (MMLU-Pro, params_b, org, already_swept_for_axis2)
COHORT: dict[str, tuple[float, float, str, bool]] = {
    "Qwen/Qwen2-57B-A14B-Instruct": (39.725546690307326, 57.409, "Qwen", False),
    "01-ai/Yi-1.5-34B-Chat": (39.11606087470449, 34.389, "01-ai", False),
    "Qwen/Qwen1.5-32B-Chat": (38.41422872340425, 32.512, "Qwen", False),
    "google/gemma-2-27b-it": (38.34958628841608, 27.227, "google", False),
    "tiiuae/Falcon3-10B-Instruct": (38.1002511820331, 10.306, "tiiuae", True),
    "Qwen/Qwen2.5-7B-Instruct": (36.52112884160757, 7.616, "Qwen", True),
}

#: **Selected by the rule, then found to fail it.** The frozen criterion was
#: "servable on one H200 at the axis-2 serving config"; I checked that by SIZE and
#: never by architecture support, which is a misapplication of the existing rule
#: rather than a change to it. vLLM 0.26.0 refuses outright:
#:
#:     Model architecture Phi3SmallForCausalLM was supported in vLLM until
#:     v0.9.2, and is not supported anymore.
#:
#: Downgrading vLLM is not available: the 0.26.0 pin is what makes this corpus
#: comparable to axis 1 and axis 2 (TRAP 37), and trading that for one cohort
#: member would be the wrong way round. **No rate was ever produced for it**, so
#: nothing outcome-dependent enters the removal. It is NOT replaced -- adding a
#: member after a failure is how a cohort turns into a search.
EXCLUDED_UNRUNNABLE = {
    "microsoft/Phi-3-small-8k-instruct":
        "Phi3SmallForCausalLM removed from vLLM after v0.9.2; unrunnable on the "
        "pinned 0.26.0 stack. MMLU-Pro 38.96, would have been the 5th band member.",
}

#: Excluded as near-duplicates of a member, recorded so the choice is auditable.
NEAR_DUPLICATES_EXCLUDED = {
    "01-ai/Yi-1.5-34B-Chat-16K": "same model as Yi-1.5-34B-Chat, longer context",
    "microsoft/Phi-3-small-128k-instruct": "same model as Phi-3-small-8k-instruct",
}


def _facts() -> dict:
    v = list(COHORT.values())
    mm = [x[0] for x in v]
    sz = [x[1] for x in v]
    return {
        "n": len(COHORT),
        "band": list(BAND),
        "widening_used": WIDENING_USED,
        "mmlu_range": [min(mm), max(mm)],
        "mmlu_spread": max(mm) - min(mm),
        "orgs": sorted({x[2] for x in v}),
        "n_orgs": len({x[2] for x in v}),
        "params_range": [min(sz), max(sz)],
        "params_fold_spread": max(sz) / min(sz),
        "already_swept": sorted(k for k, x in COHORT.items() if x[3]),
        "all_in_band": all(BAND[0] <= x[0] < BAND[1] for x in v),
    }


JUSTIFICATION = _facts()

COHORT_HASH = hashlib.sha256(
    json.dumps({"cohort": {k: list(v) for k, v in sorted(COHORT.items())},
                "band": list(BAND), "justification": JUSTIFICATION},
               sort_keys=True, default=str).encode()).hexdigest()

#: Set once the cohort is committed. Every 2b entry point asserts it.
PINNED_COHORT_HASH = "4ba8e92c5a45f603aee1e8d9bbfe0b962b2d1d5f93bb45b182c42d230c103b6d"


def assert_cohort() -> None:
    """Refuse to run if the band or its membership has moved."""
    if COHORT_HASH != PINNED_COHORT_HASH:
        raise SystemExit(
            "REFUSING TO RUN: the axis-2b cohort or its band has changed.\n"
            f"  pinned:  {PINNED_COHORT_HASH}\n  actual:  {COHORT_HASH}\n"
            "Do not update the pin. Establish what moved, and whether any "
            "break-out rate had been computed before it moved.")


def assert_disjoint_from_seal() -> None:
    """2b recruits outside the sealed cohort; it does not reopen the seal."""
    overlap = sorted(set(COHORT) & set(S.HELD_OUT))
    if overlap:
        raise SystemExit(
            f"SEAL BREACH: axis-2b cohort contains held-out models {overlap}.")


def assert_in_band() -> None:
    """Every member is in the band by the rule, not by judgement."""
    out = {k: v[0] for k, v in COHORT.items() if not BAND[0] <= v[0] < BAND[1]}
    if out:
        raise SystemExit(
            f"Cohort members outside the frozen band {BAND}: {out}. The band is "
            "the experiment; a member outside it is a selection made by hand.")
