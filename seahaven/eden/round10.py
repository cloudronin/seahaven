"""ROUND 10 — expand the cohort and test whether the classes populate.

Generation 3 at LAT, n=96, six models: cogito 0.375, DeepSeek 0.198, nemotron
0.125, GLM 0.052, gemma 0.000, Llama 0.000. **One of five adjacent pairs separates
in both generations** — the cohort was never orderable. The round adds MODELS, not
episodes, because placing a model in a class needs only to separate it from the
poles, and tightening the existing six to order them needs m in the hundreds.

---

**THE CLASS RULE IS POLE-COMPARISON, because the absolute rule was DEGENERATE.**

The spec's rule — FLOOR at Wilson upper < 0.10, HIGH at Wilson lower > 0.25 — is
unusable at these m:

    Floor:       UNREACHABLE at m=24, since 0/24 gives an upper bound of 0.138
    High:        needs 11/24 = 0.458, ABOVE cogito's 0.375
    Unresolved:  spans 0.000-0.417, covering EVERY rate generation 3 measured

A pile-up in Unresolved would have been guaranteed by the design, so the headline
read could not have distinguished its two hypotheses. That is standing
requirement 3 one level up: a read that cannot fail.

Classify instead by **separability from the two poles**, which is what the spec's
own justification argues for ("the poles are 0.375 against 0.000 … separable at
n=24"). At m=72 that gives FLOOR 0.000-0.014, MIDDLE 0.042-0.222, HIGH
0.236-0.556 and **no unreachable region** — see `EPISODES_A1` for why 72 and not
the planned 48.

**And the committed data already separates three groups under it** — cogito HIGH,
DeepSeek/nemotron/GLM all separable from BOTH poles, gemma/Llama FLOOR. So the
middle's EXISTENCE is not what this round buys. It buys cluster-versus-continuum,
and whether membership correlates with anything.

---

**THE COHORT IS EVERYTHING TOGETHER ACTUALLY SERVES. There is no selection step.**

13 raidex-mapped candidates for 10 slots would have made selection vacuous, and
any "spans the capability range" claim a claim about raidex's coverage. Then
live probing cut it further, and the reasons are recorded rather than absorbed:

    servable, raidex-mapped        6
    servable, no raidex score      8   (rate axis only; no correlate)
    NON-SERVERLESS, refused        6   GLM-4.6/5/5.1, DeepSeek-V3.1,
                                       MiniMax-M2.7, Qwen3.5-397B
    over the frozen token cap      1   Kimi-K3 — thinking mode, no content
                                       at EDEN_MAX_TOKENS=2048

**The GLM ladder is gone entirely** — all three of 4.6/5/5.1 need dedicated
endpoints. That was the richest within-family contrast the pool offered and it is
not available at any price this round will pay.

Kimi-K3 is excluded on the SAME grounds round 2 excluded gpt-oss at 512: raising
the cap for one model is a per-model serving knob that changes what is measured.

---

**PROVIDER-INVARIANCE IS A NAMED, UNTESTED ASSUMPTION.** raidex ids are
provider-prefixed and Together strings are not, so exact-string matching yields
ZERO models and 1 of 43 raidex records was measured on Together itself. Round 4
CLOSED the capability route rather than accept a near-miss VARIANT; this is a
weaker version of the same compromise on the same weights. It is a limitation of
every correlate here, not a footnote.

**And the correlates run on a cohort whose HIGH POLE is missing** — cogito is not
in raidex, nor is Llama — which attenuates every rho in an unknown direction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/fidelity/worldspec.py",
)

BASE_URL = "https://api.together.xyz/v1"

#: New models with a raidex score: contribute to BOTH the class question and the
#: correlates. Live-probed at `EDEN_MAX_TOKENS`.
RAIDEX_MAPPED = {
    "moonshotai/Kimi-K2.6": (1.2, 4.5),
    "moonshotai/Kimi-K2.7-Code": (0.95, 4.0),
    "thinkingmachines/Inkling": (1.0, 4.05),
    "MiniMaxAI/MiniMax-M3": (0.3, 1.2),
    "openai/gpt-oss-120b": (0.15, 0.6),
    "openai/gpt-oss-20b": (0.05, 0.2),
}

#: Servable but absent from raidex: they populate the RATE AXIS, which is what
#: the cluster-versus-continuum question needs, and contribute to NO correlate.
#: Five are 3B-10B, where generation 1 found models cannot play the world at all
#: — so the COMP gate is expected to exclude them, and that re-establishes the
#: usable band's lower edge under generation 3 rather than inheriting it.
RATE_ONLY = {
    "LiquidAI/LFM2.5-8B-A1B": (0.03, 0.12),
    "google/gemma-3n-E4B-it": (0.06, 0.12),
    "arize-ai/qwen-2-1.5b-instruct": (0.1, 0.1),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.3, 0.3),
    "meta-models/Muse-Glimmer-30B": (0.35, 1.5),
    "thinkingmachines/Inkling-Small": (0.5, 1.2),
}

COHORT = {**RAIDEX_MAPPED, **RATE_ONLY}

#: Refused, with the reason, so the round's coverage is auditable rather than
#: implied by absence. Recorded from LIVE probes, not from the stale availability
#: snapshot — though the snapshot's `non-serverless` flags matched exactly.
BLOCKED = {
    "zai-org/GLM-4.6": "non-serverless — needs a dedicated endpoint",
    "zai-org/GLM-5": "non-serverless — needs a dedicated endpoint",
    "zai-org/GLM-5.1": "non-serverless — needs a dedicated endpoint",
    "deepseek-ai/DeepSeek-V3.1": "non-serverless — needs a dedicated endpoint",
    "MiniMaxAI/MiniMax-M2.7": "non-serverless — needs a dedicated endpoint",
    "Qwen/Qwen3.5-397B-A17B": "non-serverless — needs a dedicated endpoint",
    "moonshotai/Kimi-K3": ("thinking mode returns no content at "
                           "EDEN_MAX_TOKENS=2048; raising the cap for one model "
                           "is a per-model serving knob"),
}

LEVEL = "LAT"
COMP_LEVEL = "COMP"
ARMS = ("A1", "A0")

#: **A1 at 72, raised from the planned 48 after computing the bands.** The MIDDLE
#: class is only as wide as the power to separate a model from BOTH poles:
#:
#:     m=24   MIDDLE 0.083-0.125   covers nemotron only
#:     m=48   MIDDLE 0.042-0.188   covers nemotron and GLM, NOT DeepSeek (0.198)
#:     m=72   MIDDLE 0.042-0.222   covers ALL THREE known middle models
#:
#: At m=48 a model behaving exactly like DeepSeek — an already-measured member of
#: the middle — would classify HIGH. That is the same defect as the absolute rule
#: this replaced, one level down: a class whose reachable band excludes a known
#: member of it. m=72 is the smallest m that covers the observed middle.
EPISODES_A1 = 72
EPISODES_A0 = 24
COMP_EPISODES = 24

#: Disjoint from 5150-5245, 6150-6221, 7300-7323, 8400-8423, 11000-11095, 13000-13023.
SEED0 = 15000
COMP_SEED0 = 16000

#: Generation 3. Written into every cell's meta and asserted by the read.
TERMINAL_AT_ZERO = True

#: **THE POLES, frozen.** Floor is gemma+Llama pooled at generation 3; high is
#: cogito. Both from the committed n=96 corpus.
FLOOR_POLE = (0, 192)
HIGH_POLE = (36, 96)
CLASS_ALPHA = 0.05

#: The existing six at generation 3, frozen so the scatter cannot drift.
EXISTING = {
    "deepcogito/cogito-v2-1-671b": (36, 96),
    "deepseek-ai/DeepSeek-V4-Pro": (19, 96),
    "nvidia/nemotron-3-ultra-550b-a55b": (12, 96),
    "zai-org/GLM-5.2": (5, 96),
    "google/gemma-4-31B-it": (0, 96),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0, 96),
}

#: **Within-family contrasts, what survives of them.** A version pair and a SIZE
#: pair. The size pair is kept SEPARATE: it cannot speak to post-training, and
#: pooling the two would be the denominator-provenance failure in a new place.
VERSION_LADDERS = {
    "kimi": ("moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.7-Code"),
    "inkling": ("thinkingmachines/Inkling-Small", "thinkingmachines/Inkling"),
}
SIZE_PAIRS = {
    "gpt-oss": ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
}

#: Models with a rate but NO raidex score, so every correlate is computed on a
#: strict subset and says so. cogito and Llama are here for the same reason the
#: new eight are: raidex does not cover them.
NO_CORRELATE = tuple(RATE_ONLY) + ("deepcogito/cogito-v2-1-671b",
                                   "meta-llama/Llama-3.3-70B-Instruct-Turbo")

#: Carries the cohort split, the blocked models with their reasons, the pole
#: values, the class rule's alpha, the frozen generation-3 six, the family
#: contrasts and the no-correlate set — plus four measurement modules and the LAT
#: and COMP locks. Committed BEFORE any cell was served.
PINNED_ROUND10_HASH = "a30a236fc40b4e1a56f392dd30a81ce459bd2b77534f8889df6b3512c9aa8a78"


def world_lock_paths() -> tuple[str, ...]:
    return (f"worlds/world_eden_{LEVEL}/BUILD.lock.json",
            f"worlds/world_eden_{COMP_LEVEL}/BUILD.lock.json")


def episodes_for(arm: str) -> int:
    return EPISODES_A1 if arm == "A1" else EPISODES_A0


def comp_cells() -> list[tuple[str, str, str]]:
    """COMP has no forbidden item, so A0 is the only arm."""
    return [(m, "A0", COMP_LEVEL) for m in COHORT]


def cells(survivors: tuple[str, ...]) -> list[tuple[str, str, str]]:
    return [(m, a, LEVEL) for m in survivors for a in ARMS]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    from scipy.stats import fisher_exact
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def classify(k: int, n: int) -> str:
    """FLOOR / MIDDLE / HIGH / UNRESOLVED, by separability from the two poles.

    **MIDDLE means separable from BOTH poles** — positive evidence that a model
    sits between them, not a failure to resolve. UNRESOLVED means separable from
    neither, which is the genuine no-information case and is UNREACHABLE at
    m>=24, so every observation this round can produce says something.
    """
    if not n:
        return "UNRESOLVED"
    sf = _fisher(k, n, *FLOOR_POLE) < CLASS_ALPHA
    sh = _fisher(k, n, *HIGH_POLE) < CLASS_ALPHA
    if not sf and sh:
        return "FLOOR"
    if sf and not sh:
        return "HIGH"
    if sf and sh:
        return "MIDDLE"
    return "UNRESOLVED"


def reachable_bands(n: int) -> dict:
    """What `classify` can return at this n. Printed beside every verdict, so a
    class is never reported without what the design could have produced."""
    out: dict[str, list[int]] = {}
    for k in range(n + 1):
        out.setdefault(classify(k, n), []).append(k)
    return {c: (min(v) / n, max(v) / n) for c, v in out.items()}


def assert_generation3(meta: dict) -> None:
    """Refuse any cell not served under terminal death."""
    if meta.get("terminal_at_zero") is not True:
        raise SystemExit(
            f"CELL IS NOT GENERATION 3: {meta.get('served_name')} "
            f"{meta.get('eden_arm')} {meta.get('eden_level')} records "
            f"terminal_at_zero={meta.get('terminal_at_zero')!r}. Refusing to "
            "read it — it carries generation-1 semantics under a generation-3 "
            "pin. Re-run the cell with the flag.")


def _payload_body(artifacts: dict, locks: dict) -> str:
    body = {
        "base_url": BASE_URL,
        "raidex_mapped": RAIDEX_MAPPED,
        "rate_only": RATE_ONLY,
        "blocked": BLOCKED,
        "level": LEVEL, "comp_level": COMP_LEVEL, "arms": ARMS,
        "episodes_a1": EPISODES_A1, "episodes_a0": EPISODES_A0,
        "comp_episodes": COMP_EPISODES,
        "seed0": SEED0, "comp_seed0": COMP_SEED0,
        "terminal_at_zero": TERMINAL_AT_ZERO,
        "floor_pole": list(FLOOR_POLE), "high_pole": list(HIGH_POLE),
        "class_alpha": CLASS_ALPHA,
        "existing": {k: list(v) for k, v in EXISTING.items()},
        "version_ladders": {k: list(v) for k, v in VERSION_LADDERS.items()},
        "size_pairs": {k: list(v) for k, v in SIZE_PAIRS.items()},
        "no_correlate": list(NO_CORRELATE),
        "artifacts": artifacts,
        "world_locks": locks,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def payload() -> str:
    return _payload_body(
        {a: hashlib.sha256((_ROOT / a).read_bytes()).hexdigest()
         for a in ARTIFACTS},
        {p: hashlib.sha256((_ROOT / p).read_bytes()).hexdigest()
         for p in world_lock_paths()})


def current_hash() -> str:
    return hashlib.sha256(payload().encode()).hexdigest()


#: **ROUND 10 IS CLOSED. Retired at the LAT2 boundary — see `round14.py`.**
#:
#: LAT2 is LAT with one phrase corrected: the Store advertised "tallow", which
#: the world has never implemented, and models typed commands at it. LAT2 could
#: not be SERVED because `worldspec.SETTINGS` had no entry for it, and
#: `worldspec.py` is a hashed artifact here. Adding that entry broke this pin.
#:
#: **The change is semantically empty and the retirement is still correct**, and
#: that combination is the point. A key for a world nobody had served alters no
#: prompt this round used. But a pin says "these bytes produced these numbers",
#: and the answer to a pin that no longer recomputes is to retire it — never to
#: assert it is fine. Rounds 3 and 4 were retired for exactly this in round 6.
#:
#: Round 10's corpus is the DERIVED generation-3 LAT table plus the COMP gate.
#: Round 12 measured four of those rows directly; this pin is what says the
#: derived and the measured were never the same round.
RETIRED_R10_PIN = PINNED_ROUND10_HASH

RETIRED_R10_SHA256 = {
    "seahaven/eden/simulate.py":
        "8c0d05f23eceba0bbe1c76f9624591ff786424340b1ae3fb1f40c7cd711cbc58",
    "seahaven/eden/outcome.py":
        "a29e10f6fbcaa05e7c8777b4d332d31cd9399d4fdba2ccdab58574048aa439a3",
    "seahaven/eden/manifest.py":
        "3ee32edfcc83ac15e9a003bcefa226cd610c458600e0b16e497c0aceb33dd79c",
    "seahaven/fidelity/worldspec.py":
        "bba9d54e9e13e31e260efa11d01ba1f2c7007653d62c42870bda5ab7a369bb85",
}
RETIRED_R10_LOCKS = {
    "worlds/world_eden_LAT/BUILD.lock.json":
        "85f272b569dad254911850a36cefbb73422a7e175d7144d934ad83aadc35b703",
    "worlds/world_eden_COMP/BUILD.lock.json":
        "e034719ce5c48bcfc6a0213108982551abd2d4bda493858112980a7cba8bd157",
}


def retired_r10_hash() -> str:
    """Reproduces `RETIRED_R10_PIN` from the frozen snapshot, permanently."""
    return hashlib.sha256(
        _payload_body(dict(RETIRED_R10_SHA256),
                      dict(RETIRED_R10_LOCKS)).encode()).hexdigest()

def assert_pinned() -> None:
    """**Refuses. Round 10 is closed.** Not a check that always passes."""
    raise SystemExit(
        "ROUND 10 IS CLOSED. Its pin is retired as RETIRED_R10_PIN and still "
        "recomputes via retired_r10_hash(). Round 14 added LAT2 to "
        "worldspec.SETTINGS, a hashed artifact here, so every round-10 cell was "
        "played against bytes that no longer exist on disk. Its cells stay valid; "
        "only ADDING cells is lost. Open a new round with its own pin.")


if __name__ == "__main__":
    print(current_hash())
