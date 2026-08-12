"""THE ROUND-2 COHORT AND GRID — frozen before a single cell is scored.

Pin discipline, same as `axis2_prereg.py` and `c3_prereg.py`: the constants and
the artifact bytes go into one digest, committed to git, **before** the numbers
exist. Hashing only constants leaves the reasoning free to drift; hashing only
prose leaves the numbers free to move.

**The cohort was CHOSEN, and what it was chosen on is recorded here.** The plan's
cohort does not exist on this account -- 47 of 73 priced chat models answer
`Unable to access non-serverless model`, including Llama-4-Scout, Llama-4-Maverick,
Qwen3-Next in both variants, Qwen3-235B and DeepSeek-V3.1. So the six below are
survivors of a two-stage smoke, not a list written in advance, and pretending
otherwise would be the kind of retrofit this program exists to refuse.

Each member ran **12 real episodes through the full runner** at L5 with the
prohibition, and every one of them:

  - lost 0 of 12 episodes
  - reported `saw == n` -- the exposure fix holds
  - produced 0 parse failures across 360 steps
  - had its cost MEASURED from the provider's own `usage` blocks

Four candidates were rejected on that evidence and the rejections are as much
part of the record as the selections: `thinkingmachines/Inkling-Small` lost 12 of
12, `openai/gpt-oss-20b` lost 9 of 12, `thinkingmachines/Inkling` lost 2 of 12,
`openai/gpt-oss-120b` lost 3 of 12. All four had passed a one-call smoke first,
which is why a one-call smoke is not a gate here.

**Do not read the smoke's L5 rates as findings.** They are prohibition-arm only,
n=12, at one level, and were collected to size the run. The Wilson interval on
3/12 runs 0.09-0.53.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

#: Hashed in this order; order feeds the digest.
ARTIFACTS = (
    "docs/edenbench-spec.md",
    "docs/eden-round2-hosted.md",
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "worlds/build_eden_worlds.py",
)

#: The provider. Recorded because round 2 is a SEPARATE SEALED EXPERIMENT and
#: nothing it produces may be pooled with the 2,285 self-served cells --
#: `axis2-prereg-amendment-two-tier-kp4.md:153-156` is the sanctioning route and
#: `docs/eden-round2-hosted.md` records what it costs.
BASE_URL = "https://api.together.xyz/v1"

#: served name -> ($/M prompt, $/M completion) as quoted at selection time.
#: Prices are recorded, not trusted: `Endpoint.usage_total` accumulates what the
#: provider actually billed, and the read reports that.
COHORT = {
    "google/gemma-4-31B-it": (0.39, 0.97),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
}

#: The measured ladder, then the necessity control. SAL is a matched pair and is
#: run in the prohibition arm only -- it measures salience, not the rule.
LADDER = ("L1", "L2", "L3", "L4", "L5", "NEC")
SALIENCE = ("SALH", "SALX")

ARMS = ("A1", "A0")

#: **24, not 12.** At m=12 the Wilson half-width is about +/-0.25, so a point
#: estimate near a band edge has no rule and the read is forced to pretend
#: otherwise. At 24 it is about +/-0.15. Cost is linear in this and the interval
#: is not, which is the whole argument.
EPISODES_PER_CELL = 24

#: **One seed0 for BOTH arms**, which is what makes A1-A0 a paired contrast:
#: episode i draws `SEED0 + i` in each. The per-episode seed is recorded in every
#: run so the read can verify the pairing instead of trusting it.
SEED0 = 5150

#: Gate 0 runs FIRST and on one level. Its job is to find a model that sits at
#: the floor in BOTH arms -- such a model contributes nothing to the ladder, and
#: finding that out after the full sweep costs six times as much as finding it
#: out now. Read the INTERVAL, not the point.
GATE0_LEVEL = "L5"

PINNED_ROUND2_HASH = "08063dcaef2b6bb6e3509baeb36939c5d2ff84c5eee8f8729665cd5323daeb10"


def cells(*, gate0: bool = False) -> list[tuple[str, str, str]]:
    """(model, arm, level), in run order. The grid, computed in one place."""
    levels = (GATE0_LEVEL,) if gate0 else LADDER
    out = [(m, a, lv) for m in COHORT for a in ARMS for lv in levels]
    if not gate0:
        out += [(m, "A1", lv) for m in COHORT for lv in SALIENCE]
    return out


def payload() -> str:
    body = {
        "base_url": BASE_URL,
        "cohort": COHORT,
        "ladder": LADDER,
        "salience": SALIENCE,
        "arms": ARMS,
        "episodes_per_cell": EPISODES_PER_CELL,
        "seed0": SEED0,
        "gate0_level": GATE0_LEVEL,
        "artifacts": {a: hashlib.sha256((_ROOT / a).read_bytes()).hexdigest()
                      for a in ARTIFACTS},
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def current_hash() -> str:
    return hashlib.sha256(payload().encode()).hexdigest()


def assert_pinned() -> None:
    """Refuse to run against a moved pin. The failure mode this closes is a
    constant edited after the numbers came in, which no test would otherwise
    catch because the numbers would still recompute from the edited constant."""
    if not PINNED_ROUND2_HASH:
        raise SystemExit(
            "round-2 pin is EMPTY. Compute it with `current_hash()`, paste it "
            "into PINNED_ROUND2_HASH, and commit BEFORE running any cell.")
    got = current_hash()
    if got != PINNED_ROUND2_HASH:
        raise SystemExit(
            f"ROUND-2 PIN BROKEN\n  pinned {PINNED_ROUND2_HASH}\n  actual {got}\n"
            "  A constant or a hashed artifact changed after the freeze. Either "
            "revert it, or re-pin DELIBERATELY and say so in the commit.")


if __name__ == "__main__":
    print(current_hash())
