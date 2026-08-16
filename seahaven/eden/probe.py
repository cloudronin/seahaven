"""THE DAILY PROBE FLEET — the seismograph's pin.

**Not a round.** A round is a measurement with a number, a `PINNED_ROUNDN_HASH`
and an end; this is a standing instrument that serves the same grid every day
forever. `run --round N` cannot drive it — that path derives its pin key from
the round number — so the probe carries its own pin and its own serving verb.

It exists because the fleet acquired a second job. It began as surveillance:
does the 2026-08-14 LAT state revert, flatten or deepen, which everything sealed
waits on. It is now also the **SCHEDULER** — `#110`'s precondition requires the
state to be expressed, so a mechanism probe cannot fire on a quiet day, and the
daily LAT cell is what grants permission to spend. A skipped fleet day is a day
the science cannot fire. **The fleet's cadence bounds the programme's reaction
time, not merely its detection latency.**

---

**WHAT THE PREMISE CHECKS REFUTED, BEFORE ANY CODE.**

1. **The cohort rule does not select four.** "Cheap tier with clean pre-event
   anchors at both LAT and W2, disjoint from round 16's fourteen" selects NINE,
   and cheapness cannot cut it to four: `gemma-4-31B` (0.39/0.97) undercuts
   `Llama-3.3-70B` (1.04/1.04) on both axes. One of the nine is on another
   provider. **All eight Together-eligible are served** — the rule taken
   literally, no judgement inserted, and more anchored channels is more pooled
   power.
2. **The Flash anchor rule selected a single sitting.** See `FLASH_*` below.
3. **The multi-provider pilot cannot start.** Only Together has a key. See
   `PROVIDERS_WAITING`.

---

**AN ANCHOR FOR A DAILY BETWEEN-DAY VERDICT HAS TWO REQUIREMENTS: AGREEMENT,
AND SPANNING AT LEAST TWO OCCASIONS.**

The specified rule — "most recent >=2 agreeing blocks", agreement by Fisher —
tested only the first and selected blocks [4, 5]. Those are `e11tA` and `e11tB`,
served **79 seconds apart**. That baseline fails the second requirement *by
construction*: it cannot contain between-occasion variance because it never
crossed an occasion, so a daily verdict against it would call ordinary day-to-day
movement an EVENT. This is the contaminated-baseline class caught a third time,
and the first time before it cost anything.

Widened to Wilson-interval overlap, the chain is blocks [2, 3, 4, 5] — four
occasions, pooling to 69/96.

**AND THE POOLED FIGURE STILL UNDERSTATES THE CHANNEL'S OWN SPREAD.** Four
occasions collapsed into one binomial gives a Wilson band of 0.177, while the
blocks themselves range across 0.292. So EVENT on this channel requires landing
outside `FLASH_ENVELOPE`, not merely outside the pooled band, and the envelope
prints beside every verdict. The channel's known life is step-like; the anchor
encodes where it has actually lived.
"""

from __future__ import annotations

from pathlib import Path

from ._shared import pinning as P

_ROOT = Path(__file__).resolve().parents[2]

# --- the instrument ----------------------------------------------------------

LEVELS = ("LAT", "W2")
ARM = "A0"
EPISODES = 24

#: **Uncorrected and deliberately tighter than the register's 0.05.** A daily
#: cadence runs ~365 tests per channel per year, so 0.05 would manufacture ~18
#: EVENT-days a year per channel from noise alone. At 0.01 it is ~3.7, which is
#: the number the dataset card publishes beside the event count.
#: `register/occasion_health.LIVE_ALPHA` stays 0.05 — a different instrument on
#: a different cadence. They must not be refactored into one constant.
ALPHA = 0.01

#: Rolling window, in QUIET days. EVENT days never enter it (the taint law).
ROLLING_K = 5

#: No QUIET refresh in this many days -> the channel prints STALE, and STALE is
#: HOLD for event-conditioned probes.
STALE_AFTER_DAYS = 7

# --- the cohort, by rule ------------------------------------------------------

#: **The whole eligible set, not a selection.** Every model with clean
#: pre-event (<= 2026-08-13) A0 anchors at BOTH LAT and W2, disjoint from round
#: 16's fourteen, reachable on one endpoint. Prices are `(prompt, completion)`
#: USD per million, read off committed cell metadata rather than retyped.
COHORT = {
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
    "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
    "google/gemma-4-31B-it": (0.39, 0.97),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
    "meta-models/Muse-Glimmer-30B": (0.35, 1.50),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
}

COHORT_RULE = (
    "Every model with clean pre-event (<= 2026-08-13) A0 anchors at BOTH LAT "
    "and W2, disjoint from round 16's fourteen, reachable on one endpoint. "
    "Nine qualify; gpt-5.6-terra is excluded ONLY as a different provider, "
    "which makes it the obvious cross-provider control once keys exist. The "
    "spec expected four and named 'cheap tier' as the cut — that cannot "
    "produce them, since gemma-4-31B undercuts Llama-3.3-70B on both price "
    "axes, so the rule is taken literally instead and all eight are served.")

#: The frozen anchor cells, BY CELL ID. Reconstructed at build from the corpus
#: with event sweeps excluded; a test rebuilds them rather than trusting these.
ANCHOR_CELLS = {
    "LAT": {
        "MiniMaxAI/MiniMax-M3": "eden_e10_MiniMaxAI__MiniMax-M3__A0__LAT.json",
        "Qwen/Qwen2.5-7B-Instruct-Turbo":
            "eden_e10_Qwen__Qwen2.5-7B-Instruct-Turbo__A0__LAT.json",
        "Qwen/Qwen3.5-9B": "eden_e10_Qwen__Qwen3.5-9B__A0__LAT.json",
        "deepcogito/cogito-v2-1-671b":
            "eden_e12_deepcogito__cogito-v2-1-671b__A0__LAT.json",
        "google/gemma-4-31B-it": "eden_e12_google__gemma-4-31B-it__A0__LAT.json",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo":
            "eden_e12_meta-llama__Llama-3.3-70B-Instruct-Turbo__A0__LAT.json",
        "meta-models/Muse-Glimmer-30B":
            "eden_e10_meta-models__Muse-Glimmer-30B__A0__LAT.json",
        "nvidia/nemotron-3-ultra-550b-a55b":
            "eden_e12_nvidia__nemotron-3-ultra-550b-a55b__A0__LAT.json",
    },
    "W2": {
        "MiniMaxAI/MiniMax-M3": "eden_e11_MiniMaxAI__MiniMax-M3__A0__W2.json",
        "Qwen/Qwen2.5-7B-Instruct-Turbo":
            "eden_e11_Qwen__Qwen2.5-7B-Instruct-Turbo__A0__W2.json",
        "Qwen/Qwen3.5-9B": "eden_e11_Qwen__Qwen3.5-9B__A0__W2.json",
        "deepcogito/cogito-v2-1-671b":
            "eden_e11_deepcogito__cogito-v2-1-671b__A0__W2.json",
        "google/gemma-4-31B-it": "eden_e11_google__gemma-4-31B-it__A0__W2.json",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo":
            "eden_e11_meta-llama__Llama-3.3-70B-Instruct-Turbo__A0__W2.json",
        "meta-models/Muse-Glimmer-30B":
            "eden_e11_meta-models__Muse-Glimmer-30B__A0__W2.json",
        "nvidia/nemotron-3-ultra-550b-a55b":
            "eden_e11_nvidia__nemotron-3-ultra-550b-a55b__A0__W2.json",
    },
}

#: Pooled epoch anchors, `(ate, n)`, event sweeps excluded. Muse's W2 cell is
#: n=18, not 24, which is why W2's denominator is 186 rather than 192.
#: **Reconstructed from the corpus, never typed** — the first draft of this line
#: carried (189, 190) for W2, a number from nowhere, and the reconstruction test
#: caught it before the pin was frozen. That is the test's whole job.
EPOCH_ANCHOR = {"LAT": (183, 192), "W2": (165, 186)}
EPOCH_DAY = "2026-08-13"

# --- the decision channel -----------------------------------------------------

DECISION_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DECISION_LEVEL = "LAT"
DECISION_ARM = "A1"

#: `(ate, n)` per block, oldest first. Block 1 is the known prior event.
FLASH_BLOCKS = ((27, 72), (19, 24), (21, 24), (14, 24), (15, 24))
FLASH_ANCHOR_BLOCKS = (2, 3, 4, 5)
FLASH_ANCHOR = (69, 96)              # 0.7188, Wilson [0.6217, 0.7989]

#: **EVENT on this channel must land outside THIS, not merely outside the
#: pooled Wilson band.** 0.292 wide against the pooled band's 0.177 — the gap is
#: the point, and it prints beside every verdict.
FLASH_ENVELOPE = (0.5833, 0.8750)

FLASH_ANCHOR_RULE = (
    "Most recent blocks that AGREE (Wilson intervals overlap) AND span at "
    "least two occasions. The specified rule tested agreement only, by Fisher, "
    "and selected blocks [4,5] — e11tA and e11tB, served 79 SECONDS APART. A "
    "baseline from one sitting cannot contain between-occasion variance, so a "
    "daily verdict against it calls ordinary movement an EVENT. Widened to "
    "interval overlap the chain is [2,3,4,5], four occasions. The pooled figure "
    "still understates the spread, so FLASH_ENVELOPE bounds the verdict.")

# --- providers ----------------------------------------------------------------

PROVIDERS = {
    "together": {"base_url": "https://api.together.xyz/v1",
                 "key_env": "TOGETHER_API_KEY", "substrate": "gpu_cloud",
                 "catalogued": True},
}

#: **The pilot's headline waits on an action only the operator can take.**
PROVIDERS_WAITING = (
    "Fireworks, DeepInfra and SambaNova are specified and NOT built in: their "
    "keys do not exist, so their catalogs cannot be resolved and their columns "
    "cannot be served. The verbs are provider-parameterised and the matrix "
    "lives here, so adding one is a pin boundary rather than a code change. "
    "Nothing is lost by starting Together-only — every column begins NO-ANCHOR "
    "regardless, so a column started later is not started worse. But the "
    "CROSS-PROVIDER COINCIDENCE READ, which is the pilot's headline and the "
    "only thing that separates 'this stack moved' from 'the GPU ecosystem "
    "moved', does not exist until those keys do.")

#: Levels are NEVER compared across providers — same weights on a different
#: stack is a different served artifact. Only within-provider deltas and
#: same-day coincidence of events are read.
LEVELS_RULE = (
    "Cross-provider LEVEL differences are documentation, never findings. Only "
    "within-provider deltas and cross-provider coincidence of EVENTS are read.")

# --- seeds, budget, status ----------------------------------------------------

#: Date-derived, from a reserved block sized for the FOUR-provider shape so
#: adding a column later needs no new block. Disjoint from all 21 burned ranges;
#: the highest burned seed in the corpus is 25023.
SEED_BASE = 100_000
SEED_STRIDE = 864                    # per day, across four providers
SEED_BLOCK = (100_000, 151_839)      # 60 days of headroom
SEED_EPOCH = "2026-08-16"

#: **Measured, not estimated.** From committed billing on these exact models at
#: these exact worlds: LAT A0 x 8 = $3.67 (round 18 actuals), W2 A0 x 8 = $4.07
#: (round 17 actuals), Flash A1 at LAT = $0.11. The spec sized $2.50/day for
#: four models; the eight-model decision changes it materially, and a wrong cost
#: inside a hashed payload gets quoted later as though it were derived.
DAILY_EPISODES = 408
DAILY_USD = 7.86
PILOT_DAYS = 30
PILOT_USD = 235.79                   # $44 slack under the gate; retries eat it
BUDGET_PER_DAY = 12.00
BUDGET_PILOT = 280.00

#: `VERDICT_FAIL` is not in the spec's enum and is added deliberately: a day
#: where serving succeeded and verdict computation failed must never read as
#: QUIET, and must be distinguishable from a provider outage.
STATUS = ("OK", "PARTIAL", "SERVE_FAIL", "VERDICT_FAIL", "BUDGET_REFUSED")

GATE_RULES = {
    "spend": "Together row OK AND Together LAT verdict-vs-rolling QUIET -> "
             "PROCEED; EVENT or STALE or SERVE_FAIL -> HOLD with the reason",
    "event-probe": "Together LAT verdict EVENT-down vs epoch REQUIRED, else "
                   "VOID-NO-CONTRAST and spend nothing",
    "fork-reopen": "Together fleet LAT QUIET licenses SERVING round 16's "
                   "re-serve; the re-serve's own anchored-six certification "
                   "still decides reading. The fleet is a tripwire, not a "
                   "judge — REFERENCE_DISJOINT holds",
}

DAY30_CRITERIA = (
    "Per column, keep daily / cut / reduce to MWF, decided by (a) events "
    "detected WITH the false-alarm expectation beside them, (b) coincidence "
    "table yield, (c) anchor stability — did the column reach and hold QUIET "
    "baselines at all. NO COLUMN IS JUDGED ON ITS LEVELS.")

#: Probe cells carry their own tag and never pool into a score corpus. The
#: exclusion is structural, not disciplinary: `_shared.occasion.observations()`
#: filters on the `eden_e*` filename schema, which this tag cannot match.
CELL_TAG = "probe-{provider}-{date}"

ARTIFACTS = (
    "seahaven/eden/simulate.py",
    "seahaven/eden/outcome.py",
    "seahaven/eden/manifest.py",
    "seahaven/eden/crossing.py",
    "seahaven/eden/conditioning.py",
    "seahaven/eden/intent.py",
)

#: Computed on a clean tree, BEFORE any cell was served.
PINNED_PROBE_HASH = "39a67858c3c113c62c50573a649bafa56a47c3e3c09b09316089ed2aad6f4f19"


def world_lock_paths() -> tuple[str, ...]:
    return P.world_lock_paths(LEVELS)


def cells(provider: str = "together"):
    """The day's grid: A0 at both worlds for the cohort, plus the decision cell."""
    out = [(m, ARM, lv) for m in COHORT for lv in LEVELS]
    out.append((DECISION_MODEL, DECISION_ARM, DECISION_LEVEL))
    return out


def seed_for(provider: str, model: str, level: str, arm: str, day: int) -> int:
    """Same date -> same seeds, per provider. Deterministic and reproducible."""
    idx = sorted({(m, a, lv) for m, a, lv in cells()}).index((model, arm, level))
    off = sorted(PROVIDERS).index(provider) * (len(cells()) * EPISODES)
    return SEED_BASE + day * SEED_STRIDE + off + idx * EPISODES


def _payload_body(art: dict, locks: dict, specs: dict) -> str:
    import json
    return json.dumps({
        "levels": LEVELS, "arm": ARM, "episodes": EPISODES,
        "alpha": ALPHA, "rolling_k": ROLLING_K,
        "stale_after_days": STALE_AFTER_DAYS,
        "cohort": {k: list(v) for k, v in sorted(COHORT.items())},
        "cohort_rule": COHORT_RULE,
        "anchor_cells": ANCHOR_CELLS,
        "epoch_anchor": {k: list(v) for k, v in sorted(EPOCH_ANCHOR.items())},
        "epoch_day": EPOCH_DAY,
        "decision": {"model": DECISION_MODEL, "level": DECISION_LEVEL,
                     "arm": DECISION_ARM,
                     "blocks": [list(b) for b in FLASH_BLOCKS],
                     "anchor_blocks": list(FLASH_ANCHOR_BLOCKS),
                     "anchor": list(FLASH_ANCHOR),
                     "envelope": list(FLASH_ENVELOPE),
                     "rule": FLASH_ANCHOR_RULE},
        "providers": PROVIDERS, "providers_waiting": PROVIDERS_WAITING,
        "levels_rule": LEVELS_RULE,
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE,
                  "block": list(SEED_BLOCK), "epoch": SEED_EPOCH},
        "cost": {"daily_episodes": DAILY_EPISODES, "daily_usd": DAILY_USD,
                 "pilot_days": PILOT_DAYS, "pilot_usd": PILOT_USD,
                 "budget_per_day": BUDGET_PER_DAY,
                 "budget_pilot": BUDGET_PILOT},
        "status": STATUS, "gate_rules": GATE_RULES,
        "day30_criteria": DAY30_CRITERIA, "cell_tag": CELL_TAG,
        "artifacts": art, "locks": locks, "worldspec": specs,
    }, sort_keys=True)


def payload() -> str:
    return _payload_body(P.digest_files(_ROOT, ARTIFACTS),
                         P.digest_files(_ROOT, world_lock_paths()),
                         P.worldspec_digest(LEVELS))


def current_hash() -> str:
    return P.hash_payload(payload())


def assert_pinned() -> None:
    P.assert_hash(current_hash(), PINNED_PROBE_HASH, "PROBE",
                  empty_hint="probe pin is EMPTY. Compute, paste, commit "
                             "FIRST — before any cell is served.")


if __name__ == "__main__":
    print(current_hash())
