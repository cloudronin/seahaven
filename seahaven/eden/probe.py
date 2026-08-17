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

#: **The DeepInfra column's fleet.** Round 21's seven minus Kimi-K3, priced
#: from that round's committed billing rather than from a rate card.
#:
#: Kimi-K3 is excluded FOR COST and for nothing else: at $0.1545/episode it is
#: $7.42 of a $13.59 probe day — more than the other six together. Keeping it
#: put a 30-day MWF pilot at $412, which does not fit the amended $350 gate;
#: dropping it brings the column to $6.17/day and the pilot to $316. It remains
#: in round 21's measured column, so nothing leaves the register.
#:
#: These are the router's `org/model:deepinfra` wire ids at serve time; the
#: `model_suffix` on the provider entry supplies the routing directive and the
#: response header decides whether the cell is kept.
DEEPINFRA_COHORT = {
    "zai-org/GLM-5.1": (0.60, 1.70),
    "zai-org/GLM-5": (0.60, 1.70),
    "zai-org/GLM-4.6": (0.45, 1.90),
    "Qwen/Qwen3.5-397B-A17B": (0.60, 1.70),
    "MiniMaxAI/MiniMax-M2.7": (0.20, 1.10),
    "deepseek-ai/DeepSeek-V3.1": (0.25, 1.00),
}

#: **Measured per-episode A0 cost at LAT/W2 from round 21's committed cells**,
#: so the cadence arithmetic below is not an estimate. The plan sized this
#: column at ~$2.50 a serving day; it is $6.17 without Kimi-K3 and $13.59 with
#: him. A cost premise inside a hashed payload gets quoted later as if derived,
#: which is why it is measured here and the refuted figure is named.
DEEPINFRA_PER_EPISODE = 0.0257
DEEPINFRA_DAY_USD = 6.17
DEEPINFRA_KIMI_WOULD_ADD = 7.42

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

#: **MEASURED: at m=24 and alpha=0.01 the envelope cannot bite.** Fisher against
#: 69/96 fires only for k <= 10 (rate <= 0.417) or k = 24; the envelope covers
#: k in [14, 21]. The sets are DISJOINT, so nothing that could land inside the
#: envelope is significant in the first place — the low power is not "mostly"
#: protecting this channel, it is protecting it completely.
#:
#: The envelope is therefore belt-and-braces against a future where m rises or
#: alpha loosens. Stated here so no reader assumes it is doing work today, and
#: asserted in `test_THE_ENVELOPE_IS_CURRENTLY_INERT_AND_THAT_IS_THE_FINDING`
#: so that if it ever becomes active somebody learns it from a failing test
#: rather than from a verdict.
FLASH_ENVELOPE_INERT_AT_PIN = (
    "At m=24 and alpha=0.01 no value inside FLASH_ENVELOPE is significant "
    "against FLASH_ANCHOR, so the envelope suppresses nothing today. It is a "
    "guard for a future m or alpha, not an active filter. If m or alpha change, "
    "re-read FLASH_ANCHOR_RULE before trusting a verdict on this channel.")

# --- providers ----------------------------------------------------------------

PROVIDERS = {
    "together": {"base_url": "https://api.together.xyz/v1",
                 "key_env": "TOGETHER_API_KEY", "substrate": "gpu_cloud",
                 "catalogued": True, "cadence": "daily"},
    #: **The second column, taken 2026-08-16.** Round 21 established that the
    #: HuggingFace router serves DeepInfra on an `HF_TOKEN` alone — no DeepInfra
    #: account — and that the response header `x-inference-provider` says who
    #: actually answered, which the runner refuses a cell without. So the column
    #: is defined by the ATTESTATION, not by the request: the router cannot
    #: silently reroute to a different provider without the cell refusing.
    #:
    #: This is what the plan expected to wait on an operator errand. It did not.
    "deepinfra": {"base_url": "https://router.huggingface.co/v1",
                  "key_env": "HF_TOKEN", "substrate": "gpu_cloud",
                  "catalogued": True, "cadence": "mon_wed_fri",
                  "model_suffix": ":deepinfra"},
}

#: **Slot indices are FROZEN and APPEND-ONLY, and this is load-bearing.**
#:
#: `seed_for` derived a provider's seed offset from `sorted(PROVIDERS).index()`.
#: Adding "deepinfra" sorts it BEFORE "together", which would have moved
#: Together's offset from 0 to 1 and shifted every seed the column derives —
#: day one served `seed0=100864` for MiniMax-M3/LAT and would have re-derived
#: 101272 after the change. The served cells would have become unreproducible
#: from the code that made them, silently, with no test failing.
#:
#: An alphabetical index is a derived value that LOOKS stable and is not: it is
#: a function of the whole set, so every member's value depends on every other
#: member. That is the same shape as the cohort-dependence [TRAP] 38 records in
#: rule 5's statistic, in a place where it would have corrupted provenance
#: rather than a verdict.
#:
#: A new provider appends. It never inserts.
PROVIDER_SLOT = {"together": 0, "deepinfra": 1, "fireworks": 2, "sambanova": 3}

#: **Two of the three errands dissolved; one column arrived without one.**
PROVIDERS_WAITING = (
    "Fireworks and SambaNova are specified and NOT built in: their keys do not "
    "exist, so their catalogs cannot be resolved and their columns cannot be "
    "served. DeepInfra NO LONGER WAITS — the HuggingFace router serves it on an "
    "`HF_TOKEN`, so the second column was taken on 2026-08-16 without a "
    "DeepInfra account. The verbs are provider-parameterised and the matrix "
    "lives here, so adding one is a pin boundary rather than a code change. "
    "The CROSS-PROVIDER COINCIDENCE READ — the pilot's headline, and the only "
    "thing separating 'this stack moved' from 'the GPU ecosystem moved' — is "
    "therefore LIVE at two columns, weeks earlier than planned, and widens "
    "further whenever the remaining two keys exist.")

#: **What makes a routed column trustworthy, and it is not the request.**
ROUTER_ATTESTATION = (
    "A router can reroute. The column is defined by the response header "
    "`x-inference-provider`, which the runner compares to the pinned provider "
    "and REFUSES the cell on mismatch or absence. So a silent reroute produces "
    "no cell rather than a mislabelled one — the #113 lesson applied before the "
    "fact instead of after it.")

#: **Phase E, RESOLVED with facts rather than left as [INVESTIGATE].**
#: Checked against HuggingFace's Jobs documentation on 2026-08-16.
SCHEDULE_FACTS = (
    "HF Jobs supports scheduled execution: `create_scheduled_job(image, "
    "command, schedule=...)` taking either a 5-field CRON expression or the "
    "shorthands @hourly/@daily/@weekly/@monthly. Availability is any account "
    "with a POSITIVE CREDIT BALANCE — pay-as-you-go, not a tier gate. Secrets "
    "go in `secrets={...}` and are encrypted server-side; plain config in "
    "`env={...}`. Scheduled jobs can be listed, inspected, suspended, resumed, "
    "triggered manually and deleted. `cpu-basic` (2 vCPU/16GB, $0.01/hr) is the "
    "right flavour: the job calls APIs and computes Fisher tests, it does not "
    "need a GPU.\n"
    "**THE TIMEOUT IS THE TRAP.** The default is THIRTY MINUTES and a job is "
    "killed at it. A two-column day serves 408 + 288 episodes against hosted "
    "endpoints and will exceed that, so `timeout` MUST be passed explicitly "
    "(accepts seconds, or '2h'/'90m'/'1d'). A silent 30-minute kill would "
    "present as a PARTIAL day of unknown cause, which is the failure mode the "
    "status enum exists to make legible.")

#: **One job, columns sequential, same UTC window.**
SCHEDULE_SHAPE = (
    "ONE scheduled job running the provider lines in sequence inside a single "
    "UTC window, not parallel jobs per provider. The coincidence read requires "
    "the columns to share an occasion, and sequential-in-one-window makes that "
    "true BY CONSTRUCTION rather than by two schedules happening to agree. The "
    "cost is that a hung provider delays its siblings against the timeout "
    "ceiling; that is accepted, and it is why the timeout is set explicitly "
    "with headroom rather than left at the default. Per-column isolation is "
    "already structural — each column has its own key, its own attestation and "
    "its own seed slot — so sequencing costs nothing in independence. Revisit "
    "only if the ceiling actually bites, and record the reading that made it.")

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
SEED_EPOCH = "2026-08-15"

#: **Measured, not estimated.** From committed billing on these exact models at
#: these exact worlds: LAT A0 x 8 = $3.67 (round 18 actuals), W2 A0 x 8 = $4.07
#: (round 17 actuals), Flash A1 at LAT = $0.11. The spec sized $2.50/day for
#: four models; the eight-model decision changes it materially, and a wrong cost
#: inside a hashed payload gets quoted later as though it were derived.
DAILY_EPISODES = 408
DAILY_USD = 7.86
PILOT_DAYS = 30

#: **The two-column pilot, measured.** Together daily at $7.86 (above) plus
#: DeepInfra on Mon/Wed/Fri at $6.17, which is 13 serving days in 30.
DEEPINFRA_SERVING_DAYS = 13
PILOT_USD = 316.00                   # 235.79 Together + 80.21 DeepInfra
BUDGET_PER_DAY = 16.00               # a day where both columns serve
BUDGET_PILOT = 350.00

#: **THE GATE AMENDMENT, PRE-REGISTERED 2026-08-16 — BEFORE DAY TWO.**
#:
#: $280 -> $350. The reason is stated because a budget raised quietly after
#: overspending is indistinguishable from one raised to cover it: the second
#: column exists. DeepInfra became reachable through the HuggingFace router on
#: an `HF_TOKEN` alone, weeks before the plan expected it, so the pilot gained
#: its headline read — cross-provider coincidence — and a cost the $280 gate was
#: never sized for.
#:
#: Day one (2026-08-16, 17 cells) was served under the previous pin,
#: `956f9059871c87961495d4c861c367c7578c9821f4dc0bf709e851931e845471`, and each
#: of those cells records it in `probe_pin`. The amendment is therefore a
#: legible boundary rather than a rewrite: day one is attributable to the
#: one-column shape and everything after to the two-column one.
#:
#: **Three figures the amendment is answerable to**, so a later reader can
#: check it rather than take it: $316 projected against $350; Kimi-K3 excluded
#: for $7.42/day; and MWF-with-Kimi would have been $412, which is what a
#: mid-pilot discovery would have looked like.
BUDGET_AMENDMENT = (
    "2026-08-16, BEFORE DAY TWO: 280.00 -> 350.00 because a second provider "
    "column (DeepInfra via the HuggingFace router) became servable earlier than "
    "planned. Projected 316.00 = Together 30 x 7.86 + DeepInfra 13 x 6.17. "
    "Kimi-K3 is excluded from the daily fleet at 7.42/day; including him would "
    "project 412 and breach this gate too. Pre-registered rather than "
    "discovered: a gate raised after the spend cannot be told apart from a gate "
    "raised to excuse it.")

#: `VERDICT_FAIL` is not in the spec's enum and is added deliberately: a day
#: where serving succeeded and verdict computation failed must never read as
#: QUIET, and must be distinguishable from a provider outage.
STATUS = ("OK", "PARTIAL", "SERVE_FAIL", "VERDICT_FAIL", "BUDGET_REFUSED")

GATE_RULES = {
    "spend": "Together row OK AND Together LAT verdict-vs-rolling QUIET -> "
             "PROCEED; EVENT or STALE or SERVE_FAIL -> HOLD with the reason",
    "event-probe": "Together LAT verdict EVENT-down vs epoch REQUIRED, else "
                   "VOID-NO-CONTRAST and spend nothing",
    #: **VOIDED 2026-08-16 — the subject never existed.** This returned
    #: PROCEED on a QUIET day, correctly by its own condition, and would have
    #: licensed spending on a fork whose cohort #113 destroyed.
    "fork-reopen": "VOID-SUBJECT, always. See FORK_VOIDED",
}

#: **The fork is void, on the #108 precedent exactly.**
#:
#: `round19.SEALED_ROUND16_FORK` asked whether eight unanchored models return
#: near 0.708 (structural) or 0.95+ (the event reached them), from round-16 LAT
#: A0 cells that were "committed and have never been read". #113 established
#: that every one of those cells was served by `deepcogito/cogito-v2-1-671b`.
#: The eight models were filenames. Reading the cells would characterise cogito.
#:
#: **And the motivating question dissolved with the event.** The fork asks how
#: far the 08-14 shift reached. The certified LAT event was withdrawn by #113,
#: so there is no 08-14 reach to measure — not an unanswered question, an
#: unasked one.
#:
#: The gate now returns VOID-SUBJECT and exits NONZERO. That is the live
#: surface telling the truth. Round 19's seal text is NOT edited: it is a
#: retired pin, restored rather than re-frozen, and it stands as the record of
#: what was believed when it was sealed.
#:
#: **The residual real question was absorbed into ordinary measurement.** What
#: are those models' LAT A0 baselines? Round 21 measured seven of the eight on
#: DeepInfra, as dated rows under the provenance rule. Nothing is left to
#: reopen; the question got answered on a different column, which is what
#: normal measurement looks like when a sealed shortcut turns out to be void.
FORK_VOIDED = (
    "VOID-SUBJECT (#113). round16.FORK's eight models were never served — "
    "their round-16 cells are cogito's, so the fork has no subject. Its "
    "premise is gone too: the 08-14 event it asked about was itself withdrawn, "
    "so there is no reach to measure. NOT reopenable on any QUIET day, because "
    "occasion was never what was missing. The residual question — those "
    "models' LAT A0 baselines — is answered by round 21's DeepInfra rows, read "
    "under the provenance rule as a different column. Round 19's sealed text "
    "stands unedited in its retired pin as the record of what was believed.")

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

#: **The pin as of the two-column amendment, 2026-08-16, before day two.**
PINNED_PROBE_HASH = "4469c1af68dfdf3af00ca96998f4917ac6d580f85943039ddb18e853758d5b4c"

#: **Day one's pin, kept because day one's cells cite it.**
#:
#: The 17 cells served 2026-08-16 each record `probe_pin` = this value. A
#: standing instrument that amends its own pin makes every earlier day
#: unverifiable UNLESS the superseded hashes stay recomputable — the same
#: reason retired rounds keep their pins rather than deleting them.
#:
#: What moved: the DeepInfra column and its cohort, the frozen provider slots,
#: the budget gate 280 -> 350 with its amendment text, the resolved schedule
#: facts, and the fork's void. What did NOT move: the Together cohort, the
#: anchors, alpha, the seed base, or the decision channel — so day one's
#: readings stand under this pin exactly as they did under that one.
SUPERSEDED_PINS = {
    "956f9059871c87961495d4c861c367c7578c9821f4dc0bf709e851931e845471":
        "one-column shape, 2026-08-15 to 2026-08-16. Day one (17 cells) was "
        "served under it. Superseded by the two-column amendment before day "
        "two; nothing it governed was re-read under different rules.",
}


def world_lock_paths() -> tuple[str, ...]:
    return P.world_lock_paths(LEVELS)


def cells(provider: str = "together"):
    """The day's grid: A0 at both worlds for the cohort, plus the decision cell."""
    out = [(m, ARM, lv) for m in COHORT for lv in LEVELS]
    out.append((DECISION_MODEL, DECISION_ARM, DECISION_LEVEL))
    return out


def seed_for(provider: str, model: str, level: str, arm: str, day: int) -> int:
    """Same date -> same seeds, per provider. Deterministic and reproducible.

    **Refuses outside `SEED_BLOCK`.** A dry run on the day before the epoch
    produced day=-1 and a seed of 99136 — below the reserved block, in space
    the block exists to keep clear. The invariant has to be enforced where the
    number is made, not assumed by the test that checks the block is free.
    """
    idx = sorted({(m, a, lv) for m, a, lv in cells()}).index((model, arm, level))
    #: **The frozen slot, not `sorted(PROVIDERS).index()`** — see PROVIDER_SLOT.
    #: The alphabetical version would have moved Together's seeds the moment a
    #: second provider was added, making day one unreproducible.
    if provider not in PROVIDER_SLOT:
        raise SystemExit(
            f"provider {provider!r} has no frozen seed slot. Add it to "
            "PROVIDER_SLOT with the NEXT free index — appending keeps every "
            "existing column's seeds where they are; inserting moves them.")
    off = PROVIDER_SLOT[provider] * (len(cells()) * EPISODES)
    seed = SEED_BASE + day * SEED_STRIDE + off + idx * EPISODES
    lo, hi = SEED_BLOCK
    if not (lo <= seed and seed + EPISODES - 1 <= hi):
        raise SystemExit(
            f"seed {seed}..{seed + EPISODES - 1} for {provider} {model} "
            f"{arm} {level} on day {day} falls OUTSIDE the reserved block "
            f"{lo}-{hi}. Day {day} is "
            + ("before the seed epoch " + SEED_EPOCH if day < 0 else
               "past the block's horizon")
            + "; extend the block deliberately at a pin boundary rather than "
              "serving into unreserved seed space.")
    return seed


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
                     "rule": FLASH_ANCHOR_RULE,
                     "envelope_inert_at_pin": FLASH_ENVELOPE_INERT_AT_PIN},
        "providers": PROVIDERS, "providers_waiting": PROVIDERS_WAITING,
        "provider_slot": PROVIDER_SLOT,
        "router_attestation": ROUTER_ATTESTATION,
        "deepinfra_cohort": {k: list(v)
                             for k, v in sorted(DEEPINFRA_COHORT.items())},
        "schedule_facts": SCHEDULE_FACTS, "schedule_shape": SCHEDULE_SHAPE,
        "levels_rule": LEVELS_RULE,
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE,
                  "block": list(SEED_BLOCK), "epoch": SEED_EPOCH},
        "cost": {"daily_episodes": DAILY_EPISODES, "daily_usd": DAILY_USD,
                 "pilot_days": PILOT_DAYS, "pilot_usd": PILOT_USD,
                 "budget_per_day": BUDGET_PER_DAY,
                 "budget_pilot": BUDGET_PILOT,
                 "deepinfra_day_usd": DEEPINFRA_DAY_USD,
                 "deepinfra_per_episode": DEEPINFRA_PER_EPISODE,
                 "deepinfra_serving_days": DEEPINFRA_SERVING_DAYS,
                 "kimi_would_add": DEEPINFRA_KIMI_WOULD_ADD,
                 "amendment": BUDGET_AMENDMENT},
        "status": STATUS, "gate_rules": GATE_RULES,
        "fork_voided": FORK_VOIDED,
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
