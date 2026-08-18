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

#: **The DeepInfra column's fleet — a MATCHED PAIR, not a common fleet.**
#:
#: Re-scoped 2026-08-17 before the scheduled job's first fire (`ls results/probe-*`
#: was 17 cells, all `probe-together-`), so this is a re-scope and not a
#: mid-flight amendment. It was round 21's six — models Together would not serve,
#: chosen for coverage. The experiment sharpened: the pilot's primary target is
#: DIFFERENTIAL STABILITY, model X holding within its band on one provider and
#: stepping on the other, compared variance-to-variance and never level-to-level.
#: That needs the SAME models on both columns, which coverage cannot give.
#:
#: Selected by rule against the resolved catalogs on 2026-08-17, before any
#: cross-column trace existed — choosing the pair after seeing traces would be
#: selection on the outcome, [TRAP] 38's cousin. Together 281 models, DeepInfra
#: 184; the exact-variant intersection of the Together fleet is six.
#:
#:   probe   `DeepSeek-V4-Flash-0731` — undamped funnel, known step history, and
#:           it already carries the decision channel on Together
#:   control `Llama-3.3-70B-Instruct-Turbo` — floor-class, damped. THE CONTROL IS
#:           WHAT MAKES DIVERGENCE ATTRIBUTABLE: a model holding on both columns
#:           proves the instrument is not manufacturing variance.
#:
#: A third was available (Qwen3.5-9B, MiniMax-M3, gemma-4-31B-it, Muse-Glimmer)
#: and declined — two clean pairs beat three ragged ones. Near-misses REFUSED:
#: `nvidia/nemotron-3-ultra-550b-a55b` against DeepInfra's
#: `NVIDIA-Nemotron-3-Ultra-550B-A55B` is a different string, not a match.
#:
#: **Prices are UNKNOWN until measured.** Round 21's per-episode figures cannot
#: carry over — different models. The `(0, 0)` is deliberate and the pre-flight
#: refuses to price from it; day one measures and the pin is amended with the
#: real numbers.
#:
#: The served grid is FIVE cells, not four: A0 at both worlds for each of the
#: pair, plus the Flash A1 LAT decision cell, which this column carries because
#: Flash is in the pair (see `cells`). Together-priced that is ~$2.4/serving
#: day — an ESTIMATE from another column's rate card, carried only so the gate
#: has something to refuse against, and `DEEPINFRA_DAY_USD_TOGETHER_PRICED`
#: says so in its name.
DEEPINFRA_COHORT = {
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.0, 0.0),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.0, 0.0),
}

MATCHED_PAIR_RULE = (
    "Servable on BOTH columns at EXACT variant, attestation green, re-resolved "
    "at build and near-misses refused. One volatile probe with a known step "
    "history (Flash, which also carries the decision channel) and one boring "
    "damped control (Llama-3.3-70B). The control is what makes any probe "
    "divergence attributable to the provider rather than to the method. "
    "SELECTED BEFORE ANY CROSS-COLUMN TRACE EXISTED; choosing the pair after "
    "seeing traces is selection on the outcome.")

#: The exhibits this design turns on, in ascending strength. Recorded so a later
#: reader does not mistake the weakest for the claim.
EXHIBITS = (
    "1 MECHANISM FLIP — same model, different behaviour class or binding stage "
    "across providers. Descriptive, legitimate, computable from round 21 today.",
    "2 DIFFERENTIAL STABILITY — within-provider variance compared across "
    "columns. No level comparison anywhere. The pilot's primary target.",
    "3 COINCIDENT EVENT — one column fires while the other stays QUIET, same "
    "day. Cannot be scheduled, only watched for.")

#: Round 21's six are NOT deleted — they remain a measured register column, and
#: Kimi-K3 with them. Only the DAILY fleet changed.
ROUND21_COLUMN_STANDS = (
    "Round 21's six (plus Kimi-K3) remain a measured column in the register. "
    "The re-scope changes what is served DAILY, not what has been measured.")

DEEPINFRA_DAY_USD_TOGETHER_PRICED = 2.4

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

#: **THESE ARE TOGETHER'S ANCHORS, AND NOTHING SAID SO.**
#:
#: `EPOCH_ANCHOR`, `ANCHOR_CELLS`, `EPOCH_DAY`, `FLASH_ANCHOR` and
#: `FLASH_ENVELOPE` are all derived from the pre-event corpus, which happened to
#: be served on Together. That fact lived in the circumstances and in no data
#: structure — so `_epoch_for(channel)` and `_envelope_for(channel)` key on
#: `"LAT.A0"`, a string a second column produces identically, and a DeepInfra
#: verdict would have Fisher-tested DeepInfra's rate against TOGETHER's epoch.
#:
#: That is precisely what `LEVELS_RULE` below forbids, and the row carrying it
#: would have been pushed to the public log with `LEVELS_RULE` printed beside it.
#:
#: **The general shape, worth stating once because it is the whole class:**
#: everything deciding WHICH CELLS is provider-scoped — filenames, seeds,
#: cohorts, the rolling window. Everything deciding WHAT TO COMPARE THEM AGAINST
#: was not.
ANCHOR_OWNER = "together"

#: A column that has not earned an anchor gets NONE, not somebody else's.
#: `earned_epoch` is written by the anchor-earning process (see EARN_RULE) and
#: is empty until a column completes it. Together is here by history, not by
#: privilege: its anchors were earned from the corpus before the fleet existed.
EARNED_EPOCH: dict = {}

#: **THE ANCHOR-EARNING RULE, PINNED BEFORE THE COLUMN CAN SERVE.**
#:
#: `probe_channel`'s docstring has promised since it was written that "a new
#: column's epoch is its own first QUIET run, frozen when reached". It was
#: implemented NOWHERE — only a test did it by hand — and the consequence is
#: that NO-ANCHOR was ABSORBING: `trace` only admits QUIET days to the rolling
#: pool, so a column with no epoch could never earn a rolling one either, and
#: would have read NO-ANCHOR for all 30 days while costing full price.
#:
#: The rule is pinned BEFORE any DeepInfra day is served, because a rule for
#: what counts as a baseline, written after seeing the baseline, is not a rule.
EARN_DAYS = 3

EARN_RULE = (
    "A column with no epoch anchor for a channel accumulates daily pooled "
    "counts. When THREE CONSECUTIVE SERVED DAYS are MUTUALLY NON-SEPARABLE — "
    "every PAIRWISE Fisher p >= ALPHA, not merely adjacent pairs — their pooled "
    "sum is frozen as that channel's epoch anchor and NEVER REVISED, and the "
    "min and max of the constituent DAY rates are frozen beside it as that "
    "channel's envelope. Until then the channel reads NO-ANCHOR and yields no "
    "verdict. "
    "SERVE_FAIL AND VOID DAYS DO NOT RESTART THE WINDOW: consecutive means "
    "consecutive among SERVED days, because a provider outage is not evidence "
    "about agreement, and an anchor-earning process that an outage can reset "
    "measures uptime rather than stability. Only served days that DISAGREE "
    "restart it, and they are dropped from the candidate pool only, never from "
    "the log. "
    "THE ENVELOPE IS NOT OPTIONAL: at n=24 a day, mutual non-separability is a "
    "WEAK standard because low power makes agreement easy, so the constituent "
    "spread printed beside every later verdict is what keeps a mushy anchor "
    "honest about how mushy it is. This is the FLASH PRECEDENT, applied to days "
    "where Flash applied it to blocks.")

#: **The earning days are NOT judged against the anchor they constitute.** They
#: read NO-ANCHOR, like every day before them; the first day judged against an
#: earned anchor is the day AFTER the window closes. Testing a day against a
#: pooled sum that contains it is the circularity [TRAP] 38 records in another
#: costume, and it would make the third day look reassuringly quiet by
#: construction.
#: **THE TRUST BOUNDARY THE ROLLING WINDOW CREATES.**
#:
#: `publish.read_rows` existed, filtered by provider correctly, and was called
#: by nothing. Making it live is what gives the scheduled job a memory — without
#: it every ephemeral run starts with today's cells alone, the rolling anchor is
#: permanently empty, and no column can ever earn an epoch. Day one worked only
#: because it ran on a machine where cells persist.
#:
#: The cost is that **the job trusts its own prior rows**: the published log
#: stops being a report of verdicts and becomes an input to them. Accepted for
#: v1 because every row carries its raw cells, so any day can be recomputed from
#: evidence rather than taken from the record — but stated here, and on the log
#: card, because a trust boundary nobody wrote down is one nobody checks.
LOG_IS_LOAD_BEARING = (
    "Verdicts read prior rows from the published log for the rolling window and "
    "the anchor-earning pool. The log is therefore LOAD-BEARING for verdicts, "
    "not merely a report of them. Mitigated, not removed, by attaching raw "
    "cells to every row: a reader can recompute any day from the evidence. "
    "Where a local cell and a published row cover the same day, THE CELL WINS. "
    "Every row records which memory computed it in `history_source`, because a "
    "NO-ANCHOR day from a run that could not read the log means something "
    "entirely different from one that could.")

EARN_EXCLUDES_ITS_OWN_DAYS = (
    "The three constituent days read NO-ANCHOR and yield no verdict. An anchor "
    "cannot be evidence about the days it is made of.")


def epoch_for(provider: str, level: str, arm: str):
    """The epoch anchor for one (provider, channel), or None.

    **Provider is the first argument because it is the first question.** The
    two call sites that used to compute this — `occasion._epoch_for` and an
    inlined copy in `commands/probe.py` — both had `provider` in scope and both
    dropped it. One helper, so they cannot diverge again.
    """
    if provider == ANCHOR_OWNER:
        if arm == DECISION_ARM and level == DECISION_LEVEL:
            return FLASH_ANCHOR
        return EPOCH_ANCHOR.get(level)
    got = EARNED_EPOCH.get((provider, level, arm))
    return got["anchor"] if got else None


def envelope_for(provider: str, level: str, arm: str):
    """The constituent-block envelope, or None.

    `FLASH_ENVELOPE` is Together's Flash block range. The matched-pair design
    puts a Flash A1 LAT cell on the DeepInfra column too, and keyed on the
    channel string alone that cell would have inherited Together's envelope —
    a spec requirement turning a latent defect into an active one.
    """
    if provider == ANCHOR_OWNER:
        if arm == DECISION_ARM and level == DECISION_LEVEL:
            return FLASH_ENVELOPE
        return None
    got = EARNED_EPOCH.get((provider, level, arm))
    return got["envelope"] if got else None

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
    #: **The second column, taken 2026-08-16 through the HuggingFace router and
    #: moved to the direct API on 2026-08-17.** Round 21 established that the
    #: router serves DeepInfra on an `HF_TOKEN` alone — no DeepInfra account —
    #: which is what let this column arrive weeks before the plan expected it,
    #: without the operator errand it was supposed to wait on.
    #:
    #: **DIRECT, not the router, from 2026-08-17.** The router was how the
    #: column was TAKEN — it needed no DeepInfra account, which is why the
    #: second column arrived weeks early. It is not how the column should be
    #: SERVED: a router adds a reroute step between the request and the answer,
    #: and the matched-pair read compares one model's variance across columns,
    #: where a silent reroute is indistinguishable from the thing being
    #: measured.
    #:
    #: `model_suffix` is GONE rather than left unread. It was applied only in
    #: the round path via `R.served_id`, so on this path it was dead config
    #: describing a transport this column no longer uses — and dead config that
    #: LOOKS live is what a later reader reconstructs a wrong story from.
    "deepinfra": {"base_url": "https://api.deepinfra.com/v1/openai",
                  "key_env": "DEEPINFRA_API_KEY", "substrate": "gpu_cloud",
                  "catalogued": True, "cadence": "daily"},
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
    "`HF_TOKEN`, so the second column was TAKEN on 2026-08-16 without a "
    "DeepInfra account, and MOVED to the direct API on 2026-08-17 once a key "
    "existed. The verbs are provider-parameterised and the matrix "
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

#: **THE RULE IN FORCE FOR THIS COLUMN, now that it serves DIRECT.**
#:
#: `ROUTER_ATTESTATION` above is kept because round 21's cells were served that
#: way and it is the rule they answer to. It is NOT the rule this column serves
#: under any more, and leaving one attestation constant to cover two transports
#: is how a reader ends up believing a header was checked that never existed.
TRANSPORT_RULE = (
    "THE COLUMN IS DEFINED BY EVIDENCE RECORDED IN THE CELL, NEVER BY INTENT. "
    "Through a ROUTER the evidence is the `x-inference-provider` response "
    "header, compared to the pinned provider and REFUSED on mismatch or "
    "absence. DIRECT — which is what this column now uses — there is no router "
    "to reroute and the evidence is the ENDPOINT HOST: api.deepinfra.com "
    "answers for DeepInfra and for nobody else. Every probe cell records "
    "`base_url` and the `served_provider` derived from it, and the serving path "
    "REFUSES to serve a host that does not map to the pinned column. "
    "THIS IS WEAKER THAN THE HEADER AND THE DIFFERENCE IS STATED RATHER THAN "
    "GLOSSED: the header attests who ANSWERED, the host attests who was ASKED. "
    "It is accepted because a direct endpoint has no reroute step for the two "
    "to diverge across; if this column ever returns to a router, the header "
    "check returns with it.")

#: **Host -> column. For a direct endpoint the provider IS the host.**
#: A host absent from this map serves NOTHING on the probe path: it cannot be
#: attributed, and a cell that cannot be attributed is worse than no cell.
HOST_PROVIDER = {
    "api.together.xyz": "together",
    "api.deepinfra.com": "deepinfra",
}


def served_provider_for(base_url: str) -> str | None:
    """Which column a cell served from this endpoint belongs to, or None.

    **Blocker 5 was that the probe path had no attestation guard at all.** It
    wrote `"provider": provider` from INTENT, with no `served_provider` and no
    `base_url`, so `identity.provider_of` fell through to `DIRECT_PROVIDER` and
    a DeepInfra probe cell partitioned as TOGETHER — silently pooling two
    providers' rates in the one place the corpus most needs them apart.
    """
    host = (base_url or "").split("//")[-1].split("/")[0].lower()
    return HOST_PROVIDER.get(host)

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

#: **15:00 UTC, and the hour is pinned because drifting it would confound.**
#:
#: 08:00 Pacific: a failure surfaces at breakfast rather than overnight. But the
#: operational reason is not why it is in the payload — **a fixed hour exists to
#: remove the hour-of-day confound.** A fleet that served at 15:00 one day and
#: 03:00 the next would fold diurnal load variation into every between-day
#: verdict, and the instrument's whole claim is that a between-day difference is
#: about the endpoint.
#:
#: Consistency is the requirement; 15:00 is the choice.
SCHEDULE_UTC_HOUR = 15
SCHEDULE_CRON = "0 15 * * *"

#: **The job runs DAILY; each column decides whether today is its day.** One
#: schedule, not two — a second cron for the MWF column would make the
#: same-window property depend on two schedules agreeing rather than on there
#: being one window.
CADENCES = {
    "daily": (0, 1, 2, 3, 4, 5, 6),
    "mon_wed_fri": (0, 2, 4),          # Python weekday(): Mon=0
}

#: The invocation, recorded so it is unambiguous rather than reconstructed.
#: Not created yet: HF Jobs needs a positive credit balance on the account.
SCHEDULE_JOB = {
    "schedule": SCHEDULE_CRON,
    "flavor": "cpu-basic",             # it calls APIs and runs Fisher tests
    "timeout": "3h",                   # NOT the 30-minute default; see above
    #: `DEEPINFRA_API_KEY` for the direct second column; `HF_TOKEN` stays and
    #: is NOT vestigial — it is what pushes the day's rows and, since Phase 3,
    #: what reads the prior ones back for the rolling window and the earn pool.
    "secrets": ("TOGETHER_API_KEY", "HF_TOKEN", "DEEPINFRA_API_KEY"),
    "columns_in_order": ("together", "deepinfra"),
}


def serves_today(provider: str, weekday: int) -> bool:
    """Does this column serve on this weekday? `weekday` is `date.weekday()`.

    **The cadence is a function, not a sentence.** "DeepInfra on Mon/Wed/Fri"
    written only in prose is a cadence someone has to remember; written here it
    is one the job obeys.
    """
    cadence = PROVIDERS[provider]["cadence"]
    return weekday in CADENCES[cadence]


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

#: **The two-column pilot.** Together daily at $7.86 (measured) plus DeepInfra
#: DAILY at ~$2.4 Together-priced — 30 serving days, not 13.
#:
#: **The cadence changed with the re-scope, and the reason is the read.**
#: Mon/Wed/Fri was sized for a six-model $6.17/day column. The matched pair costs
#: a third of that, and DIFFERENTIAL STABILITY is a day-over-day variance
#: comparison: matched cadence is what makes the two traces comparable, and
#: same-day coincidence is only observable on days both columns serve. 13 of 30
#: would have thrown away more than half the primary read to save $30.
DEEPINFRA_SERVING_DAYS = 30
PILOT_USD = 308.00                   # 235.79 Together + ~72 DeepInfra
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

#: **The pin as of the MATCHED-PAIR RE-SCOPE, 2026-08-17, before day two.**
#:
#: **Re-pinned DELIBERATELY, ONCE, at the end of phase 4.** The previous pin was
#: frozen against a design whose second column was six round-21 models on a
#: Mon/Wed/Fri cadence through the HuggingFace router; this one is frozen
#: against a matched pair, daily, on the direct DeepInfra API. Cohort, cadence,
#: cost, serving days, grid and transport all moved, so the pin moves with them
#: — a re-scope that did NOT move it would mean the payload was not carrying
#: the design.
#:
#: **Once** is the discipline, not a detail. The hash was recomputed mid-phase
#: while the transport was still the router, which would have entered
#: SUPERSEDED_PINS as a pin that existed for no commit and governed no cell —
#: churn in the one list a reader consults to ask which rules a given day
#: answered to. It was replaced rather than recorded. Only pins that actually
#: reached a commit are listed below.
#:
#: What newly travels, and why this hash differs by more than the constants:
#: `matched_pair_rule`, `exhibits`, `round21_column_stands`, `anchor_owner`,
#: the `earn` block, `log_is_load_bearing`, `transport_rule` and
#: `host_provider`. Every one was a module constant that nothing hashed —
#: which made the pair's "SELECTED BEFORE ANY CROSS-COLUMN TRACE EXISTED"
#: clause editable after the traces arrive, and the anchor-earning rule
#: editable after seeing what the first three days looked like. Documentation
#: wearing the costume of a pre-registration.
#:
#: **No cell has been served under any pin since day one.** The 17 cells in
#: `results/` all cite 956f9059, and the scheduled job is deleted, so this
#: re-pin governs nothing retroactively.
PINNED_PROBE_HASH = "8bfb3da9ed3effda91554cc10176ed864f1549fb47002e2a4c2459a8b26936ad"

#: **Day one's pin, kept because day one's cells cite it.**
#:
#: The 17 cells served 2026-08-16 each record `probe_pin` = this value. A
#: standing instrument that amends its own pin makes every earlier day
#: unverifiable UNLESS the superseded hashes stay recomputable — the same
#: reason retired rounds keep their pins rather than deleting them.
#:
#: What moved across all of them: the DeepInfra column and its cohort, the
#: frozen provider slots, the budget gate 280 -> 350 with its amendment text,
#: the resolved schedule facts, the fork's void, and — at the re-scope — the
#: second column's entire fleet and cadence. What has NEVER moved: the Together
#: cohort, the anchors, alpha, the seed base, and the decision channel. Day
#: one's readings therefore stand under the current pin exactly as they did
#: under the one they cite, which is the property that makes superseding safe.
SUPERSEDED_PINS = {
    "956f9059871c87961495d4c861c367c7578c9821f4dc0bf709e851931e845471":
        "one-column shape, 2026-08-15 to 2026-08-16. Day one (17 cells) was "
        "SERVED under it. Superseded by the two-column amendment before day "
        "two; nothing it governed was re-read under different rules.",
    #: **Committed but never served under, and recorded anyway.**
    #: It existed in the repository for one commit, between the two-column
    #: amendment and the schedule being fixed. No cell cites it. Listing it
    #: costs nothing and stops a reader who finds it in git history from
    #: wondering which day it governed: none.
    "4469c1af68dfdf3af00ca96998f4917ac6d580f85943039ddb18e853758d5b4c":
        "two-column shape before the UTC hour was pinned. NO CELLS WERE SERVED "
        "under it — it lived in the repository for one commit and was "
        "superseded by the schedule amendment on the same day.",
    #: **The six-model second column, superseded by the matched pair.**
    #: Committed at da0e265 against a deliberately RED tree, mid-re-scope. The
    #: scheduled job that would have served under it was deleted before it
    #: could fire, so again: none.
    "b380cc610df78fbd684a482cdd2e528c4a3be7758f77ebfc31c00fa938e8a08f":
        "six-model DeepInfra cohort on a Mon/Wed/Fri cadence, 2026-08-16 to "
        "2026-08-17. NO CELLS WERE SERVED under it — the scheduled job was "
        "deleted before its first fire, and the column it describes was "
        "re-scoped to the matched pair for the DIFFERENTIAL STABILITY read.",
}


def world_lock_paths() -> tuple[str, ...]:
    return P.world_lock_paths(LEVELS)


def cohort_for(provider: str) -> dict:
    """**Each column has its own cohort, because each provider serves its own
    models.** Together's eight are the anchored fleet; DeepInfra's are the
    MATCHED PAIR — Flash and Llama-3.3-70B, servable on both columns at exact
    variant. See `MATCHED_PAIR_RULE`.

    **They deliberately DO overlap, and the overlap is the whole point.** This
    docstring said the opposite until 2026-08-17, and it was correct when it was
    written: disjoint cohorts are sufficient for the COINCIDENCE read, which
    compares EVENTS across columns and never levels, each column detecting
    against its own anchors. But the pilot's primary target is now DIFFERENTIAL
    STABILITY — one model's within-provider variance compared across columns —
    and disjoint cohorts cannot produce that read at all. The coincidence read
    is unaffected and still needs no overlap; it simply is no longer the only
    read this function serves.
    """
    return DEEPINFRA_COHORT if provider == "deepinfra" else COHORT


def cells(provider: str = "together"):
    """The day's grid: A0 at both worlds for that provider's cohort, plus the
    decision cell where the provider carries one.

    **This took `provider` and ignored it**, so the DeepInfra column would have
    served TOGETHER's eight models through the router — models DeepInfra may not
    host at all, under a cohort constant that existed and was never read. Found
    when the dry run printed 408 episodes for a column whose grid is 288.

    Third instance in this build of the same shape: a constant defined, a
    function written, and nothing calling it (`DEEPINFRA_COHORT`,
    `serves_today`, and this parameter). Defining the rule is half the work.
    """
    out = [(m, ARM, lv) for m in cohort_for(provider) for lv in LEVELS]
    #: **The decision cell is served on every column that carries the model.**
    #:
    #: This appended it for Together alone, reasoning that a DeepInfra Flash
    #: cell "would be a different served artifact judged against the wrong
    #: baseline". That was TRUE while the channel string alone chose the anchor
    #: — and it is exactly the defect `epoch_for`/`envelope_for` fix. DeepInfra
    #: Flash A1 LAT now resolves through `EARNED_EPOCH`, which is empty, so the
    #: cell gets NO anchor rather than Together's. The objection was to the
    #: inheritance, not to the cell.
    #:
    #: Fourth instance in two days of the same shape, and the sharpest:
    #: `envelope_for`'s own docstring already said "the matched-pair design puts
    #: a Flash A1 LAT cell on the DeepInfra column too" while THIS function, ten
    #: lines away, refused to emit it. A rule the machinery does not read is a
    #: rule it does not follow, even when the rule is written in the machinery.
    #:
    #: Together is here by `ANCHOR_OWNER` and not by cohort membership: Flash is
    #: the decision channel, not one of the anchored eight, so `DECISION_MODEL
    #: in COHORT` is False and testing membership alone would have silently
    #: dropped Together's decision cell — 17 cells to 16, day one unreproducible.
    if provider == ANCHOR_OWNER or DECISION_MODEL in cohort_for(provider):
        out.append((DECISION_MODEL, DECISION_ARM, DECISION_LEVEL))
    return out


def seed_for(provider: str, model: str, level: str, arm: str, day: int) -> int:
    """Same date -> same seeds, per provider. Deterministic and reproducible.

    **Refuses outside `SEED_BLOCK`.** A dry run on the day before the epoch
    produced day=-1 and a seed of 99136 — below the reserved block, in space
    the block exists to keep clear. The invariant has to be enforced where the
    number is made, not assumed by the test that checks the block is free.
    """
    #: **Index within THIS provider's own grid.** It read `cells()` — Together's
    #: grid — so the first DeepInfra cell raised `ValueError: not in list`. For
    #: Together `cells("together") == cells()`, so this cannot move day one.
    grid = sorted({(m, a, lv) for m, a, lv in cells(provider)})
    if (model, arm, level) not in grid:
        raise SystemExit(
            f"{model} {arm} {level} is not in {provider}'s grid. Each column "
            "has its own cohort; serving a model outside it would take a seed "
            "from another column's range.")
    idx = grid.index((model, arm, level))
    #: **The frozen slot, not `sorted(PROVIDERS).index()`** — see PROVIDER_SLOT.
    #: The alphabetical version would have moved Together's seeds the moment a
    #: second provider was added, making day one unreproducible.
    if provider not in PROVIDER_SLOT:
        raise SystemExit(
            f"provider {provider!r} has no frozen seed slot. Add it to "
            "PROVIDER_SLOT with the NEXT free index — appending keeps every "
            "existing column's seeds where they are; inserting moves them.")
    off = PROVIDER_SLOT[provider] * (len(cells()) * EPISODES)
    #: **THE STRIDE MUST HOLD EVERY COLUMN'S SPAN, or day N+1 reuses day N's
    #: seeds.** `SEED_STRIDE` is 864 = 36 cell-slots a day, and its comment says
    #: "across four providers" — but four columns at Together's 17-cell grid
    #: need 68 slots. Slots 0 and 1 fit (max offsets 407 and 527); slot 2 runs
    #: to 1199 and slot 3 to 1631, both past the stride and therefore into the
    #: NEXT day's range, starting with Together's.
    #:
    #: Latent today, since only two columns exist — and silent when it fires,
    #: because nothing downstream would notice two days sharing seeds. The block
    #: bounds below are checked; this was not.
    #:
    #: **The fix is a refusal, not a wider stride.** Day one is day=1 and its
    #: cells derive from `100000 + 1*864`; changing SEED_STRIDE would move every
    #: served seed and make the 17 committed cells unreproducible from the code
    #: that made them — the precise corruption PROVIDER_SLOT was frozen to
    #: prevent. Widening needs a new SEED_BLOCK at a deliberate pin boundary.
    span = off + len(grid) * EPISODES
    if span > SEED_STRIDE:
        raise SystemExit(
            f"provider {provider!r} (slot {PROVIDER_SLOT[provider]}) needs "
            f"seed offsets up to {span - 1} within a day, but SEED_STRIDE is "
            f"{SEED_STRIDE} — day {day}'s seeds would run into day {day + 1}'s "
            "range and silently reuse another column's numbers. Widen the "
            "stride AND the block together at a pin boundary; note that "
            "changing SEED_STRIDE moves every seed already served, so day one "
            "must be re-derivable or explicitly retired first.")
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
        "transport_rule": TRANSPORT_RULE,
        "host_provider": HOST_PROVIDER,
        "deepinfra_cohort": {k: list(v)
                             for k, v in sorted(DEEPINFRA_COHORT.items())},
        #: **The pair's selection rule travels, or it is not a
        #: pre-registration.** MATCHED_PAIR_RULE's load-bearing clause is
        #: "SELECTED BEFORE ANY CROSS-COLUMN TRACE EXISTED" — a claim whose
        #: entire value is that it cannot be edited after the traces arrive.
        #: Stated in a module constant and absent from the payload, it could
        #: have been, with no pin breaking. Same for ANCHOR_OWNER, which is the
        #: one constant standing between a DeepInfra verdict and Together's
        #: epoch, and for the EXHIBITS, which fix what the pilot claims BEFORE
        #: the data can suggest which claim is easiest to support.
        "matched_pair_rule": MATCHED_PAIR_RULE,
        "earn": {"days": EARN_DAYS, "rule": EARN_RULE,
                 "excludes_its_own_days": EARN_EXCLUDES_ITS_OWN_DAYS},
        "log_is_load_bearing": LOG_IS_LOAD_BEARING,
        "exhibits": EXHIBITS,
        "round21_column_stands": ROUND21_COLUMN_STANDS,
        "anchor_owner": ANCHOR_OWNER,
        "schedule_facts": SCHEDULE_FACTS, "schedule_shape": SCHEDULE_SHAPE,
        "schedule_utc_hour": SCHEDULE_UTC_HOUR, "schedule_cron": SCHEDULE_CRON,
        "cadences": {k: list(v) for k, v in sorted(CADENCES.items())},
        "schedule_job": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in sorted(SCHEDULE_JOB.items())},
        "levels_rule": LEVELS_RULE,
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE,
                  "block": list(SEED_BLOCK), "epoch": SEED_EPOCH},
        "cost": {"daily_episodes": DAILY_EPISODES, "daily_usd": DAILY_USD,
                 "pilot_days": PILOT_DAYS, "pilot_usd": PILOT_USD,
                 "budget_per_day": BUDGET_PER_DAY,
                 "budget_pilot": BUDGET_PILOT,
                 "deepinfra_day_usd_together_priced":
                     DEEPINFRA_DAY_USD_TOGETHER_PRICED,
                 "deepinfra_serving_days": DEEPINFRA_SERVING_DAYS,
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
