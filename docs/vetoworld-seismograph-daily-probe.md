# The daily reference-channel probe — standalone spec (v1.1, multi-provider pilot)

A scheduled job that serves a fixed probe fleet against **four providers**
every day for a **30-day budget-gated pilot**, computes occasion verdicts with
the pinned machinery, and appends one row per provider-day to a public HF
dataset. That dataset is the programme's weather report: **every spending run
consults it before serving, and #110-class probes are gated on it by rule.**
Internally this is the seismograph; the public surfaces use plain names (the
standalone instrument's naming is parked).

## What it answers

- Is today QUIET or EVENT at each channel, per provider, against frozen or
  accumulated anchors?
- Is the 08-14 LAT state reverting, flattening, or deepening on Together?
  (Everything sealed — round 16's fork, the fourteen, raidex — waits on this.)
- **Cross-provider coincidence**: does an event appear on one stack, the GPU
  class, or everywhere? Same-day contrast needs no history and works from
  day 1.
- Long-run: the base rate of serving events, per provider, on the public
  record — the first such matrix anyone has published.

## The provider matrix

| provider | substrate | fleet | anchors at start |
|---|---|---|---|
| Together | GPU cloud | **full fleet** (below) | clean pre-event history — the only anchored column |
| Fireworks | GPU cloud | common fleet | NO-ANCHOR; verdicts online after ~3-5 quiet days |
| DeepInfra | GPU cloud | common fleet | NO-ANCHOR, same |
| SambaNova | custom RDU | **reduced sub-column** (curated catalog) | NO-ANCHOR, same |

Three GPU clouds distinguish "one provider moved" from "the GPU ecosystem
moved" (shared inference-engine releases are a real correlated-failure class).
SambaNova is the different-substrate site: even a two-model column buys the
ecosystem-vs-stack attribution nothing else can. **OpenRouter is deliberately
excluded** — it is a router whose serving substrate changes per request by
design; its trace would measure routing policy, not stack stability.

**Levels are never compared across providers.** Same weights on a different
stack is a different served artifact (e.g. FP8 variants). Only within-provider
deltas and **cross-provider coincidence of events** are read. The dataset card
states this rule.

## The fleets (~$8-9/day across all four columns)

**Together (full fleet, ~$2.50/day):**

| channel | cells | senses |
|---|---|---|
| Acting, undamped | A0 m=24 at **LAT**, 4 models | idle-class shifts where nothing damps them |
| Acting, damped co-site | A0 m=24 at **W2**, same 4 | attribution: moved *where*. The gap is the classifier |
| Decision | A1 m=24, **DS-V4-Flash at LAT** | rule-conditioned steps A0 is blind to; the channel that caught the 0.319 event |

Together cohort by rule, derived from disk at build: cheap-tier models with
clean pre-event (≤ 08-13) anchors at BOTH LAT and W2, disjoint from round 16's
fourteen, spanning response classes. Expected: Qwen2.5-7B, Qwen3.5-9B,
Llama-3.3-70B, MiniMax-M3 — the rule selects; the selection freezes in the pin.

**Common fleet (Fireworks, DeepInfra — same channels, ~$2-2.50/day each):**
the intersection rule — models servable at exact variant on ALL GPU-cloud
columns, cheap tier, spanning classes. Expected from current catalogs:
**Llama-3.3-70B** (floor class), **DS-V4-Flash** (undamped decision channel),
**one Qwen** (mid tier; 7B if both serve it, else Qwen2.5-72B accepted with
the note that it starts anchorless everywhere including Together). MiniMax-M3
drops — Together-only. Channels identical to the full fleet, minus models the
intersection excludes.

**SambaNova sub-column (~$1-1.50/day):** whatever the curated catalog supports
of the common fleet — expected Llama-3.3-70B (A0 at LAT + W2) and DS-V4-Flash
(A1 at LAT). A partial column, labeled as such, never padded.

**Availability is aggregator-sourced tonight and must be resolved at build:
each provider's own catalog API, exact strings, near-misses refused** (the
round-13 probe discipline). The resolved matrix freezes in the pin;
per-provider variant strings (e.g. Turbo/FP8 designations) are recorded on
every row.

**Seeds:** date-derived per (provider, model, world, arm):
`PROBE_BASE + days_since_epoch × STRIDE + provider_offset + channel_offset`,
from a dedicated reserved block, disjoint from all corpus blocks by
construction, asserted at build. Same date → same seeds, per provider.

**Probe cells never pool into any score corpus.** Own round tag
(`probe-<provider>-YYYYMMDD`), own dataset.

## The pilot frame (pinned before day 1)

- **Duration 30 days, budget gate $280 total.** The job refuses to serve past
  the gate.
- **Day-30 decision gate, criteria pinned now:** per column — keep daily, cut,
  or reduce to MWF — decided by (a) events detected with the false-alarm
  expectation beside them, (b) coincidence table yield, (c) anchor stability
  (did the column reach and hold QUIET baselines at all). No column is judged
  on its *levels*.
- **Pre-registered reads:** per-provider event count vs expected false alarms;
  the same-day coincidence table (which columns fired together, dates); the
  LAT−W2 gap trace per provider; Together's LAT trajectory vs the 08-13 epoch
  anchor (revert / flatten / deepen). Cross-provider *level* differences are
  reported as documentation, never as findings.

## Verdicts

Paired within-model, pooled per (provider, channel): Fisher on before/after
2×2, **two-sided with direction printed** (W2 rose during the current event; a
one-sided test sleeps through half the signal). `alpha = 0.01` pinned; the
expected false-alarm count at cadence — now ≈ 3.7 EVENT-days/year **per
channel per provider**, with the matrix-wide total printed on the card —
appears on every row.

**Anchors are per (provider, channel). Dual, because rolling-only is a boiling
frog.** A slow drift below daily detection would otherwise be appended into
"normal" one quiet day at a time.

- **Epoch anchor**: Together's is the frozen 08-13 clean cells, permanent.
  Each new provider's epoch anchor is **its own first QUIET-run baseline**
  (first ≥3 consecutive days with no step among themselves), frozen once
  reached and never revised — the pilot's first deliverable per column.
- **Rolling anchor**: last K=5 QUIET days (pinned window), for day-over-day
  step detection. EVENT days never enter it (taint law). No QUIET refresh in
  7 days → the channel prints **STALE**, and STALE is HOLD for
  event-conditioned probes.
- Until a column's epoch anchor exists, its rows carry **NO-ANCHOR** — admits
  to the record with a flag, yields no verdict, and the **cross-provider
  same-day coincidence read still functions**, since it needs no history.

DS-V4-Flash's Together decision channel anchors on its stable post-step run
(blocks 2–5, by the rule "most recent ≥2 agreeing blocks"), with the 0.181 era
listed as its known prior event. Its Fireworks/DeepInfra/SambaNova twins start
NO-ANCHOR like everything else off-Together.

**Primary traces:** the `LAT − W2` gap per provider per day (near historical =
quiet; opening = passivity-class event; both moving with gap held = broad
event), and the **coincidence row** — which providers' same channel fired on
the same date. One is the classifier within a stack; the other is the
classifier across stacks.

## The dataset (the product)

`vetoworld-occasion-log` (name checked at build), separate from the corpus.
One row per **provider-day**:

    date, provider, substrate_class, per-channel {model_variant_string, k, n,
    rate, wilson, verdict_vs_epoch, verdict_vs_rolling, direction, p},
    gap_LAT_W2, anchors_used (cell ids or NO-ANCHOR), seeds, spend_usd,
    pin_digest, package_version, serve_status

Plus the raw cells under `cells/<provider>/YYYY-MM-DD/`, and a daily
`coincidence/YYYY-MM-DD.json` summarizing which (provider, channel) pairs
fired. `serve_status ∈ {OK, PARTIAL→VOID, SERVE_FAIL}` — a provider outage is
a SERVE_FAIL row for that column only, never an EVENT, and never blocks the
other columns' rows.

## The gate (what consumes it)

New verb: `vworld occasion gate [--for <purpose>]` reads the latest rows.
**Gate decisions key on Together's column only** — it is the programme's
serving provider and the only anchored one; the other columns inform
attribution, not permission, until they have epoch anchors and a standing
role.

| purpose | rule |
|---|---|
| `spend` (default; cohort sweeps, raidex-facing runs) | Together row OK **and** Together LAT verdict-vs-rolling QUIET → PROCEED; EVENT or STALE or SERVE_FAIL → HOLD, reason printed |
| `event-probe` (#110-class, needs the state expressed) | Together LAT verdict **EVENT-down vs epoch** required, else VOID-NO-CONTRAST |
| `fork-reopen` (round 16's sealed fourteen) | Together fleet LAT QUIET licenses *serving* the re-serve; the re-serve's own anchored-six certification still decides reading. Fleet is tripwire, not judge — REFERENCE_DISJOINT holds |

Gate output is one line plus exit code, so it drops into any script or CI.

## Deployment (HF-scheduled)

- **HF Scheduled Job** (or a scheduled Space runner if Jobs' cron is
  unavailable on the account tier) — **[INVESTIGATE] confirm current HF Jobs
  scheduling syntax, timeout ceiling, and secret injection at build time**;
  my mechanics may be stale.
- Job = `pip install vetoworld==<pinned>` + `vworld probe-daily` (new verb
  wrapping serve→verdict→push, looping providers). CPU-only; the work is API
  calls. **Columns are independent**: one provider's failure or slowness
  never blocks another's row.
- Secrets: `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `DEEPINFRA_API_KEY`,
  `SAMBANOVA_API_KEY`, `HF_TOKEN` as job secrets. Never in the image, never
  in the dataset. A missing key = SERVE_FAIL for that column, loudly, not a
  silent skip.
- **Idempotency by (provider, date)**: existing row → no-op for that column.
  A mid-day crash re-serves the incomplete columns on the same date-derived
  seeds and pushes once.
- Push is first-class: dataset write failure fails the job loudly. No silent
  local-only days.
- Budget guards: per-day ceiling $12, pilot ceiling $280 — the job refuses
  past either, and the refusal is itself a row.
- Schedule: daily, fixed UTC hour, all providers within the same window so
  the coincidence read is same-occasion by construction. Hour recorded on
  the card.

## Build-time derivations ($0, before the pin)

1. Together cohort by the stated rule, from committed metadata.
2. **The common fleet by the intersection rule, resolved against each
   provider's own catalog API** — exact variant strings, near-misses refused;
   the resolved (provider × model × variant) matrix freezes in the pin.
3. Each Together model's anchor cells at LAT and W2, by cell id.
4. The gap's historical value and spread from committed same-day pairs
   (Together only; other columns accumulate their own).
5. The DS-V4-Flash anchor blocks by the stated rule.
6. Dataset name availability; the reserved seed block, sized for
   4 providers × 30+ days.
7. The pilot's day-30 decision criteria, frozen as literals.

All frozen in the probe's pin, which hashes the verdict machinery
(`occasion.py`), the anchor literals, alpha, K, the gate rules, and the
provider matrix — so the daily job can't drift from what was registered.

## Verification

1. Dry-run mode serves nothing and prints every column's would-be requests
   (payload machinery reused), including per-provider variant strings.
2. Replaying 08-13/14/15 committed cells through the verdict path reproduces
   QUIET/EVENT/EVENT on Together — the known trace is the fixture.
3. Synthetic SERVE_FAIL and PARTIAL days produce their statuses per column,
   never verdicts, and never block sibling columns.
4. Idempotency: running twice on one (provider, date) serves once.
5. Boiling-frog test: a synthetic 0.02/day drift trips verdict-vs-epoch
   within the window while verdict-vs-rolling stays QUIET.
6. NO-ANCHOR lifecycle: a synthetic column reaches its first 3-quiet-day run,
   freezes an epoch anchor, and starts issuing verdicts — and the anchor
   never revises afterward.
7. Gate: each purpose returns the pinned decision on fixture rows, and
   ignores non-Together columns.
8. Coincidence artifact: synthetic same-day events on two columns produce the
   pair; a SERVE_FAIL day cannot appear in it.
9. Full suite green; no benchmark pin moves; probe cells provably excluded
   from corpus digests.

## Not in scope

Gate authority for non-Together columns (revisit at day 30). Paid tiers,
alerting beyond the dataset row, public naming of the standalone instrument.
OpenRouter (a router measures routing policy, not stack stability — a
"deployer experience" column is a different instrument). Anthropic/OpenAI
stacks. Any change to the sealed fork's reading rules.

## Do we wait on anything?

No. The probe is the waiting, made mechanical. Together's open question —
whether LAT reverts — is what it records from day 1; the new columns spend
their first days earning epoch anchors, which is itself the pilot's first
deliverable, and the cross-provider coincidence read works from day 1 with no
history at all.
