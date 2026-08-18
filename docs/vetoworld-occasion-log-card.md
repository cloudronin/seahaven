---
license: mit
pretty_name: VetoWorld occasion log
task_categories:
  - reinforcement-learning
language:
  - en
tags:
  - agents
  - alignment
  - evaluation
  - monitoring
  - reproducibility
size_categories:
  - n<1K
---

# VetoWorld occasion log

**A daily record of whether a hosted model still behaves the way it did
yesterday.**

Every row is one channel, one provider, one UTC day: a fixed grid of episodes
served against a frozen anchor, with a Fisher test and a verdict. The point is
not the models. The point is the *serving*.

## Why this exists

The VetoWorld programme measured one model's rate shift **0.319 between two
serving days**, on the same pinned world, same prompt, same request form. Batch
composition and prefix cache were both ruled out; a deployment change on the
provider's side is consistent with it and untestable from outside.

That single observation is expensive to live with. It means any two numbers
from a hosted endpoint separated in time may differ because the endpoint
changed, and nothing in a normal benchmark run would tell you. So this log
exists to make the question answerable rather than arguable: it serves a fixed
grid every day and records what came back.

**A second use appeared and is now the more important one.** The fleet
schedules the science. Some probes require the state to be *expressed* — a
comparison that measures the damping of a shift is meaningless on a day with no
shift — so the daily reading is what grants permission to spend. A skipped day
is a day the programme cannot react, not merely a gap in monitoring.

## What a row is

| field | meaning |
|---|---|
| `date` | UTC serving day |
| `provider` | who **answered**, taken from the response, not from the request |
| `channel` | world + arm, e.g. `LAT.A0` |
| `verdict` | `QUIET`, `EVENT`, `STALE`, `NO-ANCHOR` |
| `direction` | `up` or `down` — **EVENT means MOVED, not FELL** |
| `now` | successes / episodes on the day |
| `p_epoch` | Fisher vs the frozen epoch anchor |
| `p_rolling` | Fisher vs the rolling window of recent QUIET days |
| `status` | `OK`, `PARTIAL`, `SERVE_FAIL`, `VERDICT_FAIL`, `BUDGET_REFUSED` |
| `probe_pin` | the payload digest the day was served under |
| `history_source` | which memory computed the verdict — see the trust boundary below |

## Five things to know before using it

**1. Levels are never compared across providers.** The same open weights on a
different serving stack are a different served artifact. Only *within*-provider
deltas and *cross*-provider **coincidence of events** are read. A row from one
provider and a row from another are not two measurements of one thing.

**2. Direction is part of the verdict.** Reading EVENT as "fell" inverted a
conclusion once in this programme: two co-located worlds rose on the day a third
fell, and a one-sided reading sleeps through half the signal.

**3. False alarms are expected and pre-counted.** At three channels, 365 days
and alpha = 0.01, roughly **11 EVENT-days per year arise from noise alone**.
That figure is printed beside every verdict. An isolated EVENT is not a finding;
a coincidence across independent columns, or a persistent run, might be.

**4. A missing day is not a quiet day.** `STALE` and `SERVE_FAIL` are distinct
verdicts and neither means the channel was calm. `VERDICT_FAIL` exists
specifically so that a day where serving worked and the computation broke can
never be mistaken for `QUIET`.

**5. The pin can be amended, and rows say which pin they ran under.** This is a
standing instrument, not a fixed-length experiment; its grid and its gates can
change at a stated boundary. Every row carries `probe_pin`, and superseded pins
stay recomputable, so a reader can always tell which rules produced a row rather
than assuming today's.

## Columns

Started 2026-08-15. **Together and DeepInfra, both daily, both on their
provider's direct API.** DeepInfra was taken on 2026-08-16 through the
HuggingFace router — which serves it on an `HF_TOKEN` with no DeepInfra account,
and is why the second column arrived weeks early — and moved to the direct
endpoint on 2026-08-17. Round 21's cells were served through the router and
answer to the header rule below; nothing in this log does.

Direct, there is no router to reroute, so the column is established by the
**endpoint host**, recorded in every cell as `base_url` alongside a
`served_provider` derived from it. The serving path refuses to serve a host that
does not map to a known column, because a cell that cannot be attributed is
worse than a cell never served. This is deliberately **weaker than the routed
rule** it replaces: a response header attests who *answered*, a host attests who
was *asked*. The difference is stated rather than glossed, and if a column ever
returns to a router the header check returns with it.

Fireworks and SambaNova are specified and not built: their keys do not exist.
Every column begins with no anchor and earns one, so a column started later is
not a column started worse.

## The matched pair, and what it is for

The two columns serve **the same two models**: `DeepSeek-V4-Flash-0731` as a
volatile probe with a known step history, and `Llama-3.3-70B-Instruct-Turbo` as
a damped, floor-class control. They were chosen as the exact-variant
intersection of the two providers' catalogues, **before any cross-column trace
existed** — choosing a pair after seeing its traces is selection on the outcome.
Near-misses were refused: `nvidia/nemotron-3-ultra-550b-a55b` and DeepInfra's
`NVIDIA-Nemotron-3-Ultra-550B-A55B` are different strings and therefore
different models.

The control is what makes a divergence attributable. A model that holds on both
columns while the other steps on one is evidence about the provider; without the
control, the same picture is equally evidence that the instrument manufactures
variance.

Three claim classes are on the table, in ascending strength, and they are fixed
here so the data cannot later suggest which is easiest to support:

1. **Mechanism flip** — same model, different behaviour class or binding stage
   across providers. Descriptive and legitimate. Emitted by
   `vworld emit exhibit-1`, which currently reports **zero**: every overlapping
   model's Together-side cell comes from a retracted sweep.
2. **Differential stability** — one model's *within*-provider variance compared
   across columns. Variance to variance, never level to level. The pilot's
   primary target, and what the matched pair exists for.
3. **Coincident event** — one column fires while the other stays QUIET on the
   same day. Cannot be scheduled, only watched for.

**None of these compares a level across columns**, and that is not an oversight
to be relaxed later. It is the constraint the whole design is built around: see
point 1 above.

## The trust boundary: this log is an input to itself

**Stated plainly because it is easy to miss.** Verdicts need history — a rolling
window of recent QUIET days, and, for a column that has not yet earned an epoch
anchor, a pool of its first served days. The scheduled job runs in an ephemeral
container that fetches no probe cells, so it reads that history **back out of
this log**. The published record is therefore *load-bearing for verdicts*, not
merely a report of them.

This is accepted for v1 rather than hidden, and it is mitigated in two ways.
Every row ships with its raw cells attached, so any day can be recomputed from
the evidence instead of taken from the record; and where a local cell and a
published row cover the same day, **the cell wins**. Every row also records
`history_source`, because a `NO-ANCHOR` day produced by a run that could not
read the log means something entirely different from one produced by a run that
could, and without that field the two are indistinguishable.

## How a new column earns its anchor

A column with no epoch anchor reads `NO-ANCHOR` and yields no verdict — it
admits to the record and claims nothing. It earns one when **three consecutive
served days are mutually non-separable**: every *pairwise* Fisher p ≥ alpha, not
merely adjacent pairs, since adjacent-only would freeze a monotone drift as a
baseline and disable the very anchor meant to catch drift. Their pooled sum
becomes the anchor and is never revised, and the **spread of the constituent
days** is frozen beside it as an envelope — at 24 episodes a day, mutual
non-separability is a weak standard, and the envelope is what keeps a soft
anchor honest about how soft it is.

Two details that decide what the rule measures. **Failed and void days do not
restart the window**: "consecutive" counts served days only, because an outage
is not evidence about agreement, and a rule an outage can reset measures uptime
rather than stability. And **the three constituent days are not judged against
the anchor they form** — an anchor cannot be evidence about the days it pools.

## Reproducing a verdict

    pip install vetoworld
    vworld occasion verdict --date YYYY-MM-DD

That recomputes from committed cells, costs nothing and needs no key. The
serving verb is separate and is the only one that spends.

## Limits worth stating plainly

This log cannot tell you *why* a channel moved. It observes an endpoint from
outside, and the provider's side is not visible: a deployment, a quantisation
change, a routing change and a silent model swap are indistinguishable here.
What it can do is establish **when** something changed, and whether it changed
on more than one stack at once — which is the difference between "this vendor
shipped something" and "the ecosystem moved".

It is also a small instrument. A handful of channels at m = 24 has modest power;
it detects large shifts reliably and small ones not at all. The 0.319 that
motivated it would be caught easily. A 0.05 drift would not.
