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

Started 2026-08-15. Together daily; DeepInfra on Mon/Wed/Fri through the
HuggingFace router, where the response header `x-inference-provider` decides
whether a cell is kept — a routed request that lands elsewhere produces **no
row** rather than a mislabelled one.

Fireworks and SambaNova are specified and not built: their keys do not exist.
Every column begins with no anchor and earns one, so a column started later is
not a column started worse.

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
