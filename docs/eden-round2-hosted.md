# EdenBench round 2 — the hosted route, and what it costs

Recorded because round 2 leaves the self-served regime every published number in
this program was measured under. The route itself is not new: the two-tier
amendment (`axis2-prereg-amendment-two-tier-kp4.md:153-156`) already specifies
the only sanctioned way to use a hosted API — **never pooled with the 2,285
self-served cells, its own budget, and a per-model noise floor measured
empirically rather than assumed.** This document holds round 2 to that, and
records the facts discovered by pointing the harness at the provider.

Note that `c3-discoverability-prereg.md:25` and `:133` name Together specifically
as *excluded*. That exclusion was written for C3, which needed comparability with
the self-served corpus. Round 2 is a **separate sealed experiment** and claims no
such comparability — which is precisely why it is allowed to go here, and why
nothing it produces may be pooled backwards.

## What is lost, stated up front

- **Bit-exact replication.** Axis 2b reproduced `delta +0.000` across days under
  `VLLM_BATCH_INVARIANT=1`. That flag does not exist here.
- **The version pin.** `vllm==0.26.0` was asserted in preflight. The provider's
  stack is unversioned from the client's side and may change mid-run.
- **Batch composition.** Unknown and uncontrollable; other tenants share it.
- **`seed`.** Sent on every request and best-effort at most.

Consequently **round 2 is not command-comparable to round 1 or to any
self-served cell**, and the raised token cap (below) would break that
comparability by itself even if the serving stack were identical.

## What was discovered by running it

Four facts, none of which were in the plan, each found by calling the API rather
than by reading the harness. All four are now fixed in code and covered.

### 1. Prompt caching is PER-MODEL, and half the cohort gets none.

The plan assumed caching "would cut input ~84% if Together applies it — the
conversation is perfectly cache-shaped," and quoted **$12-16 input** on that
basis. The answer is neither yes nor no. Measured `cached_tokens / prompt_tokens`
over a 12-episode cell:

    gemma-4-31B-it            94.6%        Qwen3.5-9B                 0.0%
    DeepSeek-V4-Flash-0731    90.2%        Qwen2.5-7B-Instruct-Turbo  0.0%
    Muse-Glimmer-30B          89.1%        gemma-3n-E4B-it            0.0%
    MiniMax-M3                71.8%        LFM2.5-8B-A1B              0.0%
    gpt-oss-120b              63.7%        gpt-oss-20b                0.0%

So the cache-shaped conversation pays off on some deployments and not at all on
others, and **which is which is not predictable from price, size or family** —
two Qwen models and two Google models sit on opposite ends. An estimate built on
either assumption would be wrong for half the cohort.

(An earlier note in this document said caching was absent outright. That was
generalised from the first three models measured, which all happened to be 0%,
and it was wrong. The per-model table is the fact.)

This is why cost is now a **measurement rather than an estimate**.
`Endpoint.usage_total` accumulates the `usage` blocks the provider actually bills
on, under a lock, because episodes run concurrently and a `last_usage` field
would report whichever thread finished most recently. Every per-model figure
quoted for round 2 comes from a real 12-episode cell, cache hits included, not
from a token model.

Measured shape: **~38,000 prompt tokens per 30-step episode** against the plan's
39,188 estimate — the token model was good; only the discount was guesswork.
Input dominates: non-reasoning models emit ~100 completion tokens per episode.

### 2. A missing User-Agent 403s the entire run.

`urllib` sends no User-Agent. The provider sits behind Cloudflare, which answers
a UA-less POST with `HTTP 403 error code: 1010` — a browser-signature block, not
authentication and not a throttle. 403 is inside the fatal-4xx range, so no retry
could recover it, and the error surfaced as what looked like a credentials
problem. **Every request in the sweep would have failed this way.**

### 3. The warm-up cap was the TRAP 4.1 condition.

`run_fidelity` negotiates the request form once before any concurrency, and did
so with `max_tokens=4`. A reasoning model spends four tokens thinking, returns
empty content with a populated `reasoning` field, and the endpoint raises. That
call sits outside `one_run`'s try/except, so it takes the whole cell.

`Endpoint.probe` had the same defect at `max_tokens=8`: a probe whose purpose is
to catch a checkpoint that produces nothing was manufacturing that condition.

### 4. Most of the planned cohort is not serverless on this account.

Of 169 chat models, 73 carry a nonzero price, and **21 are actually reachable**.
Forty-seven answer `Unable to access non-serverless model` — including nearly the
entire planned cohort: Llama-4-Scout, Llama-4-Maverick, Qwen3-Next-80B in both
Instruct and Thinking variants, Qwen3-235B, DeepSeek-V3.1. GLM-4.5-Air returns
503. **Pricing in `/v1/models` does not imply availability** — Llama-4-Scout is
priced at $0.18/$0.59 and still refused.

The endpoint's negotiation error made this much harder to see than it needed to
be: it read `every request form was rejected by the endpoint`, which describes
the retry loop and implicates chat templates, when the server had said something
completely different. It now quotes the server.

## The token cap, and why 512 was wrong

`EDEN_MAX_TOKENS` is **2048**, raised from 512.

A single call at 512 gets a clean, in-vocab command out of both gpt-oss
checkpoints — that is what the one-call smoke showed, and it was not enough.
Reasoning length grows with the conversation, and by step 4 or 5 the cap lands
mid-thought: empty content, populated `reasoning`, endpoint raises. **12 of 12
episodes lost, on a model that had just passed the one-call smoke.** A cap
validated on step 0 is not validated, which is why there is now a second smoke
that runs twelve real episodes through the full runner.

Raising the cap is nearly free *here* and would not be elsewhere: with no
caching, input outnumbers output roughly 380:1 per episode, so the output cap is
not where the bill is.

**`reasoning_effort="low"` was rejected.** It cuts gpt-oss reasoning from 574
characters to 42 and would have been cheaper. It is a per-model serving knob that
changes what is being measured, and round 2 exists to raise the capability of the
cohort, not to cripple the half of it that thinks.

## Noise floor

The two-tier amendment requires a per-model floor measured empirically, since
`VLLM_BATCH_INVARIANT` is unavailable. The verification plan requires **five or
more repeats on one model, or no serving-noise claim at all** — two repeats
cannot separate serving nondeterminism from binomial sampling, and at m=24 with
p≈0.2 the sampling SD alone is 0.082, which would make a two-repeat "floor"
`determinism_map`'s n=4 wearing new clothes.

If five repeats are not affordable inside the budget, the read reports Wilson
intervals and **makes no serving-noise claim**. It does not report a floor it
cannot support.

## What does not change

The worlds, the ladder, the arithmetic, the fact-scored outcome, the funnel, and
every survivability guarantee are computed offline and are unaffected by where
the model is served. `eaten(<item>: f)` in the fact stream remains the only thing
that counts as eating.
