# The sealed partition — gate zero of the dimensional program

**Seal hash:** `d86c105c0768ea676e3e6d367820d97adcb472eb9d6271ba69fb2b16f29d6359`

Locked before any exploration model is loaded. The held-out twelve are
never loaded, never inspected, never analysed, until Phase 2.

What is sealed is not twelve names but *twelve names that satisfy the
coverage facts below* — the justification is recomputed from the sealed
data and hashed alongside the lists, so the coverage claim cannot drift.

**Capability window: 20–25 MMLU-Pro.** At 5 points this is ~12% of the
cohort's 7.75–43.38 range, so members are genuinely **capability-matched**.
It was briefly widened to 18–28 on a candidate partition where held-out
could hold at most two of it; requiring proxy coverage then forced all nine
uncovered models into exploration, the covered ones redistributed, and the
narrow bin became viable. Kept narrow because it licenses the stronger
phrase — at the cost of 3 held-out members rather than 5.

| | EXPLORATION | HELD-OUT |
|---|---|---|
| models | 18 | 12 |
| families | 7 | 5 |
| all proxy-covered | False (9 uncovered, by design) | **True** |
| 20–25 window | 3 models, 2 families | 3 models, 3 families |
| base checkpoints | 6 — Mistral, Qwen2.5, Qwen3 | 2 — Qwen2.5 |
| intact base/instruct pairs | 6 | 2 |

## HELD-OUT (12) — sealed

| model | family | MMLU-Pro | in window |
|---|---|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | Qwen2.5 | 43.38 |  |
| `tiiuae/Falcon3-7B-Instruct` | Falcon3 | 34.30 |  |
| `ibm-granite/granite-3.1-8b-instruct` | Granite-3.1 | 28.19 |  |
| `tiiuae/Falcon3-3B-Instruct` | Falcon3 | 22.28 | yes |
| `Qwen/Qwen2.5-1.5B` | Qwen2.5 | 20.61 | yes |
| `ibm-granite/granite-3.1-2b-instruct` | Granite-3.1 | 20.21 | yes |
| `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2.5 | 19.99 |  |
| `allenai/OLMo-2-1124-7B-Instruct` | OLMo-2 | 18.58 |  |
| `google/gemma-2-2b-it` | Gemma-2 | 17.22 |  |
| `Qwen/Qwen2.5-0.5B` | Qwen2.5 | 10.06 |  |
| `tiiuae/Falcon3-1B-Instruct` | Falcon3 | 9.32 |  |
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2.5 | 7.75 |  |

## EXPLORATION (18)

Includes all four burned smoke-test models and all nine models without a
pinned-proxy number.

| model | family | MMLU-Pro | note |
|---|---|---|---|
| `tiiuae/Falcon3-10B-Instruct` | Falcon3 | 38.10 | burned |
| `Qwen/Qwen2.5-7B` | Qwen2.5 | 37.39 |  |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 | 36.52 |  |
| `google/gemma-2-9b-it` | Gemma-2 | 31.95 | burned |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama-3.1 | 31.09 | burned |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2.5 | 25.05 |  |
| `Qwen/Qwen2.5-3B` | Qwen2.5 | 24.48 |  |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | 23.06 | burned |
| `mistralai/Mistral-7B-v0.3` | Mistral | 21.70 |  |
| `Qwen/Qwen3-0.6B` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-1.7B` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-1.7B-Base` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-4B` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-4B-Base` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-8B` | Qwen3 | — | no proxy |
| `Qwen/Qwen3-8B-Base` | Qwen3 | — | no proxy |
| `allenai/OLMo-2-0425-1B-Instruct` | OLMo-2 | — | no proxy |
| `allenai/OLMo-2-1124-13B-Instruct` | OLMo-2 | — | no proxy |

## The limitation the cohort cannot fix

Base/instruct is Qwen-concentrated and family-confounded. Held-out base checkpoints are Qwen2.5 only, because the sole non-Qwen pair (Mistral-7B) has its instruct half burned. The base/instruct axis is therefore EXCLUDED from Phase 2's confirmatory set — confirming it on held-out would confirm a Qwen property, not a base/instruct one.

This is recorded rather than pretended solved: the phased spec asks that
base/instruct pairs split so neither set is Qwen-only, and the cohort
provably cannot. The only non-Qwen pair is Mistral-7B and its instruct
half is burned.

## Do not rescue the proxy gap

Nine models have no MMLU-Pro on the leaderboard pinned at `4d32c08`
*before any lookup*. Substituting HELM, a model card, or any other source
to fill them — from anyone — is the proxy-shopping the pin exists to
prevent, and it is tempting because Qwen3 is otherwise the cleanest ladder.
