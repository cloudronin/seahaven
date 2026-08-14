# DS-V4-Flash instability — serving-stack diagnostic

## What is being explained

    DS-V4-Flash, LAT, terminal death, same pin, same seed block
      block 1 (round 10)      27/72 = 0.375
      block 2 (two days later) 19/24 = 0.792     Fisher p = 0.0007

Localized entirely to one funnel stage and one mode:

    took             72/72 -> 24/24   pinned, cannot move
    ate-given-took   27/72 -> 19/24   p = 0.0007   all of it
    pre-emptive  (step <= 2)   0.194 -> 0.292   p = 0.3934   did not move
    under duress (step >= 16)  0.181 -> 0.500   p = 0.0059   moved

**Already ruled out.** Payload construction byte-identical across both serving
paths, same `EdenPolicy` (`max_tokens=2048`, `temperature=0.9`,
`seed*100003+step`), parsed-command history on both. No provider signal: token
delta fully explained by episode length, `reasoning_tokens` 0 in both halves,
reply shape stable outside a six-reply tail (expected 2.25, P(0) = 0.106).
cogito and gemma probed, neither moved — though gemma takes the item in 0% of
episodes so it never engages the decision, and cogito is the control that carries.

**The hypothesis now under test: the difference is in the serving stack, not the
model's disposition.** `docs/eden-round2-hosted.md` recorded at the time that
`VLLM_BATCH_INVARIANT` is unavailable on a hosted API and that a per-model noise
floor would have to be measured empirically. That caveat has now arrived as a
measurement.

Two candidate mechanisms, and they make different predictions:

| Mechanism | Predicts |
|---|---|
| Per-request batch composition | Noise averages out within a block; cannot produce two blocks differing at p = 0.0007 |
| KV-cache / prefix-cache state | Persists across a block, since cache state is a function of what the server recently saw. **Can** produce block-level disagreement |
| Deployment change (replica, quantization, rehost) | Persists across days; no within-day signal |

---

## Step 1 — cache-hit fraction, $0, do this first

`usage` is captured per cell and includes `cached_tokens`. The conversation is
append-only and resent whole each step, so it is perfectly prefix-cacheable and
the hit rate should be high and stable if nothing changed.

Compare, per cell:

    cached_tokens / prompt_tokens     DS-V4-Flash block 1  (72 episodes)
                                      DS-V4-Flash block 2  (24 episodes)
                                      cogito probe block   (did not move)
                                      gemma probe block    (did not move)

**If DS-V4-Flash's shifted block ran at a markedly different hit rate**, that is
the strongest evidence available without provider cooperation. Report the
distribution, not just the mean — a block that started cold and warmed up looks
different from one that was warm throughout.

Also check whether `cached_tokens` is populated at all for this provider. If it
comes back absent or zero everywhere, this read is unavailable and say so rather
than inferring from a missing field.

## Step 2 — trajectory state at the decision, $0

The only remaining place a mundane explanation could hide: **the two blocks may
have reached the decision in systematically different states**, in which case the
decision is not what changed.

Per block, for episodes that reached the under-duress window (step >= 16):

- Rooms visited before the eat
- Legal food eaten before the eat
- Health at the step of the eat, and at step 16
- Step at which the item was taken (expected 0 in both — it is pinned)
- Crossing step per episode (from the recorded health trace, via the consolidated
  `crossing()` helper — not a formula)

Two-sample tests on each, with n stated. **If the later block's episodes arrived
at the decision in a different state, the shift is upstream of the decision.**

This is the same check that dissolved cogito's normalized slopes.

## Step 3 — timing-structured probe (~$0.45)

Only if steps 1 and 2 come back clean.

**The design varies the thing now suspected, rather than sampling more.**

    Block A   24 episodes, DS-V4-Flash, LAT, A1
    Block B   24 episodes, DS-V4-Flash, LAT, A1   — immediately after A, same session
    Block C   24 episodes, DS-V4-Flash, LAT, A1   — different time of day, >= 6h later

Fresh seeds per block, asserted disjoint against disk. `terminal_at_zero` TRUE in
every meta. Record wall-clock start and end per block and the cache-hit fraction
per block.

**Pre-committed reading, fixed before any cell:**

| Pattern | Reading |
|---|---|
| A ~= B, and C differs from both | Deployment or session state. Persists within a session, changes across them |
| A differs from B within the same session | Finer-grained than session — cache or batch state at request granularity |
| A ~= B ~= C, all near 0.375 | Block 2 was the outlier; DS-V4-Flash's central estimate is the original |
| A ~= B ~= C, all near 0.792 | The model shifted and stayed shifted |
| All three differ | Unstable at block granularity; the rate is reported as a range, never a point |

Read **split by mode** (pre-emptive vs under duress), not pooled — pooled it
inherits the mixture and cannot test the specific claim. Report every block
separately with pairwise tests and **each MDS printed**; at 24 vs 24 near p = 0.4
the MDS is roughly 0.28, so a null means "no shift larger than about a quarter".

**These are observations, not arbiters.** Do not take a majority of five blocks.

## What each outcome costs the program

**If it is cache or batch state:** it is a nuisance variable no pin can cover, and
it becomes a stated limitation on every hosted result. Mitigations are
randomizing episode order across cells, or measuring the noise floor it produces
and reporting rates with it attached. Neither is free, and the second is the
per-model noise floor round 2 said would be needed.

**If it is a deployment change:** the pin covers your code and your worlds and
cannot cover their serving stack. Every cross-day comparison in the program
inherits that, and it belongs in the writeup rather than in a footnote.

**If it is the model:** DS-V4-Flash's rate is a range and the undamped
repeat-measures study becomes the next round rather than a deferred one.

## Verification

1. Full suite — 902 passed / 2 xfailed now, must not drop.
2. Round 10's hashes verify before any new cell is pooled into its corpus; if
   Stage B has already retired round 10, these blocks are a new round and must be
   pinned as such.
3. `terminal_at_zero` TRUE in every new cell's meta.
4. Seed values disjoint from every burned block, checked against disk.
5. Crossing computed per-episode from the recorded health trace via the
   consolidated helper, not from either legacy formula.
6. Wall-clock timestamps and cache-hit fraction recorded per block, since they are
   the independent variable.

## Ordering note

**Run this before Stage B.** Retiring round 10 permanently closes the ability to
add cells to its corpus, and blocks A and B belong there if they are to be
compared directly against block 1 under the same pin.

## Cost

    Step 1  cache-hit read       $0
    Step 2  trajectory state     $0
    Step 3  timing probe        ~$0.45
