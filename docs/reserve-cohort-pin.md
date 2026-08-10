# Reserve cohort — pinned list, bench, and substitution rule

Pinned in the commit adjacent to the PHASE SWITCH entry
(`docs/research-log.md`, 2026-08-09), **before any anchor iteration begins**.

The entry's sequence puts reserve enumeration after the three-anchor survey,
while the reserve paragraph requires the list pinned before anchor iteration —
and the survey *is* anchor iteration. Split to satisfy both: **selection is
pinned now**, since it is by frozen criteria that no anchor result can touch;
**loop tests run later**, on a warm GPU. That ordering is safe because a loop
test is subtractive and mechanical — it removes checkpoints that cannot hold a
parse loop, and cannot promote one that the criteria did not already select.

## The frozen criteria, restated

1. 4–6 checkpoints.
2. Family coverage including **at least one family absent from the dev seven**.
3. **At least one base/instruct pair.**
4. A4-style loop test before inclusion.
5. **Never selected by any containment-adjacent result.**

The dev seven — Alibaba, MistralAI, AI2, IBM, TII, Meta, Google — are *all*
instruct checkpoints (`scripts/gpu_job18/entrypoint.sh`), so criterion 3 cannot
be satisfied from any family already burned without adding a base half.

## The slate

| # | checkpoint | family | role |
|---|---|---|---|
| 1 | `01-ai/Yi-1.5-9B-Chat` | 01.AI | pair, instruct half |
| 2 | `01-ai/Yi-1.5-9B` | 01.AI | pair, base half |
| 3 | `microsoft/Phi-3.5-mini-instruct` | Microsoft | fresh family |
| 4 | `internlm/internlm2_5-7b-chat` | InternLM | fresh family |
| 5 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | HuggingFaceTB | fresh family, small |

All five families are absent from the dev seven, so criterion 2 is satisfied
four times over rather than once. All verified present and ungated on the Hub at
pin time.

**On SmolLM2-1.7B-Instruct — recorded because the writeup must state it rather
than a reviewer finding it:** *SmolLM2-1.7B-Instruct was selected expecting low
containment, so the reserve look's non-degeneracy is partly engineered by
composition rather than discovered. This is acceptable for the possibility
claim, whose hard part is stability rather than non-degeneracy, and is recorded
here so the writeup states it rather than a reviewer finding it.*

## The bench, ordered

A loop-test failure must not send us shopping for a replacement after survey
results exist — at that point the substitution is no longer mechanical. The
bench is pinned now, in order, and a failure consumes the next entry and nothing
else.

| order | checkpoint | family | replaces |
|---|---|---|---|
| B1 | `deepseek-ai/deepseek-llm-7b-chat` + `deepseek-ai/deepseek-llm-7b-base` | DeepSeek | a **pair** failure |
| B2 | `zai-org/glm-4-9b-chat` | Zhipu / Z.ai | the next singleton |
| B3 | `stabilityai/stablelm-2-12b-chat` | Stability | the singleton after that |

`zai-org/glm-4-9b-chat` is the current canonical id; `THUDM/glm-4-9b-chat`
307-redirects to it.

### A pair failure consumes a pair

If **either** Yi half fails its loop test, **both** come out and B1 enters
whole. The base/instruct criterion cannot be repaired by a singleton, so letting
one half fall through to the next single name would silently drop criterion 3
rather than satisfy it. This is the one substitution that is not
one-name-for-one-name, and it is written down here so it does not have to be
decided later.

## Standing constraints on the reserve

- **Reserve models never enter the sandbox.** Nothing moves from reserve to
  sandbox; sandbox additions are burned by definition and are a separate list.
- **At most two looks, ever**, and only once a candidate instrument has cleared
  the possibility bar.
- Loop testing is **not** a look. It checks that a checkpoint can hold a
  parseable action loop; it produces no containment number and touches no
  anchor.
- `world_v3` is the other held-out reserve: it may be authored at any time, but
  must not be swept until after the freeze. Authoring does not burn it; play
  does.
