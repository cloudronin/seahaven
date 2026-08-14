# `expedientbench` CLI — make every number in the paper recomputable by anyone

## The design principle

The tool separates two claims that replication usually conflates:

1. **The paper says what the data says.** Byte-exact, $0, achievable by anyone.
2. **A new run lands where ours did.** Achievable only within bands, because the
   program's own findings say so: a 0.319 between-occasion shift on one model
   with mechanism unresolved, hosted serving that is not batch-invariant, and
   temperature 0.9. A replicator who expects point estimates will "fail to
   replicate" and be wrong about what that means.

The CLI encodes that distinction as two different commands with two different
success criteria.

## Commands

### `vworld verify` — recompute the paper, $0

- Fetches the committed corpus on first run (see Packaging).
- Recomputes **every figure that appears in the manuscript** from cells on disk:
  the master matrix, both metrics and the gap, funnel stages, floor mechanisms,
  band assignments, direction-preservation counts, corrections-ledger numbers,
  spend totals from accumulated `usage`.
- Checks each against the **claims register** (below). Exits nonzero on any
  drift, printing the figure, the manuscript value, and the recomputed value.

This is the strong replication claim and it is exact.

### `vworld replicate` — serve new cells, judged against bands

- Runs any cell (model, world, arm, m) against a provider with the **user's own
  key**. Same `EdenPolicy` assembly, same pins, same preconditions, same seed
  formula; fresh seeds by default, `--seeds original` to reuse ours.
- **Judged against a pre-stated replication band per cell, never the point
  estimate**: Wilson interval on our n, widened by the measured occasion
  component where one exists. Bands ship in the corpus as data, computed once,
  not improvised at run time.
- Output: PASS (inside band) / FAIL (outside) / VOID (a precondition failed),
  with the precondition table printed either way. A0 >= 0.90, `saw == n`, parse
  0.0%, non-food profile — the same gates as every round.
- Prints running spend per cell; refuses to start without an explicit `--budget`.
- Records the resolved model string and any version identifier the provider
  returns, per cell, as ours do.

Durability facts, stated in the docs rather than discovered in issues:

- The **open cohort** is durably replicable in principle — weights exist — though
  exact-string availability on any one provider will churn. The tool checks the
  string against the provider at run time and refuses near-misses.
- The **Terra cell** replicates only as "current Terra, within band." Closed
  weights, version-churned serving, mid-cycle behaviour updates. Say so.
- The **temperature deviation** on the Terra corpus (1.0 vs 0.9) is carried in
  the cell metadata and printed by any command that touches those cells.

### `vworld worlds` — the $0 gates, standalone

Runs the full world-validation stack on any lock, ours or a new one:

- Held-Karp optima (order **and timing** — the corrected optimizer), greedyMin,
  S per level
- both necessity legs, with forbidden-sufficient **as a deadline** under
  terminal death
- lock consistency (`assert_lock_consistent`), head-noun disjointness, the
  numbered-name guard, the room-text audit against the entity list
- crossing step per world from the consolidated helper

This is the piece most likely to be reused by other people building worlds.

### `vworld read` — the standing reads on any corpus

Funnel decomposition, `rate_any` and `intent_rate` with the gap, the A0 licence
gate (suppressing `ate_given_took` on failure), mode split with the pinned
thresholds and their DS-derived provenance stated, preconditions, occasion
labels. Works on our corpus or on cells the user generated with `replicate`.

## The claims register (build this first)

One module: every figure in the manuscript is emitted by a named function from
committed cells. The manuscript cites the function; `vworld verify` recomputes all
of them; a regression fails if any quoted figure drifts.

Register rows carry: figure id, value, emitting function, cells consumed,
generation, measured-vs-derived, occasion label.

This is the structural-rigor principle applied to the paper itself, and it is
what would have caught "then a gap" and 93-vs-92 mechanically.

### Register outputs — the paper's tables, emitted not maintained

`vworld emit <artifact>` renders each of these from the register; `vworld verify`
recomputes them all. None of these is a hand-maintained document.

| Artifact | Content |
|---|---|
| `matrix` | The master table: model x world, both metrics, gap, funnel stages, n, Wilson, generation, measured-vs-derived flag, occasion label |
| `corrections` | The corrections ledger: claim, round published, round retracted, mechanism, what caught it, cost — each row verified against commits |
| `predictions` | Everything pinned before data and how it landed: G3 Cliff floor, the raidex say-side prediction, the LAT2 derived values, the round-8 pinned prediction, the band rules |
| `generations` | The three generations: what changed at each boundary, why, and what does not pool |
| `floor-mechanisms` | The three routes to zero: never-touches, takes-and-returns, carries-and-excludes, with the funnel evidence per model |
| `occasions` | **The occasion audit — currently unrun.** Which comparisons in the paper span serving days, flagged in place. Mandatory given the 0.319 shift |
| `spend` | Total from accumulated `usage`, not the running estimates |
| `seeds` | The burned-block registry, consolidated |
| `related-work` | The contrast table: pressure channel, choice offered vs constructed, cost asserted vs enforced, counterfactual per-episode vs group |
| `limitations` | One phrasing; cohort accretion; capability open with both routes documented closed; occasion component with mechanism unresolved; worlds authored not sampled; mode thresholds DS-derived |
| `disclosures` | Claude Code ran the harness; Claude judge inside two raidex dimensions; gpt-4o-mini judging StrongREJECT; competitor model as subject |

### Pins carried as corpus data, checked by `verify`

- **Prompt fixtures**: assembled A1/A0 bytes per world per generation, committed;
  `verify` asserts assembly still reproduces them (also verification item 5).
- **Primary-metric declaration**: `intent_rate` primary, `rate_any` beside, with
  the pre-commit date — the date is what makes the ordering provable against the
  floor result landing after it.
- **The Terra temperature deviation** (1.0 vs 0.9) in cell metadata, printed by
  any command touching those cells (as above).
- **Repo tag + corpus hash** at submission, cited in the manuscript.

## Naming

- **Package: `expedientbench`. Console command: `vworld`.** The public name carries
  the construct (expedience), not the estimator — `intent_rate` is the current
  primary metric and estimators have changed four times; the construct has not.
- **Do not** name anything public after "intent": it is a mental-state noun the
  design refuses to claim (the metric is a *verified reach rate*, behaviourally
  defined), and in ML it reads as intent-classification NLU.
- Repo internals stay `eden_*` unchanged — they are hashed into pins, and the
  internal/external vocabulary split is established practice in this program.
  The name `eden` appears in **no** user-facing surface: command, package,
  dataset, or docs.
- HuggingFace dataset name matches the package: `expedientbench-corpus` or
  similar, not `eden-*`.
- **Verify `expedientbench` is free on PyPI at build time** (and `vworld` as the
  console script has no conflicting binary on a clean PATH). Tonight's search
  established `xbench` is taken by an existing AI-evals framework and `eden` was
  never a candidate; the final string still gets checked against the index, not
  against a conversation.

### `vworld run` — new cells on any model, including yours

`replicate` re-serves **our** cells and judges against bands. `run` measures a
**new** model — one we never ran — with the same instrument: any world, any arm,
same pins, same preconditions, same reads. No bands exist for a model we never
measured, so output is the model's own table (both metrics, funnel, preconditions),
not a PASS/FAIL.

This is the command that makes the benchmark usable on **fine-tuned and
self-hosted models**: point it at any OpenAI-compatible endpoint. Note for
self-hosters: your own vLLM with fixed batching is the only serving mode in which
the between-occasion component can actually be pinned — the hosted-API occasion
caveat is a property of hosted serving, not of the instrument.

## Backend abstraction (the raidex pattern, minus the translation layer)

One `LLMBackend` interface behind all commands, following raidex's shape —
provider-prefixed model strings, keys from env vars only — with one deliberate
difference: **no litellm or other per-provider translation layer in the serving
path.** The serving diagnostic ran on byte-level payload diffs; every byte
between `EdenPolicy` and the wire must remain auditable. The core client is the
round-13 generic **OpenAI-compatible** client, which covers Together, OpenAI,
xAI, Modal, Moonshot, and any self-hosted vLLM/Ollama endpoint — including
fine-tuned models.

- **Endpoint config**: `--base-url` + `--model` + key env var, or a named entry in
  `endpoints.toml` (name, base_url, key env, model string, notes). Raidex-style
  `provider/model` strings resolve through that file.
- **Conversation assembly is shared and provider-agnostic**: one `EdenPolicy`,
  one seed formula, temperature pinned at 0.9. The message-assembly identity test
  (round 13) runs against every configured backend — a fixture seed must produce
  byte-identical messages regardless of endpoint. Any parameter deviation (e.g.
  Terra's temperature 1.0) is recorded in cell meta and printed wherever those
  cells are read.
- **Model-string discipline adapts to the endpoint class.** Catalogued providers:
  assert against the availability record, refuse near-misses, as always.
  Custom/fine-tuned endpoints have no catalogue — so the rule becomes
  **record-and-pin**: capture the served model string and any version identifier
  the endpoint returns, write it into every cell's meta, and refuse to pool cells
  whose recorded strings differ.
- **Per-backend metadata capture**: `usage` including cached-token fields where
  present, wall-clock per cell, resolved model string. The occasion caveat
  applies **per stack**, independently.
- **Preconditions and the parser are backend-invariant.** A fine-tuned model that
  cannot produce parseable commands is reported as parse-failure rate — a result
  about that model under this interface — never patched with model-specific
  parsing.
- Anthropic/Google native APIs are **out of scope for the core**; if those cells
  are ever wanted, they arrive as explicit adapters with their own assembly
  identity tests, not through a translation dependency.

## Packaging

- **pip-installable**, console command `vworld`. The repo's existing modules become the
  library; the CLI is a thin layer over functions that already exist
  (`eden_read`, the simulator, the gates, the pins).
- **Corpus on HuggingFace** (the raidex pattern): all committed cells, locks,
  BUILD.lock files, pins, the claims register, and the replication bands.
  `vworld verify` fetches and checksums it. The paper cites the dataset DOI and
  the repo tag.
- **Repo tag at submission**: code + corpus hash pinned in the manuscript.
- Keys via environment only. `verify`, `worlds`, and `read` must run with **no
  key at all** — $0 commands cannot require credentials.

## Verification for this work itself

1. Full suite must not drop; the CLI is a layer, not a rewrite — a test asserts
   `vworld read` output matches the existing round-read output byte-for-byte on a
   fixture cell.
2. `vworld verify` green on the committed corpus before the manuscript quotes it.
3. A synthetic drift test: perturb one cell in a scratch copy, `verify` must
   fail and name the figure.
4. `vworld replicate --dry-run` and `vworld run --dry-run` assemble and print a
   cell's first request without serving it (the payload-diff machinery, reused) —
   against any configured backend, including a custom `--base-url`.
5. Prompt fixtures: assembled A1/A0 bytes per world per generation, committed,
   and `verify` checks assembly still reproduces them.
6. The banned-command test: `replicate` refuses a near-miss model string, with
   the availability record cited.

## Out of scope

New cells, new worlds, Sol/Grok/K3, the persona arm, the block design. This is
packaging and proof, not measurement.
