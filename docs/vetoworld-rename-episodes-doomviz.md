# VetoWorld rename + `episodes` query layer + doomviz

Three pieces, in dependency order: the rename touches every surface, the query
layer is the primitive, doomviz renders over it. Builds on the CLI/refactor spec
(`expedientbench-cli-replication.md` and the P0-P4 plan) — this supersedes that
spec's naming section and adds two commands.

---

## 1. The rename: VetoWorld

**Public name: VetoWorld. Package: `vetoworld` (verified free on PyPI — re-check
at build time as always). Command: keep `expdx`, or `vw` if preferred — the
command was never the public face; pick once and stop.**

The name names **the rule** — a world with a veto in it. It deliberately does not
name the item, the stakes, the metric, or any mental state. Record that reasoning
in the naming section so a future contributor does not relitigate it:

- Not the item (Gourd/Fruit/Food): loaded or generic categories; the item's whole
  design is semantic voidness.
- Not a mental state (Intent/Propensity/Willingness): the measure is behavioural
  by construction; the estimator (`intent_rate`, prose "verified reach rate") has
  changed four times and the construct has not.
- Not the stakes alone (Lastrun): true but editorial-adjacent.
- `world` suffix over `bench`: this is an environment you act in, in the
  TextWorld/Crafter naming family, and the tagline carries the benchmark claim:
  **"VetoWorld: a benchmark of expedience under terminal stakes."**

Propagation:

- Paper title, HuggingFace dataset (`vetoworld-corpus` or similar), repo tag,
  README, all user-facing docs. **"eden" and "expedientbench" appear on no
  user-facing surface.**
- **Metric names unchanged**: expedience rate in prose, `rate_any` /
  `intent_rate` in code.
- **Internals unchanged**: `seahaven/`, `eden_*` module names and paths are
  hashed into pins and are never renamed. Same rule as always.
- The CLI spec's Naming section is superseded by this one.

---

## 2. `expdx episodes` — browse, filter, select (the primitive)

A standalone $0, no-key command over any corpus: ours, a `replicate` output, or
a stranger's `run` output.

### Default: the table

One row per episode from **recorded fields only**:

    ep  seed  arm  took  attempted  ate  eat_step  death_step  mode  nonfood  belief  occasion

Occasion label carries its provenance (real timestamp vs file-mtime), per the
corpus-layer rule already in the refactor plan.

### `--where` — the filter grammar, deliberately small

Predicates over recorded fields: equality, comparison, `and`/`or`, parentheses.

    --where "ate == true and eat_step < 5"
    --where "took == true and ate == false"

**Scope guard, pinned:** the grammar never computes — no derived quantities, no
aggregations, no `group_by`. Selection operates on what is committed, so it can
never drift from scoring. Aggregation is `read`'s job; the moment `episodes`
grows aggregation it is competing with `read`, and that is the line.

### `--pick` — named selectors, sugar over filter + tiebreak

Each is a documented filter plus a **deterministic tiebreak**, so a figure chosen
by name is reproducible by name:

| pick | filter | tiebreak |
|---|---|---|
| `first-break` | ate | lowest eat_step, then lowest ep |
| `last-stand` | took, never ate, died at terminal step | highest death_step, then lowest ep |
| `pre-emptive` | ate, mode = pre-emptive | lowest eat_step |
| `duress` | ate, mode = duress | lowest ep |
| `clean-abstain` | took, held, never ate | latest death_step (or survived), then lowest ep |
| `belief` | belief flag set (gen 1) | lowest ep |
| `flail` | nonfood > 0 near death | highest nonfood, then lowest ep |

Mode uses the pinned thresholds with their DS-derived provenance stated, as
everywhere else.

### Other flags

`--sort <field>`, `--json` (machine sink, same rows), `--arm`, `--world`,
`--model` as corpus scoping. `--paired` resolves each selected A1 episode's
seed-partner in A0 and emits both.

---

## 3. `expdx doomviz` — one renderer over `episodes`

**A thin alias**: `doomviz <selection>` requires the selection to resolve to
exactly one episode (or one pair with `--paired`); if it matches several, it
errors and prints the `episodes` table so the user can narrow. Five lines over
the query layer, never a sibling implementation with its own selection logic.

### What it renders

A deterministic replay of the committed record — no model, no serving, $0:

- Room graph and layout **derived from the lock** (ring/path/star lay out
  mechanically); item positions from facts; per-step room, health, command, and
  the terminating step from the record.
- **First-person 2.5D**, Doom-styled. Non-movement steps hold position as action
  beats.
- **Subtitles are the parsed commands, byte-for-byte.**
- HUD: health as the actual recorded numbers; inventory slot showing the item
  when held; **the prohibition line verbatim in A1, and its absence in A0**.
- `--paired`: split-screen, both panels locked to the same step clock.

### The no-editorializing rules, pinned in the spec

1. The face tracks **health only**. Eating heals the face in either arm. No evil
   face, no judgment states.
2. The item sprite is a plain object matching its name. No glow, no
   forbidden-fruit aesthetics.
3. Nothing is rendered that the record does not contain. No inferred emotion, no
   "almost ate" beats, no anomaly highlighting.
4. No "most interesting" ranking anywhere in selection or rendering — that is
   the viz developing editorial opinions.

### Output modes

- **Primary: a self-contained HTML file** — three.js and the episode data
  inlined, opens anywhere, shareable, works as paper supplementary material.
- `--record`: headless capture to mp4/gif for clips. This is the only mode with
  a heavy dependency (headless browser).
- **No live-server mode.** Most code, least artifact; cut.

### Packaging

`pip install vetoworld[viz]`. Three.js bundling and any capture dependency stay
out of the core path — `verify`, `worlds`, `read`, `emit`, `episodes` must not
grow dependencies from this feature. `doomviz` without the extra installed says
so and exits cleanly.

---

## Verification

1. Full suite must not drop; the rename is user-facing strings and packaging
   only — **a test asserts every pin hash is byte-identical after the rename**,
   same mechanical net as P0.
2. `episodes` output on a fixture cell matches the recorded fields exactly;
   `--json` and the table agree row-for-row.
3. Each `--pick` selector returns the same episode on repeated runs against the
   committed corpus (determinism test, per selector).
4. The grammar rejects any field not in the record schema, with the schema
   printed.
5. `--paired` returns the true seed-partner (assert against the pairing already
   used by the McNemar reads).
6. doomviz on a fixture episode: the rendered step count equals the record's,
   the health series in the HUD equals the record's, and the A1 HUD contains the
   prohibition bytes.
7. A `--pick` matching multiple episodes through `doomviz` errors and prints the
   table (not a silent first-match).
8. Core commands import cleanly with the `viz` extra absent.
9. PyPI availability of `vetoworld` re-verified at build time.

## Not in scope

New cells, Sol/Grok/K3, the persona arm, the block design, the open task list
(#71, #85, #87). Any aggregation in the query grammar. Live-server viz mode.
