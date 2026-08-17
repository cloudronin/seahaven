---
license: mit
pretty_name: VetoWorld corpus
task_categories:
  - reinforcement-learning
  - text-generation
language:
  - en
tags:
  - agents
  - alignment
  - evaluation
  - textworld
  - benchmark
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: results/*.json
---

# VetoWorld corpus

The committed cells behind *VetoWorld: a benchmark of expedience under terminal
stakes*, plus everything needed to recompute the paper from them.

    pip install vetoworld
    vworld corpus fetch          # this dataset, checksummed on arrival
    vworld verify                # every figure recomputes, $0, no key

`verify` recomputes every quoted figure and exits nonzero naming any that
drifted. It needs **no API key and costs nothing**.

`corpus fetch` pulls the cells over plain HTTPS — **no `huggingface-cli` and no
`huggingface_hub`**, because this package declares no runtime dependencies and
the verb a replicator runs first is the wrong place to break that. It stages the
download, checks the digest against the manifest, and installs nothing that does
not match: a corpus that arrived corrupted would otherwise make `verify` report
drifted figures, which is a statement about the network dressed as a statement
about the manuscript.

It also pulls `results/raidex_pool.json`, the frozen index axis every correlate
is plotted against. That file is **not** part of the manifest digest — the
digest is over cells — so it is fetched beside them and reported separately. It
was missing from this dataset for weeks, during which `verify` exited nonzero
for anyone who fetched, on a figure that recomputed perfectly in the authors'
own working tree. A digest that matches over an incomplete file set only tells
you the incomplete set arrived intact.

## What is in here

| | |
|---|---|
| `results/` | One JSON per (round, model, arm, world), each holding every episode's full command trace with per-step health, room, parse status and funnel flags. The count grows with every round — `vworld corpus status` prints it, and this card deliberately does not, because a number written here is wrong by the next sweep. |
| `worlds/` | The compiled `.z8` worlds, their `.json` sidecars, and `BUILD.lock.json` per world — topology, larder, params, derived block, and the `.z8` sha256. |
| pins | Eleven frozen payload digests, two live and eleven retired, each recomputable from literals inside its round module. |
| claims register | One named function per manuscript figure. The paper cites the function; `verify` runs it. |
| replication bands | Per-cell Wilson intervals widened by the measured occasion component. Computed once and shipped as data, never improvised at run time. |
| prompt fixtures | The assembled A1/A0 bytes per world per generation, so prompt assembly can be checked byte-for-byte. |

**Manifest digest: `fb2234415b322451`.** Cite it beside the repo tag. The cell
count that used to sit next to it is not written here — `vworld corpus status`
prints it, and the digest is the thing that has to be exact.

## Four things to know before using it

**0. Nothing pools across a PROVIDER boundary either.** Most cells were served
by Together; one round was served through the HuggingFace router on DeepInfra,
and one model is OpenAI's. Every cell records the provider that **answered**,
taken from the response rather than from what was requested. A model's rate at
a level from one provider may not be compared to the same model's rate at that
level from another: serving stack, hardware, quantisation and request handling
all differ, and this corpus has measured that the last of those can turn
content into silence. Cross-provider deltas measure providers, not models.

The register is mixed-provider by necessity — seven models exist in it only
because a second provider would serve them — and it states the provider **per
row**. Each model appears once, from one provider, so no published row is that
forbidden comparison. What a mixed cohort does cost is that a rank difference
between two rows served by different stacks carries a component that is not the
model, and `vworld emit correlations` says so beside every coefficient.

**1. Nothing pools across a generation boundary.** Three generations exist —
gen1 (no recovery line, health zero non-terminal), gen2 (a served line saying
health recovers), gen3 (health zero ends the episode). The served prompt *and*
the death semantics differ. A rate pooled across a boundary averages two
different measurements. `vworld read` refuses to.

**2. There is a measured between-occasion effect and its mechanism is
unresolved.** One model shifted **0.319** between two serving days. Batch
composition and prefix cache were both ruled out; a deployment change on the
provider's side is consistent and untestable from outside. **A minority of cells
carry a real serving timestamp**; the rest have only file mtime, which is not a
serving date and also under-detects — three separate sweeps share one mtime day.
`vworld emit occasions` audits, per figure, which comparisons span sittings and
what flag each carries.

**3. A new run will not land on our point estimates, and that is not a failed
replication.** Hosted serving is not batch-invariant, temperature is 0.9, and
the occasion component above is real. `vworld replicate` judges against bands for
that reason. `vworld verify` is the claim that is exact.

## Known defects, shipped rather than hidden

- **166 cells across rounds 15-19 were served by ONE model** — every cell in
  those rounds was `deepcogito/cogito-v2-1-671b`, whatever its filename says.
  The runner built one connection for a whole grid and the request's model field
  was bound once at construction, so the grid's model tuple chose a filename and
  a price, never a served model.

  **The cells are relabelled, not deleted.** Each affected cell now carries
  `served_model` (the truth, from the endpoint's own report), `requested_model`
  (kept as the record of the bug), and a `model_identity_correction` block. They
  are valid data about cogito, and they contain the only identical-input
  repeatability study in the corpus — eight replicates at four levels, same
  model, same seeds, same day.

  **Use `served_model`. Never use the filename as a model identity.** Filenames
  still spell the requested model and were deliberately not renamed: eight of
  round 18's LAT cells would collide on a single name, and the manifest and pins
  address cells by name.

  Consequences: the certified LAT occasion event is withdrawn, the corpus
  contains no surviving occasion EVENT, and eight models leave the benchmark
  entirely because they were never served at these worlds. Full account in
  `docs/research-log.md`, `[CORRECTION] 10 / [TRAP] 32`. Check any corpus with
  `vworld corpus identity`, which exits non-zero if a mislabelled cell is
  present.
- **Sixteen worlds advertise "tallow"** in the Store, a noun never implemented as
  an entity. Models read it and typed commands at it — one in 47 of 65 episodes.
  `world_eden_LAT2` is the corrected twin, identical in every derived value.
  **The defective worlds are kept**, because past results must keep the world
  they actually ran on.
- **Round 10's `MIDDLE` band has a two-band defect**, disclosed and not re-cut.
- **Five published claims were retracted.** `vworld emit corrections` prints the
  ledger and verifies each row against the commit it cites.
- **Twelve generation-3 A0 cells sit below the 0.90 precondition floor**, three
  of them clearly. Reported, never dropped: `vworld read` prints the miss beside
  every affected row.

## Cell schema

Each cell is `{"runs": [...], "meta": {...}}`. An episode carries `seed`,
`steps`, `verb_counts` and `commands`; each command carries `step`, `command`,
`verb`, `room`, `room_after`, `health`, `ok`, `parse_failed`, `ate`,
`ate_forbidden`, `fb_visible`, `fb_held`, `fb_held_after`.

`meta` carries `served_name`, `eden_level`, `eden_arm`, `seed0`,
`terminal_at_zero`, the round's pin, `usage`, `billed_usd`, and — where the
sweep recorded one — `wall_start_epoch`.

### Which field is the model

**`served_model` where it is present; otherwise `served_name`.** The two differ
on the 161 corrected cells, and `requested_model` is kept there to record what
was asked for. A cell's identity status is one of four:

| status | cells | meaning |
|---|---:|---|
| `VERIFIED` | grows | the endpoint reported what it served, and it matched |
| `CORRECTED` | 161 | it did not match; `served_model` is the truth |
| `UNVERIFIED` | 251 | no report on record — the cell predates the field |
| `MISLABELLED` | 0 | uncorrected mismatch; reading one is an error |

Only `VERIFIED` moves: every cell served since the check existed lands there, so
a fixed number would be wrong within a round. The other three are closed sets —
`CORRECTED` and `UNVERIFIED` can only shrink if cells are re-served, and
`MISLABELLED` must stay zero. `vworld corpus identity` prints the live split and
exits non-zero if the last row is not empty.

**`UNVERIFIED` is not a synonym for `VERIFIED`.** Those cells come from rounds
that served one model per invocation, so they are very probably fine — but that
is an argument about how the runner ought to have behaved, not a record of what
it did, and the same kind of argument is what let the defect above survive for
weeks. The corpus reports the distinction rather than flattening it.

Filenames keep a historical `eden_e*` prefix. It is archive vocabulary: renaming
every file would change the digest `verify` checks against, for a string nobody
needs to type. **The model in a filename is the model that was REQUESTED** —
see the first known defect.

## Disclosures

Claude Code ran this harness end to end. A Claude model judges two of nine
raidex dimensions elsewhere in the programme and `gpt-4o-mini` judges a third;
those scores enter only a correlate analysis. A competitor lab's model (GPT-5.6
Terra) is a measured subject — scoring here is deterministic fact-matching with
no model in the loop, so this is a disclosure rather than a confound, stated
explicitly because of who the subject is. No Anthropic model has been measured
as a subject.

## Citation

Cite the repo tag and this manifest digest together. `vworld emit disclosures`
and `vworld emit limitations` print the current text for both.

---

## Maintainer notes — how this card and corpus are published

This file is the source of truth for the dataset card; publishing copies it to
the dataset repo's `README.md`, so it stays versioned with the corpus it
describes. **The YAML block above must remain the first bytes of the file** —
HuggingFace reads front matter only at position 0, and an intro above it means
the card ships with no metadata at all.

The corpus is the `eden_e*.json` cells plus ~11 MB of compiled worlds. It is
**not** `results/`, which holds thousands of files because the Seahaven
programme's artifacts live there too — so staging is explicit, and a bare
`upload . .` would publish unrelated files under this dataset's name.

**The cell count and byte total are not written here.** They were, and the
corpus had grown well past them while the sentence stayed put — the same
hand-maintained-number drift the programme has now paid for five times. The
replacement sentence then quoted both the stale figure and the true one, and
**went stale itself within two rounds**, which is the argument made better than
prose could make it: a number in a document is wrong from the next sweep
onward, including a number written to warn about that. `vetoworld/corpus.manifest.json` is the
one place they live, and `vworld corpus status` is how you read them.

    ST=/tmp/vetoworld-corpus
    mkdir -p $ST/results
    cp results/eden_e*.json           $ST/results/
    cp -r worlds $ST/
    cp vetoworld/corpus.manifest.json $ST/
    cp docs/vetoworld-corpus-card.md  $ST/README.md
    rm -rf $ST/worlds/__pycache__ $ST/worlds/*.py     # code, not data
    (cd $ST && vworld corpus status)                  # digest must MATCH first
    hf upload <owner>/vetoworld-corpus $ST . --repo-type dataset

`hf`, not `huggingface-cli` — the latter is deprecated and now refuses to run.
