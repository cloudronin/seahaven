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
    vworld verify                # all 17 figures recompute, $0, no key

`verify` recomputes every quoted figure and exits nonzero naming any that
drifted. It needs **no API key and costs nothing**.

`corpus fetch` pulls the cells over plain HTTPS — **no `huggingface-cli` and no
`huggingface_hub`**, because this package declares no runtime dependencies and
the verb a replicator runs first is the wrong place to break that. It stages the
download, checks the digest against the manifest, and installs nothing that does
not match: a corpus that arrived corrupted would otherwise make `verify` report
drifted figures, which is a statement about the network dressed as a statement
about the manuscript.

## What is in here

| | |
|---|---|
| `results/` | **259 cells**, one JSON per (round, model, arm, world). Each holds every episode's full command trace with per-step health, room, parse status and funnel flags. |
| `worlds/` | The compiled `.z8` worlds, their `.json` sidecars, and `BUILD.lock.json` per world — topology, larder, params, derived block, and the `.z8` sha256. |
| pins | Eleven frozen payload digests, two live and eleven retired, each recomputable from literals inside its round module. |
| claims register | One named function per manuscript figure. The paper cites the function; `verify` runs it. |
| replication bands | Per-cell Wilson intervals widened by the measured occasion component. Computed once and shipped as data, never improvised at run time. |
| prompt fixtures | The assembled A1/A0 bytes per world per generation, so prompt assembly can be checked byte-for-byte. |

**Manifest digest: `8fb0cb6e18cca6eb`** (259 cells). Cite it beside the repo tag.

## Three things to know before using it

**1. Nothing pools across a generation boundary.** Three generations exist —
gen1 (no recovery line, health zero non-terminal), gen2 (a served line saying
health recovers), gen3 (health zero ends the episode). The served prompt *and*
the death semantics differ. A rate pooled across a boundary averages two
different measurements. `vworld read` refuses to.

**2. There is a measured between-occasion effect and its mechanism is
unresolved.** One model shifted **0.319** between two serving days. Batch
composition and prefix cache were both ruled out; a deployment change on the
provider's side is consistent and untestable from outside. **Ten of 259 cells
carry a real serving timestamp**; the rest have only file mtime, which is not a
serving date and also under-detects — three separate sweeps share one mtime day.
`vworld emit occasions` audits, per figure, which comparisons span sittings and
what flag each carries.

**3. A new run will not land on our point estimates, and that is not a failed
replication.** Hosted serving is not batch-invariant, temperature is 0.9, and
the occasion component above is real. `vworld replicate` judges against bands for
that reason. `vworld verify` is the claim that is exact.

## Known defects, shipped rather than hidden

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

Filenames keep a historical `eden_e*` prefix. It is archive vocabulary: renaming
259 files would change the digest `verify` checks against, for a string nobody
needs to type.

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

The corpus is 107 MB: the 259 `eden_e*.json` cells (100,517,734 bytes, the
number the manifest digests) plus 11 MB of compiled worlds. It is **not**
`results/`, which holds 2,644 files and 247 MB because the Seahaven programme's
artifacts live there too — so staging is explicit, and a bare `upload . .` would
publish 2,385 unrelated files under this dataset's name.

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
