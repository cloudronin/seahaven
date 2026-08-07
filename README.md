# Seahaven

Harness for the Seahaven character-divergence experiment. See
`seahaven-spec-v0.1.md` for the research design and the plan file for the build.

**Status:** Phase A (de-risk spikes). Nothing scientific has been run.

## Layout

| Path | Purpose |
|---|---|
| `env/` | Conda environments. Dev (local, arm64) and CUDA (rented H100). |
| `worlds/` | Compiled world artifacts — `.z8` + `.json` sidecar, committed. |
| `seahaven/` | The harness package. |
| `scripts/` | Setup, world builds, sweeps, verification. |
| `tests/` | Hermetic suite; no model, no GPU. `-m slow` for the rest. |

## Setup

```bash
bash scripts/setup_dev_env.sh
```

Then `conda activate seahaven-dev`.

## Two architecture facts that will bite you

**The dev environment must be arm64, and conda will not do that by default.**
The miniconda install on this machine is x86_64, so `conda env create` produces
an x86_64 environment. But `jericho` compiles `libfrotz.so` from C using the
*system* clang, which targets arm64 regardless of the interpreter. The resulting
arm64 library cannot be loaded by an x86_64 Python.

The failure is silent at import time — jericho loads `libfrotz` lazily through
`ctypes`, so `import jericho` succeeds and the mismatch only appears when a world
is first opened:

```
incompatible architecture (have 'arm64', need 'x86_64')
```

`scripts/setup_dev_env.sh` sets `CONDA_SUBDIR=osx-arm64` and then asserts the
invariant. Do not create the environment by hand.

**Compiling a world needs Rosetta 2; playing one does not.** TextWorld's
installer lifts `ni` and `inform6` out of a 2015-era Intel Inform 7 disk image,
so those binaries are x86_64 and run under Rosetta as subprocesses. Compiled
artifacts are committed, which keeps Rosetta off the runtime path entirely.

## Observation hygiene

`seahaven/world/scrub.py` is the only module allowed to turn raw z-machine output
into agent-facing text. It removes four things, all verified against real
TextWorld 1.7.0 output rather than assumed:

1. **The TextWorld ASCII banner** prepended to `reset()`. It spells TEXT WORLD in
   `$` characters, which means a word-based lexicon check cannot see it —
   `"textworld" in obs.lower()` is `False`. Banner suppression cannot be
   delegated to the containment lint.
2. **The z-machine status line**, e.g. `-= Galley =-0/1`. Those trailing digits
   are a score/turn readout, and they appear **even in a no-quest build where
   `max_score == 0`**. Compiling without a quest is necessary but not sufficient
   to keep numbers away from the agent.
3. Inform 7 score chatter, e.g. `[Your score has just gone up by one point.]`.
4. Terminal banners, e.g. `*** You have won ***`.

## Verified on this machine

| Claim | Result |
|---|---|
| `step()` returns `(obs, score, done, infos)` | 4-tuple, old-gym convention — not gymnasium |
| Ground truth is separable from observation | `facts` / `entities` arrive in `infos`; nothing leaks verbatim into `obs` |
| A world compiles with **no quest** | Yes — `max_score == 0`, no quest object required |
| The `.json` sidecar is written beside the `.z8` | Yes — and it is what supplies `facts`/`entities`, which `JerichoEnv` does not populate on its own |

## Testing

```bash
conda run -n seahaven-dev python -m pytest
```
