"""B2 — is the vLLM serving path deterministic, with and without the flag?

Written before the data exists, per the addendum's script-first rule.

**What hangs on it.** The P1 cell-reuse check in Phase C is currently specified
as byte-identical transcripts. TRAP 32 showed that acceptance test is
uninterpretable on a nondeterministic path: at n=1 the wrapper looked like code
drift, at n=2 the control agreed by drawing the same sample twice, and only at
n=4 did both paths turn out nondeterministic and modally identical. If this
stack has ambient noise, "P1 differs" cannot distinguish drift from noise, and
the reuse check needs the same argument-fidelity restructuring.

**Why four whole evals per config.** The mechanism is batch composition, and the
sweep runs 12 episodes concurrently. Repeating a single rollout would never vary
the batch and would report a determinism the sweep does not have.

Research log §8.2 measured 1/16 distinct outputs with `VLLM_BATCH_INVARIANT=1`
against 4/16 without, and called the flag mandatory — but it was set in **none**
of `gpu_job15`–`18`, so the entire existing corpus was generated without it.
That finding was on the LoRA path, whose mechanism is the shrink kernel's
relaxed float atomics, and these sweeps run no LoRA. Whether it transfers has
never been checked. This checks it.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_reps(config: str) -> dict[int, list[tuple[str, ...]]]:
    """Per repeat, the command sequence of every run index."""
    out: dict[int, list[tuple[str, ...]]] = {}
    for f in sorted(glob.glob(f"results/b2_{config}_rep*.json")):
        rep = int(Path(f).stem.split("rep")[1])
        d = json.loads(Path(f).read_text())
        seqs = []
        for run in sorted(d["runs"], key=lambda r: r["run"]):
            seqs.append(tuple(c["command"] for c in run.get("commands", [])))
        out[rep] = seqs
    return out


def report(config: str) -> dict:
    reps = load_reps(config)
    if len(reps) < 2:
        print(f"  {config}: only {len(reps)} repeat(s) — cannot assess")
        return {"config": config, "repeats": len(reps), "assessable": False}

    n_runs = min(len(v) for v in reps.values())
    per_run_distinct = []
    for i in range(n_runs):
        variants = Counter(reps[r][i] for r in sorted(reps))
        per_run_distinct.append(len(variants))

    identical_runs = sum(1 for d in per_run_distinct if d == 1)
    # Command-level divergence, so a single differing token is not scored the
    # same as a wholly different episode.
    total_cmds = diff_cmds = 0
    base = sorted(reps)[0]
    for i in range(n_runs):
        ref = reps[base][i]
        for r in sorted(reps)[1:]:
            other = reps[r][i]
            total_cmds += max(len(ref), len(other))
            diff_cmds += sum(1 for a, b in zip(ref, other) if a != b)
            diff_cmds += abs(len(ref) - len(other))
    rate = diff_cmds / total_cmds if total_cmds else 0.0

    print(f"  {config:<10} repeats={len(reps)}  runs={n_runs}")
    print(f"    runs identical across all repeats: {identical_runs}/{n_runs}")
    print(f"    distinct sequences per run       : {Counter(per_run_distinct)}")
    print(f"    command-level divergence         : {100*rate:.3f}%")
    return {"config": config, "repeats": len(reps), "runs": n_runs,
            "identical_runs": identical_runs,
            "distinct_hist": dict(Counter(per_run_distinct)),
            "divergence_rate": rate,
            "deterministic": identical_runs == n_runs, "assessable": True}


def main() -> int:
    print("B2 — determinism of the vLLM serving path\n")
    results = [report(c) for c in ("default", "invariant")]
    ok = [r for r in results if r.get("assessable")]
    if not ok:
        print("\nno usable output", file=sys.stderr)
        return 2

    d = {r["config"]: r for r in ok}
    print("\nCONSEQUENCES\n")
    default_det = d.get("default", {}).get("deterministic")
    inv_det = d.get("invariant", {}).get("deterministic")

    if default_det:
        verdict = ("The path is deterministic as already run. Byte-identity is a "
                   "valid acceptance test, the P1 reuse check stands as written, "
                   "and the existing corpus needs no noise-floor caveat.")
        flag = "not required"
    elif inv_det:
        verdict = ("The path is NOT deterministic by default but VLLM_BATCH_INVARIANT=1 "
                   "fixes it. The flag binds from here forward, so V-P and Stage 1 "
                   "share a serving configuration. Byte-identity is valid ONLY under "
                   "the flag. Everything measured before this point was generated "
                   "without it and carries the noise floor below.")
        flag = "REQUIRED from here forward"
    else:
        verdict = ("Neither configuration is deterministic. Byte-identity is "
                   "abandoned as an acceptance test: the P1 reuse check is "
                   "restructured to argument fidelity plus config hash, exactly as "
                   "TRAP 32 disposed of the wrapper check, and the divergence rate "
                   "below becomes the stated noise floor that H1's coefficients are "
                   "interpreted against.")
        flag = "does not fix it"

    print(f"  VLLM_BATCH_INVARIANT=1: {flag}")
    print(f"  {verdict}")
    if not default_det:
        print(f"\n  noise floor (default flags): "
              f"{100*d['default']['divergence_rate']:.3f}% of commands")

    Path("results/b2_determinism.json").write_text(json.dumps(
        {"configs": ok, "flag": flag, "verdict": verdict}, indent=2) + "\n")
    print("\nwrote results/b2_determinism.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
