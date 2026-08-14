"""PHASE 1c — the failure-response axis across the exploration set.

**Exploration, not a result.** The output is a frozen hypothesis set for Phase 2
to test against, never a claim. The held-out twelve are not loaded here and the
loader cannot reach them: cells are addressed by `dim_m<NN>` where NN indexes
`seal.EXPLORATION`, so the firewall holds at analysis time as well as at sweep
time.

---

**The reading, declared before the numbers.** The smoke test found, on two
burned pairs, that models bend after failure and that the junk bin *masks*
between-model signal rather than driving it. Phase 1 asks whether either
generalises to seventeen more models:

1. **Do models bend at all?** Per model, `bend` against that model's own
   100-draw self-split null. A bend inside a model's own null is INDETERMINATE
   for that model, not signal.
2. **Do models differ in how much they bend?** The spread of bends across models
   against the typical within-model null. If between-model spread does not
   exceed the within-model floor, the axis carries no separating structure and
   KP-1/KP-2 apply.
3. **Does junk-masking generalise?** If junk masks, removing the `other` bin
   should *widen* the between-model spread, as it did on both smoke pairs
   (+0.073 and +0.053). If it drives instead, the spread should shrink.

**Three references, per the program note's Stage 4.** The self-split null is the
operative floor here because it is measured in the same TVD units as the bend.
The determinism map's `adherence_sd` is a *different* quantity — adherence
points, not distribution distance — so it is reported as a per-model flag rather
than silently mixed into a TVD threshold. Capability is the pinned MMLU-Pro,
available for 21 of 30 cohort models and never rescued with another proxy.

**Data gap, carried openly.** 501 of 540 cells survived Phase 1a's push
failures. `m15` (Mistral-7B-Instruct, a burned smoke-test model) has **no p2/p3/p4
data at all** and is excluded from phrasing-varying reads; `m06` is thin at 17/30
but covers every phrasing. Both are flagged in the output rather than quietly
averaged.
"""

from __future__ import annotations

import glob
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_state_conditioned import (BINS, N_SELF_SPLITS, TIE_TOL,  # noqa: E402
                                     bend, bucket_items, bucketize,
                                     self_split_null)

from seahaven.dimensional import seal as S  # noqa: E402
from seahaven.fidelity.worldspec import load as load_world  # noqa: E402

SEED = 5150
PHRASINGS = ("p1", "p2", "p3", "p4", "p5")
WORLDS = ("world_v0", "world_v2")


def load_cells(idx: int, world: str | None = None, phrasings=PHRASINGS):
    """Episodes for one exploration model, addressed through the seal.

    `dim_m<NN>` indexes `seal.EXPLORATION`, so a held-out model has no filename
    the loader will ever construct — the firewall is structural, not a check.
    """
    repo = S.EXPLORATION[idx]
    S.assert_not_held_out([repo])
    specs, out = {}, []
    for f in sorted(glob.glob(f"results/dim_m{idx:02d}_*.json")):
        m = re.match(rf"dim_m{idx:02d}_(p\d)_(world_v\d)_(\d+)\.json",
                     Path(f).name)
        if not m or m.group(1) not in phrasings:
            continue
        if world and m.group(2) != world:
            continue
        w = m.group(2)
        if w not in specs:
            specs[w] = load_world(w).entity_kinds()
        for run in json.loads(Path(f).read_text())["runs"]:
            ep = bucketize(run.get("commands", []), specs[w])
            if ep:
                out.append(ep)
    return out


def strip_junk(eps):
    """Drop `other` items, keeping the episode structure the null shuffles over.

    **This is the fix for an instrument defect inherited from the smoke test.**
    The pinned `bend`/`self_split_null` pass `legal_only` down to
    `distribution`, which drops junk *after* `rng.sample(items, n)` — so the
    legal-only read ran at an effective n of n x (1 - junk rate), which is
    model-dependent. Across the exploration set that rate spans 0.2%–94.7%, so
    "n=600" meant n=595 for gemma-2-9b-it and n=32 for Qwen3-8B-Base, and TVD's
    small-n upward bias did the rest: Spearman(effective n, legal-only null) =
    -0.833. The three anomalous nulls were the three junk-heaviest models.

    Filtering here instead means the sample is drawn FROM the legal pool, so the
    effective n is the declared n. The pinned functions are then called with
    `legal_only=False` and are left untouched: `other` is zero in both
    distributions, contributes 0 to the TVD, and the 12-bin read equals the
    11-bin one exactly.
    """
    return [[(bk, it) for (bk, it) in e if it != "other"] for e in eps]


def _n_for(eps) -> tuple[int, int]:
    """Largest n with a null, and the smaller bucket. A self-split needs 2n."""
    m = min(len(bucket_items(eps, b)) for b in ("after_ok", "after_fail"))
    return min(600, m // 2), m


def _read(eps, n, idx) -> dict:
    """One bend against its own null at a declared, honoured n."""
    b = bend(eps, n, random.Random(SEED + idx))
    nul = self_split_null(eps, n, random.Random(SEED + idx))
    p95 = nul.get("p95")
    return {"n": n, "bend": b, "null_p95": p95, "null_median": nul.get("median"),
            "draws": nul.get("draws"),
            "above_own_null": bool(p95 is not None and b > p95 + TIE_TOL)}


def profile(idx: int, world: str | None = None) -> dict | None:
    """One model's bend, its own null, and the junk decomposition.

    Two n's, because they answer different questions. `n_full` is the largest n
    the full read supports and maximises coverage for "do models bend at all".
    `n_common` is the largest n at which the full and legal-only reads can both
    run, and is the only n at which their spreads may be compared — the smoke
    test's own common-n principle, applied to the junk decomposition rather than
    just to bucket size.
    """
    eps = load_cells(idx, world)
    if not eps:
        return None
    legal = strip_junk(eps)
    n_full, m_full = _n_for(eps)
    n_legal, m_legal = _n_for(legal)
    junk = 1.0 - (m_legal / m_full) if m_full else 1.0
    out = {"repo": S.EXPLORATION[idx], "n_full": n_full, "n_legal": n_legal,
           "min_bucket": m_full, "min_bucket_legal": m_legal, "junk_rate": junk,
           "usable": n_full >= 100}
    if not out["usable"]:
        return out
    out["full"] = _read(eps, n_full, idx)
    n_common = min(n_full, n_legal)
    out["n_common"] = n_common
    out["paired"] = n_common >= 100
    if out["paired"]:
        out["full_at_common"] = _read(eps, n_common, idx)
        out["legal_at_common"] = _read(legal, n_common, idx)
    return out


def spread(vals) -> float:
    return st.pstdev(vals) if len(vals) > 1 else 0.0


def main() -> int:
    S.assert_sealed()
    cap = json.loads(Path("results/cohort_capability.json").read_text())
    det = {r["repo"]: r for r in
           json.loads(Path("results/determinism_map.json").read_text())["models"]}

    print("PHASE 1c — failure-response axis, EXPLORATION set only")
    print(f"seal {S.SEAL_HASH[:16]}   held-out {len(S.HELD_OUT)} untouched\n")

    profs = {i: p for i in range(len(S.EXPLORATION))
             if (p := profile(i)) is not None}

    print(f"  {'model':<36}{'junk%':>7}{'n':>6}{'bend':>7}{'null':>7}"
          f"{'|':>3}{'nc':>5}{'full':>7}{'legal':>7}{'null':>7}  flags")
    for i, p in sorted(profs.items()):
        repo, flags = p["repo"], []
        if not p["usable"]:
            print(f"  {repo:<36}{100*p['junk_rate']:>7.1f}{p['min_bucket']:>6}"
                  f"   too thin — excluded")
            continue
        d = det.get(repo)
        if d and not d["deterministic"]:
            flags.append(f"noisy sd={d['adherence_sd']:.2f}")
        if cap.get(repo.lower()) is None:
            flags.append("no proxy")
        if not p["full"]["above_own_null"]:
            flags.append("INDETERMINATE")
        f = p["full"]
        row = (f"  {repo:<36}{100*p['junk_rate']:>7.1f}{f['n']:>6}"
               f"{f['bend']:>7.3f}{f['null_p95']:>7.3f}")
        if p["paired"]:
            fc, lc = p["full_at_common"], p["legal_at_common"]
            row += (f"{'|':>3}{p['n_common']:>5}{fc['bend']:>7.3f}"
                    f"{lc['bend']:>7.3f}{lc['null_p95']:>7.3f}")
        else:
            flags.append(f"no legal read (legal n={p['n_legal']})")
            row += f"{'|':>3}{'--':>5}{'--':>7}{'--':>7}{'--':>7}"
        print(row + "  " + " ".join(flags))

    ok = {i: p for i, p in profs.items() if p["usable"]}
    bends = [p["full"]["bend"] for p in ok.values()]
    nulls = [p["full"]["null_p95"] for p in ok.values()]
    n_above = sum(p["full"]["above_own_null"] for p in ok.values())

    print(f"\n1. DO MODELS BEND?  {n_above}/{len(ok)} exceed their own null")
    print(f"   bend range {min(bends):.3f}–{max(bends):.3f}, "
          f"median null {st.median(nulls):.3f}")

    print(f"\n2. DO MODELS DIFFER?  spread(bend) = {spread(bends):.4f} "
          f"vs median within-model null {st.median(nulls):.4f}")
    sep = spread(bends) > st.median(nulls)
    print(f"   between-model spread {'EXCEEDS' if sep else 'does NOT exceed'} "
          f"the within-model floor")

    # Section 3 runs ONLY at the common n. Comparing a full read at one
    # effective n against a legal-only read at another is what produced the
    # inherited defect; the comparison is only meaningful where both are
    # honoured at the same n.
    pr = {i: p for i, p in ok.items() if p["paired"]}
    bf = [p["full_at_common"]["bend"] for p in pr.values()]
    bl = [p["legal_at_common"]["bend"] for p in pr.values()]
    dl = spread(bl) - spread(bf)
    dropped = sorted(p["repo"] for p in ok.values() if not p["paired"])
    print(f"\n3. DOES JUNK-MASKING GENERALISE?  (common-n read, {len(pr)}/{len(ok)} models)")
    print(f"   spread full {spread(bf):.4f} -> legal-only {spread(bl):.4f}"
          f"  ({dl:+.4f})")
    print(f"   {'WIDENS — consistent with masking' if dl > 0 else 'SHRINKS — junk was contributing, not masking'}")
    grew = sum(1 for p in pr.values()
               if p["legal_at_common"]["bend"] > p["full_at_common"]["bend"])
    print(f"   per model: {grew}/{len(pr)} bend MORE with junk removed")
    if dropped:
        print(f"   NOT SILENT: {len(dropped)} model(s) have too few legal "
              f"commands for any legal-only read at all —")
        for r in dropped:
            print(f"     {r}  (junk {100*ok_by_repo(ok, r)['junk_rate']:.1f}%)")

    print("\n4. PER-WORLD (pooling can hide a world effect as easily as survive one)")
    pw = {}
    for w in WORLDS:
        bs = [q["full"]["bend"] for i in ok
              if (q := profile(i, w)) and q.get("usable")]
        pw[w] = bs
        print(f"   {w}: {len(bs)} models, bend median {st.median(bs):.3f}, "
              f"spread {spread(bs):.4f}")

    out = {"phase": "exploration", "seal": S.SEAL_HASH, "seed": SEED,
           "models": {p["repo"]: p for p in profs.values()},
           "n_above_own_null": n_above, "n_usable": len(ok),
           "spread_bend": spread(bends), "median_null": st.median(nulls),
           "separates": bool(sep),
           "common_n_read": {"n_models": len(pr), "spread_full": spread(bf),
                             "spread_legal": spread(bl), "delta": dl,
                             "dropped_no_legal_read": dropped},
           "per_world_spread": {w: spread(b) for w, b in pw.items()}}
    Path("results/phase1_bend.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/phase1_bend.json  (phase: exploration)")
    return 0


def ok_by_repo(ok, repo):
    return next(p for p in ok.values() if p["repo"] == repo)


if __name__ == "__main__":
    raise SystemExit(main())
