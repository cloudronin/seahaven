"""Smoke test: does the state-conditioned representation carry signal?

**This file is committed before the first bend is computed.** Every constant
below — the bucket rule, the vocabulary, the distance, the subsampling rule, the
null protocol and the NOT-DEAD threshold — is frozen in git before any number
exists, the same pin discipline as `flag.py` and the capability proxy at
`4d32c08`. Without that, "clearly exceeds the null" is a judgment made after
seeing the answer.

**It can only kill.** Two models cannot distinguish signal from a lucky split.
Outcomes are **DEAD / NOT-DEAD, never "works"**, and NOT-DEAD licenses building
the real many-model held-out test and *nothing else* — not the cohort spend, not
the axis enumeration, not the program spec.

---

**Three things the data forced, none of them in the original spec.**

1. The spec's named pair does not exist: `Qwen2.5-7B` was never swept. Pairs
   below are drawn from the 8 bit-exact (>=7B) models that are on disk.

2. **TVD is upward-biased at small n, and bucket sizes vary 8.5x.** Failure
   rates run 5.3% to 45.6%, so `after_fail` buckets run 292 to 2485, and two
   samples from an *identical* distribution give TVD 0.083 at n=292 against
   0.029 at n=2485. A naive between-model comparison would partly measure how
   often a model fails. Hence `COMMON_N`: every bucket in a pair is subsampled
   to the pair's minimum, so the bias is identical and cancels in the difference.

3. The `verb` field is the **raw first token**, not normalised — 378 distinct
   values across the 8 models, 10 for Alibaba against 224 for AI2Mid. Vocabulary
   size tracks junk rate, so an unfrozen vocabulary would make bend measure how
   much garbage a model emits. Hence the frozen `BINS`.
"""

from __future__ import annotations

import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.fidelity.adherence import ACTION_VOCAB, DIRECTIONS  # noqa: E402
from seahaven.fidelity.worldspec import load as load_world  # noqa: E402
from seahaven.fidelity.worldspec import match_forms  # noqa: E402

SEED = 5150

#: **The vocabulary.** Verb + object-class, frozen. `legal|unresolved` catches a
#: legal verb whose object does not resolve to an allowed kind — which
#: deliberately conflates "object not recognised" with "resolved to a kind this
#: verb may not take" (`take crate`). Both are legal-verb, non-canonical-object
#: usage; separating them is the full test's job, not this probe's.
BINS = (
    "go|direction",
    "look", "inventory",
    "examine|o", "examine|c", "examine|s",
    "take|o", "drop|o", "open|c", "close|c",
    "legal|unresolved",
    "other",
)

#: Which (verb, kind) pairs get their own bin. Anything legal outside this falls
#: to `legal|unresolved`.
VERB_KIND_BINS = {("examine", "o"), ("examine", "c"), ("examine", "s"),
                  ("take", "o"), ("drop", "o"), ("open", "c"), ("close", "c")}

N_SELF_SPLITS = 100        #: null is a DISTRIBUTION, not a single draw
N_BOOT = 2000              #: bootstrap draws, resampling episodes
NULL_PERCENTILE = 95       #: NOT-DEAD iff the gap sits ABOVE this percentile

#: Both pairs declared before running; **both reported whatever they say.**
#: Reporting only one after seeing both is the fishing this discipline exists
#: to prevent.
PAIRS = (
    {"name": "P1-max-gap", "a": "TII", "b": "MistralAI",
     "mmlu": (38.10, 23.06),
     "why": "maximum capability gap among bit-exact models, different labs, "
            "OLMo avoided; richest failure buckets in the corpus"},
    {"name": "P2-matched", "a": "Google", "b": "Meta",
     "mmlu": (31.95, 31.09),
     "why": "closest capability match on disk (0.86 MMLU-Pro apart), "
            "different labs; previews whether any signal is non-capability"},
)


# --------------------------------------------------------------------------
# the representation
# --------------------------------------------------------------------------

def object_kind(phrase: str, kinds: dict[str, tuple[str, ...]]) -> str | None:
    """Resolve an object phrase to `o` / `c` / `s`, or None.

    Deterministic by construction: kinds and names are visited in sorted order,
    so a phrase naming two entities always resolves the same way.
    """
    for kind in sorted(kinds):
        for name in sorted(kinds[kind]):
            for form in match_forms(name):
                if form in phrase:
                    return kind
    return None


def bin_of(command: str, kinds: dict[str, tuple[str, ...]]) -> str:
    """One of `BINS`. Reads the command text, never the stored `verb` field.

    The stored field is the raw first token; this normalises exactly as
    `adherence.classify` does, so the vocabulary is the frozen one.
    """
    toks = (command or "").strip().lower().split()
    if not toks:
        return "other"
    verb = "".join(c for c in toks[0] if c.isalpha())
    if verb not in ACTION_VOCAB:
        return "other"
    if verb in ("look", "inventory"):
        return verb
    rest = " ".join(toks[1:])
    if verb == "go":
        return "go|direction" if any(d in rest for d in DIRECTIONS) \
            else "legal|unresolved"
    kind = object_kind(rest, kinds)
    if kind is not None and (verb, kind) in VERB_KIND_BINS:
        return f"{verb}|{kind}"
    return "legal|unresolved"


def distribution(items: list[str], legal_only: bool = False) -> list[float]:
    """Normalised frequency over `BINS`, in `BINS` order.

    `legal_only` drops the `other` bin and renormalises — the decomposition that
    shows whether a bend lives in the junk.
    """
    bins = [b for b in BINS if not (legal_only and b == "other")]
    if legal_only:
        items = [i for i in items if i != "other"]
    n = len(items)
    if n == 0:
        return [0.0] * len(bins)
    counts = {b: 0 for b in bins}
    for i in items:
        if i in counts:
            counts[i] += 1
    return [counts[b] / n for b in bins]


def tvd(p: list[float], q: list[float]) -> float:
    """Total variation. Bounded [0, 1], no distributional assumption."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


# --------------------------------------------------------------------------
# episodes, buckets, subsampling
# --------------------------------------------------------------------------

def bucketize(cmds: list[dict], kinds: dict[str, tuple[str, ...]]):
    """One episode -> [(bucket, bin), ...].

    **The whole conditioning structure.** The episode's FIRST step has no
    predecessor and is dropped; every other step is keyed by the PREVIOUS step's
    `ok`. One split, no choices, no fishing room.
    """
    return [("after_ok" if cmds[i - 1]["ok"] else "after_fail",
             bin_of(cmds[i]["command"], kinds))
            for i in range(1, len(cmds))]


def load_episodes(lab: str, pattern: str = "results/vp_*.json"):
    """Per-episode [(bucket, bin), ...] for one model, pooled over all 30 cells."""
    specs = {}
    out = []
    for f in sorted(glob.glob(pattern)):
        parts = Path(f).stem.split("_")
        # `vp_<lab>_<phrasing>_world_<v>_<seed>` and nothing else; analysis
        # output lands in the same directory and matches this glob.
        if len(parts) < 6 or not parts[-1].isdigit() or parts[1] != lab:
            continue
        world = f"{parts[3]}_{parts[4]}"
        if world not in specs:
            specs[world] = load_world(world).entity_kinds()
        kinds = specs[world]
        for run in json.loads(Path(f).read_text())["runs"]:
            cmds = run.get("commands", [])
            ep = bucketize(cmds, kinds)
            if ep:
                out.append(ep)
    return out


def bucket_items(episodes, bucket: str) -> list[str]:
    return [b for ep in episodes for (bk, b) in ep if bk == bucket]


def bend(episodes, n: int, rng: random.Random, legal_only: bool = False) -> float:
    """TVD between the two buckets, each subsampled to exactly `n`.

    Subsampling is what makes bends comparable across models: without it the
    smaller bucket carries more TVD bias and a model that fails rarely looks
    like it bends more.
    """
    ds = []
    for bucket in ("after_ok", "after_fail"):
        items = bucket_items(episodes, bucket)
        if len(items) < n:
            return float("nan")
        ds.append(distribution(rng.sample(items, n), legal_only))
    return tvd(ds[0], ds[1])


def self_split_null(episodes, n: int, rng: random.Random,
                    legal_only: bool = False) -> dict:
    """The load-bearing control, as a DISTRIBUTION over `N_SELF_SPLITS` draws.

    How much "bend" appears from sampling alone: split the model's EPISODES into
    random halves and compare a bucket against itself across the halves. A
    single draw would leave the read an eyeball comparison; the distribution is
    what makes it a test.

    Run at the same `n` and the same binning as the bend, so it self-calibrates
    to whatever resolution the vocabulary implies. **Only buckets with at least
    `2n` items can supply two disjoint same-size draws** — in practice
    `after_ok` always can and `after_fail` usually cannot, so the null is
    estimated wherever the headroom exists and the decision uses the largest
    percentile found, which is the conservative direction for a kill-only test.
    """
    draws, used = [], []
    for bucket in ("after_ok", "after_fail"):
        if len(bucket_items(episodes, bucket)) < 2 * n:
            continue
        used.append(bucket)
        for _ in range(N_SELF_SPLITS):
            shuffled = episodes[:]
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            a = bucket_items(shuffled[:half], bucket)
            b = bucket_items(shuffled[half:], bucket)
            if len(a) < n or len(b) < n:
                continue
            draws.append(tvd(distribution(rng.sample(a, n), legal_only),
                             distribution(rng.sample(b, n), legal_only)))
    draws.sort()
    if not draws:
        return {"draws": 0, "buckets": used}
    return {"draws": len(draws), "buckets": used,
            "median": draws[len(draws) // 2],
            "p95": draws[int(0.95 * (len(draws) - 1))],
            "max": draws[-1]}


def percentile_of(value: float, draws: list[float]) -> float:
    return 100.0 * sum(1 for d in draws if d < value) / len(draws) if draws else float("nan")


# --------------------------------------------------------------------------
# the read
# --------------------------------------------------------------------------

def boot_gap(eps_a, eps_b, n: int, rng: random.Random, legal_only: bool) -> list:
    """Bootstrap |bend(A) − bend(B)| by resampling EPISODES."""
    out = []
    for _ in range(N_BOOT):
        ra = [eps_a[rng.randrange(len(eps_a))] for _ in eps_a]
        rb = [eps_b[rng.randrange(len(eps_b))] for _ in eps_b]
        ba, bb = bend(ra, n, rng, legal_only), bend(rb, n, rng, legal_only)
        if ba == ba and bb == bb:
            out.append(abs(ba - bb))
    out.sort()
    return out


def run_pair(pair: dict) -> dict:
    rng = random.Random(SEED)
    a, b = pair["a"], pair["b"]
    eps = {a: load_episodes(a), b: load_episodes(b)}

    sizes = {m: {bk: len(bucket_items(eps[m], bk))
                 for bk in ("after_ok", "after_fail")} for m in (a, b)}
    n = min(v for m in sizes.values() for v in m.values())

    print(f"\n{'=' * 72}\n{pair['name']}: {a} vs {b}   "
          f"MMLU-Pro {pair['mmlu'][0]} / {pair['mmlu'][1]} "
          f"(gap {abs(pair['mmlu'][0] - pair['mmlu'][1]):.2f})")
    print(f"  {pair['why']}")
    for m in (a, b):
        print(f"  {m:<11} {len(eps[m]):>4} episodes   "
              f"after_ok {sizes[m]['after_ok']:>5}   "
              f"after_fail {sizes[m]['after_fail']:>5}")
    print(f"  common n = {n} (every bucket subsampled to this, so TVD's "
          f"size-bias cancels)")

    res = {"pair": pair["name"], "a": a, "b": b, "mmlu": pair["mmlu"],
           "common_n": n, "sizes": sizes, "variants": {}}

    for legal_only in (False, True):
        tag = "legal-only" if legal_only else "full"
        bends = {m: bend(eps[m], n, rng, legal_only) for m in (a, b)}
        nulls = {m: self_split_null(eps[m], n, rng, legal_only) for m in (a, b)}
        gap = abs(bends[a] - bends[b])

        pooled = []
        for m in (a, b):
            r = random.Random(SEED + hash(m) % 1000)
            d = self_split_null(eps[m], n, r, legal_only)
            pooled.append(d)
        p95 = max((d.get("p95", 0.0) for d in pooled), default=0.0)

        gaps = boot_gap(eps[a], eps[b], n, random.Random(SEED + 7), legal_only)
        lo, hi = (gaps[int(0.05 * len(gaps))], gaps[int(0.95 * len(gaps))]) \
            if gaps else (float("nan"), float("nan"))
        verdict = "NOT-DEAD" if gap > p95 else "DEAD"

        print(f"\n  --- {tag} bend ---")
        for m in (a, b):
            nl = nulls[m]
            own = nl.get("p95", float("nan"))
            print(f"    bend({m:<10}) = {bends[m]:.4f}    "
                  f"own null p95 {own:.4f} ({nl.get('draws', 0)} draws, "
                  f"buckets {nl.get('buckets', [])})    "
                  f"{'above own null' if bends[m] > own else 'INSIDE own null'}")
        print(f"    |gap| = {gap:.4f}   90% CI [{lo:.4f}, {hi:.4f}]")
        print(f"    null p95 (pair)  = {p95:.4f}")
        print(f"    => {tag}: {verdict}")

        res["variants"][tag] = {
            "bend": bends, "gap": gap, "ci90": [lo, hi], "null_p95": p95,
            "null_per_model": nulls, "verdict": verdict,
            "bend_above_own_null": {m: bool(bends[m] > nulls[m].get("p95", 1e9))
                                    for m in (a, b)}}
    fu, lo_ = res["variants"]["full"], res["variants"]["legal-only"]
    res["junk_driven"] = bool(fu["verdict"] == "NOT-DEAD"
                              and lo_["verdict"] == "DEAD")
    res["gap_shrink_on_legal_only"] = fu["gap"] - lo_["gap"]
    return res


def main() -> int:
    print("SMOKE TEST — does the state-conditioned representation carry signal?")
    print("Frozen before running: bucket rule, verb+object-class vocabulary, TVD,")
    print(f"common-n subsampling, {N_SELF_SPLITS}-split null, NOT-DEAD at the "
          f"{NULL_PERCENTILE}th percentile, seed {SEED}.")
    print("DEAD / NOT-DEAD only. NOT-DEAD licenses building the real test, "
          "nothing else.")

    results = [run_pair(p) for p in PAIRS]

    print(f"\n{'=' * 72}\nREADING\n")
    p1, p2 = results[0], results[1]
    for r in results:
        print(f"  {r['pair']:<12} full {r['variants']['full']['verdict']:<9} "
              f"legal-only {r['variants']['legal-only']['verdict']}")

    print("\n  Two independent routes to 'is this capability?' — they should agree:")
    print(f"    route 1, matched pair : {p2['variants']['full']['verdict']}")
    print(f"    route 2, junk decomp  : "
          f"{'junk-driven' if p1['junk_driven'] else 'survives legal-only'} "
          f"(gap shrinks {p1['gap_shrink_on_legal_only']:+.4f} on legal-only)")

    agree = not (p1["junk_driven"] ^ (p2["variants"]["full"]["verdict"] == "DEAD"))
    if p1["variants"]["full"]["verdict"] == "DEAD":
        note = ("P1 is DEAD at maximum capability gap — the representation is "
                "flat where it had the best chance. The cross-check does not "
                "arise.")
    elif agree:
        note = ("the two capability routes AGREE. "
                + ("Both say the bend is capability-driven."
                   if p1["junk_driven"] else
                   "Neither flags capability; the signal survives both."))
    else:
        note = ("**INFORMATIVE INCOHERENCE** — the two capability routes "
                "DISAGREE. Stop and look; do not average it away. Logged as a "
                "result in its own right.")
    print(f"\n  {note}")

    out = {"phase": "exploration", "seed": SEED, "bins": list(BINS),
           "n_self_splits": N_SELF_SPLITS, "n_boot": N_BOOT,
           "null_percentile": NULL_PERCENTILE,
           "pairs": results, "routes_agree": bool(agree), "note": note,
           "licenses": "NOT-DEAD licenses building the real many-model held-out "
                       "test and nothing else."}
    Path("results/smoke_state_conditioned.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print("\nwrote results/smoke_state_conditioned.json  (phase: exploration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
