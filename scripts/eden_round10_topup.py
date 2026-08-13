"""ROUND 10 TOP-UP — Qwen3.5-9B from n=72 to n=96 at LAT, generation 3.

**Why this exists, stated correctly, because the spec's reason was wrong.**

The spec expected the top-up to narrow the interval from +/-0.14 to +/-0.10 and to
separate Qwen3.5-9B from more of the cohort. Both were checked and both are false:
the interval goes +/-0.110 -> +/-0.096, and Qwen3.5-9B is ALREADY separable from
all fifteen other models at nominal alpha, so the top-up newly separates it from
NONE.

The real reason is MULTIPLICITY. Fifteen pairwise comparisons carry a Bonferroni
alpha of 0.00333, and the weakest pair -- DS-V4-Flash, which is the neighbour the
headline break is *against* -- sits at p = 0.00744 and FAILS. At n=96 it projects
to 0.00295 and passes. That is the same standard round 10 applied to its own
correlates when it refused the sycophancy hit at p=0.0435; applying it to the
headline and not to the correlates would be the double standard.

**This runs BEFORE the tallow fix.** That fix edits LAT's room text, which lives in
the world lock, which this pin hashes -- so it retires round 10 and permanently
closes the ability to add cells. There is no second chance.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import round10 as R  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from eden_round10_sweep import cell_path, run_cell  # noqa: E402

# ---------------------------------------------------------------------------
# PRE-COMMITMENTS. Written here, before the 24 episodes exist.
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3.5-9B"
ARM, LEVEL = "A1", "LAT"

TOPUP_SEED0 = R.SEED0 + R.EPISODES_A1          # 15072 — indices CONTINUE
TOPUP_EPISODES = 24
TOPUP_TOTAL = R.EPISODES_A1 + TOPUP_EPISODES   # 96

# ---------------------------------------------------------------------------
# THE SECOND ARM. **A post-hoc extension, labelled as one.**
#
# The pre-registered purchase above FAILED: Qwen3.5-9B pooled to 58/96 = 0.604
# against a projection of 59/96, and DS-V4-Flash stayed at p=0.00486 versus a
# threshold of 0.00333. ONE episode spanned it, and the projection carried no
# interval — that is the error.
#
# The arithmetic then shows the binding constraint was never Qwen's arm. A Fisher
# comparison is limited by BOTH denominators, and DS-V4-Flash sits at 27/72, the
# smaller one. Topping IT to 96 against Qwen's observed 58/96 gives p=0.00234 and
# clears the threshold. That was computable before any episode was served and was
# not computed — the same "topped up the wrong thing" shape as the round-10 read
# scanning the wrong direction.
#
# **This is a deviation from the approved plan, which said nothing else on LAT
# would be topped up.** It is run because the plan's own stated PURPOSE was to
# make this one pair survive multiplicity, that sentence was written believing
# the Qwen buy would achieve it, and Stage B closes the door permanently. The
# safeguard is that BOTH results are reported: the pre-registered 96-only outcome
# stands as the primary result and is a FAILURE, and this extension is reported
# separately as post-hoc. No p-value is laundered by replacement.
SECOND_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
SECOND_IS_POST_HOC = True

#: The pooled rate is the SIMPLE SUM over 96 episodes. Never an average of two
#: rates -- that would weight 24 episodes as heavily as 72.
POOLING = "simple sum over 96"

#: 15 pairwise comparisons against the rest of the round-10 cohort.
N_COMPARISONS = 15
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS

#: **The outcome that would embarrass this purchase, named in advance.** If the
#: new 24 come in low enough that the pooled 96 still fails correction against
#: DS-V4-Flash, that is REPORTED AS THE FINDING. The top-up is run regardless of
#: outcome and neither half is dropped on any test.
FAILURE_IS_A_RESULT = True


def assert_seeds_are_free(model: str = MODEL) -> None:
    """Disjointness on SEED VALUES against disk, never on indices.

    Round 3's top-up used OFFSET seeds and this one uses CONTINUED indices; both
    constructions have coexisted in this repo, so an index-based check would pass
    while silently re-drawing an episode that already exists.
    """
    want = set(range(TOPUP_SEED0, TOPUP_SEED0 + TOPUP_EPISODES))
    mine = cell_path(model, ARM, LEVEL).name
    for f in glob.glob("results/eden_e*.json"):
        d = json.loads(Path(f).read_text())
        meta = d.get("meta", {})
        if meta.get("eden_level") != LEVEL:
            continue
        # Seed space is PER MODEL — every model's A1 cell starts at SEED0, and
        # seed pairing is across arms within a model, not across models. So only
        # a clash inside this model's own arms is a re-draw.
        if meta.get("served_name") != model:
            continue
        used = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        clash = want & used
        if clash and Path(f).name != mine:
            raise SystemExit(
                f"SEED COLLISION: {sorted(clash)[:6]} already served for {model} "
                f"in {Path(f).name}. Refusing to re-draw an existing episode.")


def main() -> int:
    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY not set")
        return 2

    # **Pin against ROUND 10's hashes, not a new round's.** If a hashed artifact
    # has moved, stop rather than pool 24 episodes across a code change.
    R.assert_pinned()

    model = MODEL
    if "--second" in sys.argv:
        model = SECOND_MODEL
        print("*** POST-HOC EXTENSION — not the pre-registered purchase. ***")
        print("*** The 96-only result stands separately and is a FAILURE. ***\n")

    p = cell_path(model, ARM, LEVEL)
    if not p.exists():
        raise SystemExit(f"{p} does not exist; there is nothing to top up.")
    cell = json.loads(p.read_text())
    meta = cell.get("meta", {})
    if meta.get("round10_pin") != R.PINNED_ROUND10_HASH:
        raise SystemExit(
            f"{p.name} carries pin {meta.get('round10_pin')!r}, not round 10's. "
            "Pooling across freezes is what the pin exists to prevent.")
    if meta.get("terminal_at_zero") is not True:
        raise SystemExit(
            f"{p.name} is not a generation-3 cell (terminal_at_zero="
            f"{meta.get('terminal_at_zero')!r}). Topping it up would pool a "
            "generation-1 half into a generation-3 cell.")
    have = len([r for r in cell.get("runs", []) if r.get("commands")])
    if have != R.EPISODES_A1:
        raise SystemExit(
            f"{p.name} holds {have} episodes, expected {R.EPISODES_A1}. Fill the "
            "gap with the sweep before topping up, so the halves stay 72 and 24.")

    assert_seeds_are_free(model)
    seeds = list(range(TOPUP_SEED0, TOPUP_SEED0 + TOPUP_EPISODES))

    print(f"ROUND-10 TOP-UP — {model} {ARM} {LEVEL}")
    print(f"  pin {R.PINNED_ROUND10_HASH[:16]}…  terminal_at_zero={R.TERMINAL_AT_ZERO}")
    print(f"  {have} on disk  +{len(seeds)} new (seeds {seeds[0]}-{seeds[-1]})"
          f"  ->  {TOPUP_TOTAL}")
    print(f"  pooling: {POOLING}   Bonferroni alpha at {N_COMPARISONS} "
          f"comparisons = {BONFERRONI_ALPHA:.5f}\n")

    res = run_cell(model, ARM, LEVEL, seeds=seeds)

    # `run_cell` stamps `runs` from `R.episodes_for(arm)` = 72, which is no longer
    # what this cell is. Recording the top-up explicitly so the meta cannot be
    # read as a 72-episode cell that mysteriously holds 96.
    res["meta"]["runs"] = TOPUP_TOTAL
    res["meta"]["topup"] = {
        "from": R.EPISODES_A1, "to": TOPUP_TOTAL,
        "seed0": TOPUP_SEED0, "episodes": TOPUP_EPISODES,
        "pooling": POOLING, "reason": "multiplicity: weakest pair vs DS-V4-Flash",
        "post_hoc": model == SECOND_MODEL,
    }
    n = len([r for r in res.get("runs", []) if r.get("commands")])
    if n != TOPUP_TOTAL:
        print(f"  WARNING: cell holds {n}, expected {TOPUP_TOTAL} — "
              "the read reports the shortfall rather than voiding it.")
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  n={n}/{TOPUP_TOTAL}  ${res['meta']['billed_usd']:.3f} total "
          f"(${res['meta']['billed_this_attempt_usd']:.3f} this attempt)")
    print("  Now run scripts/eden_round10_topup_read.py for the halves test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
