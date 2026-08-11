"""Compute the axis-2 capability floor from the pinned proxy, and emit it.

**This exists so the floor is a committed artifact rather than a number retyped
from a chat message.** The two-tier KP-4 amendment turns on one threshold, and a
threshold that lives only in prose can drift silently between the reasoning that
justified it and the code that enforces it.

---

**The rule, and why it is not the amendment's example.** The amendment names the
floor as *"the top of the open-weight cohort's range, the level above which the
cohort has no models"*, and illustrates it with *"e.g. ~40"*. Those disagree: the
cohort maximum is `Qwen2.5-14B-Instruct` at 43.38, so a floor of 40 would place a
cohort member **above its own cohort's floor** — contradicting the rule the
example is illustrating. **The rule governs.**

**The regime separation is arithmetic, not argument.** The model that sets the
floor is HELD-OUT, so the exploration set axis 2 actually measures tops out at
38.10 (`Falcon3-10B-Instruct`) — 5.28 points below the floor. No cohort model is
above it. That is what licenses reading an open-weight capability reduction as
*wrong regime* rather than *instrument dead*.

**The frontier margin exists because "above the floor" is not "saturated".** A
model at 45 clears the floor by 1.6 points; the discoverability confound the
amendment predicts should vanish above the floor cannot be seen to vanish while
sitting on top of it, and the result would be another "maybe not capable enough"
— the infinite-deferral trap the amendment closes. So `FRONTIER_FLOOR = 60.0`,
derived: the cohort's whole covered span is 35.63 points, and 60.0 sits 16.62
above the ceiling, ~47% of that span. The band between them is admitted to
**neither** tier.

Neither threshold moves after either run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.dimensional import seal as S  # noqa: E402

#: The rule: the level above which the open-weight cohort has NO models.
#:
#: **Rounded UP from the exact ceiling (43.382461583924346), never down.** At a
#: floor of 43.38 the ceiling model itself tests as above its own cohort's floor
#: -- the same error as the amendment's "~40" example, three orders of magnitude
#: smaller and correspondingly easier to miss. The assertion in `main()` is what
#: caught it; it stays there for whoever edits this next.
FLOOR = 43.4

#: "Above the floor" must mean saturated, not marginal. See the module docstring.
FRONTIER_FLOOR = 60.0


def facts() -> dict:
    """Recomputed from the sealed cohort, never asserted by hand."""
    covered = {m: v[3] for m, v in S.COHORT.items() if v[3] is not None}
    expl = {m: c for m, c in covered.items() if m in S.EXPLORATION}
    ceiling_model = max(covered, key=lambda m: covered[m])
    expl_model = max(expl, key=lambda m: expl[m])
    lo, hi = min(covered.values()), max(covered.values())
    return {
        "n_covered": len(covered),
        "n_cohort": len(S.COHORT),
        "range": [lo, hi],
        "span": hi - lo,
        "ceiling": hi,
        "ceiling_model": ceiling_model,
        "ceiling_is_held_out": ceiling_model in S.HELD_OUT,
        "exploration_ceiling": max(expl.values()),
        "exploration_ceiling_model": expl_model,
        "headroom_below_floor": FLOOR - max(expl.values()),
        "floor": FLOOR,
        "frontier_floor": FRONTIER_FLOOR,
        "frontier_margin": FRONTIER_FLOOR - FLOOR,
        "frontier_margin_as_span_fraction": (FRONTIER_FLOOR - FLOOR) / (hi - lo),
        "exclusion_band": [FLOOR, FRONTIER_FLOOR],
        "n_cohort_above_floor": sum(1 for c in covered.values() if c > FLOOR),
        "proxy": dict(sorted(covered.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    S.assert_sealed()
    f = facts()

    # The rule is only satisfied if NO cohort model sits above the floor. If this
    # ever fails, the cohort gained a model above 43.38 and the floor -- not the
    # cohort -- is what has to be re-derived, in a fresh pre-registration.
    assert f["n_cohort_above_floor"] == 0, (
        f"{f['n_cohort_above_floor']} cohort model(s) above the floor; the floor "
        "no longer means 'the level above which the cohort has no models'")
    # The floor must be the ceiling rounded UP -- not an arbitrary larger number,
    # which would quietly widen the "below the floor" regime to flatter the
    # open-weight result.
    assert FLOOR >= f["ceiling"], (FLOOR, f["ceiling"])
    assert FLOOR - f["ceiling"] < 0.1, (
        f"floor {FLOOR} is {FLOOR - f['ceiling']:.3f} above the ceiling; it "
        "must be the ceiling rounded up, not a chosen higher number")

    print("AXIS-2 CAPABILITY FLOOR — computed from the pinned proxy")
    print(f"seal {S.SEAL_HASH[:16]}\n")
    print(f"  covered              {f['n_covered']}/{f['n_cohort']} "
          f"(9 uncovered by seal design, never rescued)")
    print(f"  cohort range         {f['range'][0]:.2f} - {f['range'][1]:.2f}"
          f"   span {f['span']:.2f}")
    print(f"  ceiling              {f['ceiling']:.2f}  {f['ceiling_model']}"
          f"   {'HELD-OUT' if f['ceiling_is_held_out'] else 'exploration'}")
    print(f"  exploration ceiling  {f['exploration_ceiling']:.2f}  "
          f"{f['exploration_ceiling_model']}")
    print(f"\n  FLOOR                {FLOOR:.2f}")
    print(f"  FRONTIER_FLOOR       {FRONTIER_FLOOR:.2f}   "
          f"(+{f['frontier_margin']:.2f}, "
          f"{100 * f['frontier_margin_as_span_fraction']:.0f}% of the cohort span)")
    print(f"  exclusion band       ({FLOOR:.2f}, {FRONTIER_FLOOR:.2f}) "
          f"-- admitted to NEITHER tier")
    print(f"\n  cohort models above the floor: {f['n_cohort_above_floor']}")
    print(f"  open-weight run sits {f['headroom_below_floor']:.2f} points BELOW "
          f"the floor by construction")

    out = {"phase": "pre-registration", "seal": S.SEAL_HASH, **f}
    Path("results/axis2_floor.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/axis2_floor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
