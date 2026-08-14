"""`expdx emit <artifact>` — render a register artifact. $0, no key.

None of these is a hand-maintained document; each is computed from committed
cells so it cannot drift from what the corpus says.
"""

from __future__ import annotations

from seahaven.eden._shared import corpus as C

ARTIFACTS = ("matrix", "occasions", "seeds", "spend",
             "floor-mechanisms", "generations", "limitations",
             "disclosures", "predictions", "corrections", "related-work")


def _occasions() -> int:
    """**The mandatory audit**: which comparisons span serving days.

    Every row prints its SOURCE. Only the timing-probe cells carry a real
    serving timestamp; everything else falls back to file mtime, which for a
    gap-filled cell is its LAST attempt. Rendering the two identically would
    launder an mtime into a serving date inside the one artifact whose whole
    purpose is to be trusted about occasions.
    """
    rows = []
    for p, d in C.iter_cells():
        meta = d.get("meta", {})
        if not meta.get("eden_level"):
            continue
        val, src = C.occasion_of(p, meta)
        rows.append((val, src, meta.get("served_name", "?"),
                     meta.get("eden_level"), meta.get("eden_arm"),
                     C.generation_of(meta)))
    real = sum(1 for r in rows if r[1] == "wall_start_epoch")
    print(f"OCCASION AUDIT — {len(rows)} cells, "
          f"{real} with a REAL serving timestamp, {len(rows)-real} mtime-only\n")
    print(f"  {'when':<18}{'source':<18}{'model':<26}{'world':<6}{'arm':<5}gen")
    for val, src, mdl, lv, arm, gen in sorted(rows):
        print(f"  {val:<18}{src:<18}{mdl.split('/')[-1][:24]:<26}"
              f"{lv:<6}{str(arm):<5}{gen}")
    print("\n  **mtime is NOT a serving date.** It is when the file was last")
    print("  written; for a gap-filled cell that is its last attempt, not its")
    print("  first. Any comparison whose two sides differ in `when` spans")
    print("  occasions, and the programme measured a 0.319 between-day shift on")
    print("  one model with the mechanism unresolved.")
    return 0


def _spend() -> int:
    """From accumulated `usage`, not the running estimates printed at run time."""
    tot = 0.0
    by_round: dict[str, float] = {}
    for p, d in C.iter_cells():
        b = d.get("meta", {}).get("billed_usd")
        if b is None:
            continue
        got = C.parse_cell_name(p.name)
        r = got["round"] if got else "?"
        by_round[r] = by_round.get(r, 0.0) + b
        tot += b
    print("SPEND, from committed cell metadata")
    for r in sorted(by_round, key=lambda x: (len(x), x)):
        print(f"  e{r:<8}${by_round[r]:>9.2f}")
    print(f"  {'TOTAL':<9}${tot:>9.2f}")
    return 0


def _seeds() -> int:
    from .seeds import main as seeds_main

    class _A:
        check = None
        model = None
        level = None
    return seeds_main(_A())


def main(args) -> int:
    from ..register import artifacts as A
    fn = {"occasions": _occasions, "spend": _spend, "seeds": _seeds,
          "matrix": A.matrix, "floor-mechanisms": A.floor_mechanisms,
          "generations": A.generations, "limitations": A.limitations,
          "disclosures": A.disclosures, "predictions": A.predictions,
          "corrections": A.corrections, "related-work": A.related_work}
    if args.artifact not in fn:
        print(f"unknown artifact {args.artifact!r}; have: {', '.join(sorted(fn))}")
        return 1
    return fn[args.artifact]()
