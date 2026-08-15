"""`vworld emit <artifact>` — render a register artifact. $0, no key.

None of these is a hand-maintained document; each is computed from committed
cells so it cannot drift from what the corpus says.
"""

from __future__ import annotations

from seahaven.eden._shared import corpus as C

ARTIFACTS = ("matrix", "occasions", "occasion-health", "seeds", "spend",
             "floor-mechanisms", "generations", "limitations",
             "disclosures", "predictions", "corrections", "related-work",
             "correlations", "convergence")


def _occasions() -> int:
    """**The mandatory audit**: which FIGURES compare cells across serving days.

    Per figure, not per cell. A per-cell inventory says when each file was
    written and leaves the reader to work out which comparisons that endangers;
    the manuscript needs the answer attached to the figure, because the flag has
    to travel with the number into the paper.

    Every row prints its SOURCE. A minority of cells carry a real serving timestamp;
    everything else falls back to file mtime, which for a gap-filled cell is its
    LAST attempt. Rendering the two identically would launder an mtime into a
    serving date inside the one artifact whose whole purpose is to be trusted
    about occasions.
    """
    from ..register import occasions as OC

    rows = OC.audit()
    need = [r for r in rows if r.needs_flag]
    print(f"OCCASION AUDIT — {len(rows)} registered figures, "
          f"{len(need)} carrying an occasion flag\n")
    print(f"  {'figure':<22}{'cells':>6}  {'same occasion?':<15}sweeps")
    for r in rows:
        print(f"  {r.fid:<22}{len(r.cells):>6}  {r.verdict:<15}"
              f"{'e' + ', e'.join(r.sweeps)}")
        if r.flag:
            for line in _wrap(r.flag, 68):
                print(f"        {line}")
    print()

    consumed = sorted({p for r in rows for p in r.cells})
    real = sum(1 for p in consumed
               if C.occasion_of(p, C.load_cell(p).get("meta", {}))[1]
               == "wall_start_epoch")
    print(f"  PROVENANCE OF THE {len(consumed)} CELLS THESE FIGURES READ — "
          f"{real} real, {len(consumed)-real} mtime-only\n")
    print(f"  {'when':<18}{'source':<18}{'cell'}")
    for p in consumed:
        val, src = C.occasion_of(p, C.load_cell(p).get("meta", {}))
        print(f"  {val:<18}{src:<18}{_cell_id(p)}")

    print("\n  **mtime is NOT a serving date.** It is when the file was last")
    print("  written; for a gap-filled cell that is its last attempt, not its")
    print("  first. It also UNDER-DETECTS: rounds 11, 12 and 13 were three")
    print("  sweeps and share one mtime day, so a timestamps-only audit would")
    print("  call those comparisons clean. The sweep column is the honest")
    print("  signal — it is recorded in the cell's own name, not inferred.")
    print("\n  The programme measured a 0.319 between-day shift on one model")
    print("  with the mechanism unresolved. Batch composition and prefix cache")
    print("  were ruled out; a deployment change is consistent and untestable")
    print("  from here. That is why a flag, not a correction.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


def _cell_id(path) -> str:
    """A cell's PUBLIC identity: sweep, model, arm, world.

    Not its filename. The committed files keep a historical prefix — renaming
    357 of them would change the corpus digest that `verify` checks against, for
    a string no reader needs — so the identity is rendered from the parts
    instead. It is also what a reader actually wants: `e12 cogito A1 LAT` says
    what the cell is, which the path only implies.
    """
    got = C.parse_cell_name(path.name)
    if not got:
        return path.stem
    who = (got["model"] or "?").split("/")[-1]
    return (f"e{got['round']:<6}{who[:26]:<28}"
            f"{str(got['arm'] or '-'):<4}{got['level']}")


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


def registry() -> dict:
    """The dispatch table, **exposed so the vocabulary cannot fork.**

    `ARTIFACTS` is what `--help` and the tests enumerate; this dict is what
    actually dispatches. They were two literals maintained by hand, and an
    artifact added to one escaped the other — `correlations` and `convergence`
    dispatched for weeks without ever being rendered by a test. That is the same
    duplicated-vocabulary drift as doctor's `ROUNDS` and `test_occasions`'
    `_TIMESTAMPED`, which is three instances, so this one is asserted rather
    than merely fixed: `test_ARTIFACTS_and_the_DISPATCH_TABLE_cannot_fork`.
    """
    from ..register import artifacts as A
    from ..register import correlations as CO
    from ..register import occasion_health as OH
    return {"occasions": _occasions, "spend": _spend, "seeds": _seeds,
            "occasion-health": OH.occasion_health,
            "matrix": A.matrix, "floor-mechanisms": A.floor_mechanisms,
            "generations": A.generations, "limitations": A.limitations,
            "disclosures": A.disclosures, "predictions": A.predictions,
            "corrections": A.corrections, "related-work": A.related_work,
            "correlations": CO.correlations,
            "convergence": CO.candidate_convergence}


def main(args) -> int:
    fn = registry()
    if args.artifact not in fn:
        print(f"unknown artifact {args.artifact!r}; have: {', '.join(sorted(fn))}")
        return 1
    return fn[args.artifact]()
