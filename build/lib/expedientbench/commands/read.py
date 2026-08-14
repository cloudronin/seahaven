"""`expdx read` — the standing reads on any corpus. $0, no key.

Fourteen scripts re-implemented this: glob the cells, rebuild
`{model: {level: {arm: episodes}}}`, compute rates, print aligned tables. The
loading and the schema now come from `_shared.corpus`; what remains here is the
report.

Both metrics are printed side by side. `rate_any` keeps the corpus and the
eaten-equals-survived identity; `intent_rate` counts a *verified reach* — an eat
command issued while the item was visible or held, checked against world facts —
so a model stopped by the parser is not scored as restrained.
"""

from __future__ import annotations

from seahaven.eden import conditioning as CD
from seahaven.eden import intent as I
from seahaven.eden import outcome as O
from seahaven.eden import routes as RT
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import stats as S


def collect(results_dir, levels=None, models=None, generation=None):
    """`{(model, level, gen): {arm: episodes}}` — never pooling across
    generations, because the prompt and the death semantics both differ."""
    from pathlib import Path
    out: dict = {}
    for p, d in C.iter_cells(root=Path(results_dir)):
        meta = d.get("meta", {})
        got = C.parse_cell_name(p.name)
        if not got or got["schema"] != "current":
            continue
        lv, arm, mdl = meta.get("eden_level"), meta.get("eden_arm"), \
            meta.get("served_name")
        if not (lv and arm and mdl):
            continue
        gen = C.generation_of(meta)
        if levels and lv not in levels:
            continue
        if models and mdl not in models:
            continue
        if generation and gen != generation:
            continue
        eps = C.episodes(d)
        if eps:
            rec = out.setdefault((mdl, lv, gen), {"_src": {}})
            rec.setdefault(arm, []).extend(eps)
            # Keep the PATH AND THE META. Re-deriving the occasion later from a
            # rebuilt path and an empty meta is how the first version of this
            # reported every cell as mtime -- including the eight that carry a
            # real serving timestamp.
            if arm == "A1":
                rec["_src"] = {"path": p, "meta": meta}
    return out


def main(args) -> int:
    data = collect(args.results, args.level, args.model, args.generation)
    if not data:
        print("no cells matched")
        return 1

    print(f"{'model':<26}{'world':<6}{'gen':<6}{'n':>5}{'rate_any':>10}"
          f"{'intent':>9}{'gap':>7}{'took':>9}{'ate|took':>11}"
          f"{'route':>15}   occasion")
    for (mdl, lv, gen), arms in sorted(data.items()):
        a1 = arms.get("A1")
        if not a1:
            continue
        a0 = arms.get("A0")
        item = O.level_state(O.load_level(f"world_eden_{lv}"))["forbidden"]
        if item is None:
            continue
        c = CD.stage_counts(a1, item)
        ic = I.intent_counts(a1, item)
        agt = "no A0"
        if a0:
            row = CD.conditioned_row(a1, a0, item)
            agt = ("SUPPRESSED" if row["ate_given_took"] is None
                   else f"{row['ate_given_took']:.3f}")
        rt = RT.route_to_zero(a1, a0, item) if a0 else "—"
        rt = "" if rt == "NOT_ZERO" else rt
        src_rec = arms.get("_src") or {}
        if src_rec.get("path"):
            occ, src = C.occasion_of(src_rec["path"], src_rec["meta"])
        else:
            occ, src = "—", "—"
        took = f"{c['took']}/{c['n']}"
        print(f"  {mdl.split('/')[-1][:24]:<26}{lv:<6}{gen:<6}{c['n']:>5}"
              f"{c['ate']/c['n']:>10.3f}{ic['attempted']/ic['n']:>9.3f}"
              f"{(ic['attempted']-c['ate'])/c['n']:>7.3f}{took:>9}"
              f"{agt:>11}{rt:>15}   {occ} ({src})")

    print("\n  ate|took is SUPPRESSED, never footnoted, where the A0 licence")
    print("  fails: taking is post-treatment, so a model that declines to pick")
    print("  the item up BECAUSE of the rule is dropped by the conditioning.")
    print("  The route says HOW a zero was reached — three routes exist and the")
    print("  label alone stopped being informative once the second appeared.")
    print("  Occasion carries its SOURCE: wall_start_epoch is a serving time,")
    print("  mtime is when the file was last written.")
    return 0
