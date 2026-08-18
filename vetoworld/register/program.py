"""§2 and §6 of the bundle: the cohort's shape, and the programme's totals.

Two sections share one module because they share one question — *how much of
this is there, and how sure are we* — and splitting them would mean two copies
of the same cohort walk.

**§2** is the register's shape: the GLM family contrast, and the per-model
breakdown behind the spread and ceiling claims. **§6** is what the abstract
quotes: cells, pins, rounds, and the power the correlation section actually has.

The power window is the sharp one. `claims.vetohold.cohort` is `(17, 11)` —
seventeen models scored, eleven joinable — so E1's n is ELEVEN, and every
closed-arms bound computed at 17 is describing a cohort that does not exist.
"""

from __future__ import annotations

__all__ = ["glm_rows", "glm_family", "ceiling_rows", "spread_and_ceiling",
           "power_rows", "power_window", "program_rows", "program_counts"]


def _vh():
    from . import correlations as CO
    return CO.veto_hold(), CO.providers()


# --- §2 the GLM family -------------------------------------------------------

def glm_rows() -> list[dict]:
    """Every GLM version on record, with its score, provider and occasion.

    **The family is selected by name prefix, not listed.** A hand-typed list of
    four would silently omit a fifth the day one is measured, and the whole
    argument this table carries is that versions of ONE family diverge — which
    is only checkable if the family is complete.
    """
    from seahaven.eden._shared import occasion as OQ

    vh, prov = _vh()
    days: dict = {}
    for o in OQ.observations():
        days.setdefault(o.model, set()).add(o.day)

    out = []
    for model in sorted(vh):
        if not model.lower().startswith("zai-org/glm"):
            continue
        seen = sorted(days.get(model, ()))
        out.append({
            "model": model,
            "C1": vh[model],
            "provider": prov.get(model, "—"),
            "occasions": ", ".join(seen) if seen else "—",
            "n_occasions": len(seen),
        })
    return out


def glm_family() -> int:
    from .markdown import table

    rows = glm_rows()
    print("THE GLM FAMILY — one family, several post-training choices\n")
    print(table(["model", "C1 (veto-hold)", "provider", "occasions"],
                [[r["model"], r["C1"], r["provider"], r["occasions"]]
                 for r in rows]))
    print(f"\n  {len(rows)} versions on record, selected by name prefix rather")
    print("  than listed — a typed list of four would omit a fifth the day it")
    print("  is measured, and completeness is what the argument rests on.")
    print("\n  **What this table is for.** These are versions of ONE model")
    print("  family. Where they diverge, the difference is a post-training")
    print("  choice rather than a difference of scale or architecture, which is")
    print("  the cleanest evidence in the register that the behaviour measured")
    print("  here is CHOSEN and not emergent.")
    #: **Computed, not cautioned.** A general warning that provider matters is
    #: useless beside a table that actually crosses providers; the reader needs
    #: to know THIS one does, and which rows.
    by_prov: dict = {}
    for r in rows:
        by_prov.setdefault(r["provider"], []).append(r["model"])
    print("\n  Provider is a HARD PARTITION (round 21, Rule 1): a rate compared "
          "across")
    print("  serving stacks is documentation, never a finding.")
    if len(by_prov) > 1:
        print(f"\n  ** THIS TABLE SPANS {len(by_prov)} SERVING STACKS. **")
        for prov, models in sorted(by_prov.items()):
            print(f"       {prov}: {', '.join(m.split('/')[-1] for m in models)}")
        print("     Within-stack contrasts are readable as post-training")
        print("     choices. ACROSS the split they are not — the difference")
        print("     there is confounded with the serving stack, and the")
        print("     manuscript must say so beside the number rather than")
        print("     letting the family read as one comparison.")
    else:
        print("\n  All versions served on one stack, so the family contrast is")
        print("  within-provider and reads cleanly.")
    return 0


# --- §2 spread and ceiling ---------------------------------------------------

def ceiling_rows() -> list[dict]:
    """Per-model C1 with a Wilson upper bound on the A1 rate, and provider.

    `claims.vetohold.spread` and `.ceiling` publish the totals; this is the
    breakdown behind them, which is what a reader needs to see that the ceiling
    is four models at the SAME n rather than one model measured harder.
    """
    from seahaven.eden import intent as I
    from seahaven.eden import outcome as O
    from seahaven.eden._shared.stats import wilson_ci
    from . import correlations as CO

    vh, prov = _vh()
    #: **The same canonical cells C1 is computed from**, via `CO._cells()`.
    #: `a1_rates()` returns RATES, not counts, so a Wilson bound cannot be built
    #: from it — and pooling some other selection would put a bound on a
    #: different quantity than the score it sits beside.
    cells = CO._cells()
    item = {w: O.level_state(O.load_level(f"world_eden_{w}"))["forbidden"]
            for w in CO.WORLDS}
    counts: dict = {}
    for (mdl, w), (_rank, eps, _sweep, _prov) in cells.items():
        c = I.intent_counts(eps, item[w])
        acc = counts.setdefault(mdl, [0, 0])
        acc[0] += c["attempted"]
        acc[1] += c["n"]

    out = []
    for model in sorted(vh, key=lambda m: (-vh[m], m)):
        k, n = counts.get(model, (0, 0))
        lo, hi = wilson_ci(k, n) if n else (None, None)
        out.append({
            "model": model, "C1": vh[model], "attempted": k, "n": n,
            "wilson_lo": lo, "wilson_hi": hi,
            "provider": prov.get(model, "—"),
            "at_ceiling": vh[model] == 100.0,
        })
    return out


def spread_and_ceiling() -> int:
    from .markdown import table

    rows = ceiling_rows()
    ceiling = [r for r in rows if r["at_ceiling"]]
    print("SPREAD AND CEILING — the breakdown behind the published totals\n")
    print(table(
        ["model", "C1", "A1 intent (pooled)", "Wilson upper", "provider",
         "ceiling"],
        [[r["model"], r["C1"], f"{r['attempted']} / {r['n']}", r["wilson_hi"],
          r["provider"], "yes" if r["at_ceiling"] else ""] for r in rows]))

    print("\n  **C1 and the pooled column are not the same statistic.** C1 is")
    print("  `100 x (1 - MEAN over worlds of intent_A1)`; the counts here are")
    print("  POOLED across worlds, so the Wilson bound describes the pooled")
    print("  intent rate rather than C1 itself. It is printed as the precision")
    print("  of the underlying measurement, not as an interval on the score —")
    print("  and the distinction is stated because reading it as the latter")
    print("  would understate the between-world component C1's mean carries.")

    print(f"\n  {len(ceiling)} models at a saturating 100.0, "
          f"{len(rows)} scored in total.")
    ns = {r["n"] for r in ceiling}
    print(f"  The ceiling models share n = {sorted(ns)}, so the last is "
          "measured as")
    print("  precisely as the first — a tie of equals, not of unequal effort.")
    print("\n  **A saturating scale depresses every correlation it enters.** The")
    print("  attenuation disclosure printed beside every coefficient matters")
    print("  more as the tie grows, and it has grown: see `claims.vetohold.ceiling`.")
    provs = {r["provider"] for r in ceiling}
    print(f"\n  The ceiling spans {len(provs)} providers ({', '.join(sorted(provs))}).")
    print("  A ceiling reproduced ACROSS serving stacks is stronger evidence of")
    print("  a real ceiling than one inside a single stack — the one thing a")
    print("  mixed-provider cohort buys, and the reason its cost is worth it.")
    return 0


# --- §6 power ----------------------------------------------------------------

def power_rows() -> list[dict]:
    """The closed-arms window at the LIVE n, beside the stale one.

    **The n=17 bounds describe a cohort that does not exist.** Seventeen models
    are scored; eleven are joinable onto the second axis, so every correlation
    E1 reports has n=11. A window computed at 17 is narrower than the evidence
    supports, and quoting it would overstate the section's power.
    """
    from . import correlations as CO

    scored = len(CO.veto_hold())
    joined_models, _score, _dims = CO.joined()
    live = len(joined_models)
    out = []
    for n, label in ((live, "LIVE — the joined cohort"),
                     (scored, "scored cohort (NOT the correlation n)")):
        lo, hi = CO._window(n)
        out.append({"n": n, "label": label, "lo": lo, "hi": hi,
                    "excludes_zero": bool(lo is not None and hi is not None
                                          and (lo > 0 or hi < 0))})
    return out


def power_window() -> int:
    from .markdown import table

    rows = power_rows()
    print("E1 POWER — the closed-arms window at the n that actually applies\n")
    print(table(["n", "which cohort", "lower", "upper", "excludes zero"],
                [[r["n"], r["label"], r["lo"], r["hi"],
                  "yes" if r["excludes_zero"] else "**no**"] for r in rows]))

    live = rows[0]
    print(f"\n  **The correlation n is {live['n']}, not {rows[1]['n']}.**")
    print("  Seventeen models are scored; only the joinable subset carries the")
    print("  second axis, and a coefficient is computed on the join.")
    if not live["excludes_zero"]:
        print("\n  **AT THE LIVE n THE WINDOW INCLUDES ZERO.** This section is")
        print("  underpowered BY CONSTRUCTION: even a perfect result could not")
        print("  exclude no-correlation at this cohort size. That is a")
        print("  limitation to state in the paper, not a result to report — and")
        print("  it is why the window is emitted beside every coefficient.")
    print("\n  Any window quoted from an earlier session at the larger n is")
    print("  describing a cohort that does not exist. See MANIFEST supersessions.")
    return 0


# --- §6 programme totals -----------------------------------------------------

def program_rows() -> list[tuple[str, object]]:
    """Cells, corpus digest, providers, pins, rounds — the abstract's numbers.

    Collected from the functions that already compute them, rather than scraped
    from three terminal outputs by hand.
    """
    from pathlib import Path

    from seahaven.eden._shared.corpus import RESULTS
    from vetoworld.commands.corpus import _digest_corpus
    from vetoworld.commands.pin import ROUNDS
    from . import correlations as CO
    from .claims import CLAIMS

    digest, cells, nbytes = _digest_corpus(RESULTS)
    probe_cells = len(list(Path(RESULTS).glob("probe-*.json")))
    provs = sorted(set(CO.providers().values()))
    suite = len(list(Path("tests").glob("test_*.py")))
    return [
        ("corpus cells (eden_e*)", cells),
        ("corpus digest", digest),
        ("corpus bytes", nbytes),
        ("probe cells (excluded from the corpus BY CONSTRUCTION)", probe_cells),
        ("models scored", len(CO.veto_hold())),
        ("models joined (the correlation n)", len(CO.joined()[0])),
        ("providers in the register", ", ".join(provs)),
        ("registered figures", len(CLAIMS)),
        ("pinned rounds", len(ROUNDS)),
        ("test files", suite),
    ]


def program_counts() -> int:
    from .markdown import kv_table

    print("PROGRAMME-LEVEL COUNTS — what the abstract and intro quote\n")
    print(kv_table(program_rows(), key="quantity", value="value"))
    print("\n  Probe cells are counted SEPARATELY and are not part of the")
    print("  corpus digest. Their filename schema cannot match `eden_e*`, so")
    print("  they are invisible to the corpus reader by construction rather")
    print("  than by discipline — which is the only version of that guarantee")
    print("  worth having.")
    print("\n  `vworld pin check` verifies each pinned round recomputes;")
    print("  `vworld verify` recomputes every registered figure. This artifact")
    print("  counts them rather than re-running them.")
    return 0
