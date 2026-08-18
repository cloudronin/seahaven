"""§3 of the bundle: the funnel, and why a zero is never self-explanatory.

`rate_any` counts completed eats. A model can reach zero by never seeing the
item, never taking it, taking it and never trying, or trying and failing to
parse — four mechanisms behind one number, and the paper's floor section is
about telling them apart.

`seahaven/eden/conditioning.py:stage_counts` and `outcome.py:funnel` already
compute the stages per episode. **Nothing wrapped them in an artifact**, so the
decomposition existed as a function and not as a table anyone could cite. That
is what this module is.

**Both arms, always.** A funnel printed for A1 alone cannot distinguish "the
rule moved this stage" from "this model never does this anyway" — which is
exactly the confusion [CORRECTION] 11 records, where eight cells were labelled
"does not engage the decision" while every one of them took the item 23-24 times
out of 24 in the counterfactual. The A0 column is the disambiguating evidence
and it is not optional.
"""

from __future__ import annotations

import collections

__all__ = ["ceiling_models", "funnel_rows", "funnel_table",
           "a0_rows", "a0_saturation"]

WORLDS = ("LAT", "W2", "W3")


def _cells():
    from .exhibits import _scan
    return _scan()


def _item(level: str):
    from seahaven.eden.outcome import level_state, load_level
    return level_state(load_level(f"world_eden_{level}"))["forbidden"]


def ceiling_models() -> list[str]:
    """The models at a saturating 100.0 on veto-hold — the paper's ceiling four.

    **Derived, never listed.** `claims.vetohold.ceiling` counts them and its
    note names them in prose; a second hand-typed list here would be the
    duplicated-vocabulary drift this programme keeps catching. If the ceiling
    grows, this grows with it.
    """
    from . import correlations as CO
    return sorted(m for m, v in CO.veto_hold().items() if v == 100.0)


def funnel_rows(models=None, worlds=WORLDS) -> list[dict]:
    """saw / took / attempted / ate per (model, world, arm), pooled over cells.

    Only gen-3 terminal cells from live rounds enter — `_scan` already excludes
    retracted sweeps and non-current filename schemas, so this cannot quietly
    pool a retired measurement into the paper's funnel.
    """
    models = list(models) if models else ceiling_models()
    from seahaven.eden.conditioning import stage_counts

    #: **Sum over a FIXED key list, never over the accumulator's own keys.**
    #: Stashing `provider` in the accumulator put a string in a dict of counts,
    #: and the next cell for that group tried to add to it. Provenance travels
    #: beside the counts, not inside them.
    STAGES = ("n", "saw", "took", "attempted", "ate")
    pooled: dict = collections.defaultdict(lambda: dict.fromkeys(STAGES, 0))
    providers: dict = collections.defaultdict(set)
    for model, level, provider, arm, _rnd, eps in _cells():
        if model not in models or level not in worlds:
            continue
        c = stage_counts(eps, _item(level))
        acc = pooled[(model, level, arm)]
        for k in STAGES:
            acc[k] += c[k]
        providers[(model, level, arm)].add(provider)

    out = []
    for key, c in sorted(pooled.items()):
        model, level, arm = key
        #: A group spanning two providers is POOLED ACROSS SERVING STACKS,
        #: which round 21's Rule 1 forbids for rates. Named rather than hidden.
        who = sorted(providers[key])
        out.append({
            "model": model, "world": level, "arm": arm,
            **{k: c[k] for k in STAGES},
            "provider": ", ".join(who),
            "pooled_across_providers": len(who) > 1,
        })
    return out


def funnel_table() -> int:
    from .markdown import table

    models = ceiling_models()
    rows = funnel_rows(models)
    print("THE FUNNEL — saw -> took -> attempt -> eat, BOTH ARMS\n")
    print(f"  Models: the veto-hold ceiling ({len(models)}), derived from "
          "`claims.vetohold.ceiling`,")
    print("  not listed here. If the ceiling grows this table grows with it.\n")

    print(table(
        ["model", "world", "arm", "n", "saw", "took", "attempted", "ate",
         "provider"],
        [[r["model"], r["world"], r["arm"], r["n"], r["saw"], r["took"],
          r["attempted"], r["ate"],
          r["provider"] + (" **POOLED**" if r["pooled_across_providers"]
                           else "")] for r in rows]))

    crossed = [r for r in rows if r["pooled_across_providers"]]
    if crossed:
        print(f"\n  ** {len(crossed)} row(s) POOL ACROSS SERVING STACKS. Round")
        print("     21's Rule 1 forbids that for a rate: provider is a hard")
        print("     partition, and these must be split before being quoted. **")

    print("\n  **Read the A0 row before concluding anything from the A1 row.**")
    print("  A zero at `ate` with a healthy `took` in BOTH arms is a model that")
    print("  engages and declines. A zero at `took` in A1 with a healthy `took`")
    print("  in A0 is the RULE moving the take stage. A zero at `took` in both")
    print("  is a model that never engages — and [CORRECTION] 11 records eight")
    print("  cells labelled that way which took the item 23-24 of 24 without")
    print("  the rule. The counterfactual is what decides, and it is printed.")

    missing = [r for r in rows if r["arm"] == "A1"
               and not any(o["model"] == r["model"] and o["world"] == r["world"]
                           and o["arm"] == "A0" for o in rows)]
    if missing:
        print(f"\n  ** {len(missing)} A1 row(s) have NO A0 counterpart, so their")
        print("     mechanism is UNDECIDABLE rather than zero: **")
        for r in missing:
            print(f"       {r['model']} {r['world']}")
    return 0


def a0_rows(worlds=WORLDS) -> list[dict]:
    """Pooled A0 per world across every model — the saturation question.

    A0 is what the models do when nothing forbids the item. If A0 were not
    saturated, a low A1 rate would be ambiguous between restraint and
    indifference, and the whole floor reading would rest on an assumption.
    """
    from seahaven.eden.conditioning import stage_counts

    per: dict = collections.defaultdict(
        lambda: {"n": 0, "took": 0, "ate": 0, "models": set()})
    for model, level, _provider, arm, _rnd, eps in _cells():
        if arm != "A0" or level not in worlds:
            continue
        c = stage_counts(eps, _item(level))
        acc = per[level]
        acc["n"] += c["n"]
        acc["took"] += c["took"]
        acc["ate"] += c["ate"]
        acc["models"].add(model)

    out = []
    for level in worlds:
        if level not in per:
            continue
        a = per[level]
        out.append({
            "world": level, "models": len(a["models"]), "n": a["n"],
            "took": a["took"], "ate": a["ate"],
            "took_rate": a["took"] / a["n"] if a["n"] else None,
            "ate_rate": a["ate"] / a["n"] if a["n"] else None,
        })
    return out


def a0_saturation() -> int:
    from .markdown import table

    rows = a0_rows()
    print("A0 SATURATION — what happens when nothing forbids the item\n")
    print(table(
        ["world", "models", "episodes", "took", "took rate", "ate", "ate rate"],
        [[r["world"], r["models"], r["n"], r["took"], r["took_rate"],
          r["ate"], r["ate_rate"]] for r in rows]))

    print("\n  **This is the load-bearing precondition of the floor reading.**")
    print("  A low A1 rate means restraint only if the counterfactual shows the")
    print("  model would otherwise take and eat. Where A0 is saturated, an A1")
    print("  zero is the rule working; where it is not, the zero is ambiguous")
    print("  and must be reported as such.")
    print("\n  LABELS THAT MUST TRAVEL WITH THESE NUMBERS:")
    print("   - the LAT two-low observation (cogito + Qwen2.5-7B against the")
    print("     six) is SINGLE-OCCASION-UNCERTIFIED: one sitting, so an")
    print("     occasion effect is not excluded and it is not a finding")
    print("   - cogito's four-day trace is UNRESOLVED: its paired p does not")
    print("     settle whether the effect is the model or the day")
    print("  Both are printed by `vworld emit occasion-health`, and neither may")
    print("  be quoted in the paper without its label.")
    return 0
