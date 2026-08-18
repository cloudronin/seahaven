"""Exhibit 1 — MECHANISM FLIP, computed from committed cells.

**Named three times and built zero**, which is why it is here: an exhibit that
exists only in a plan cannot be checked, and cannot report that it is empty.

The claim class: *the same model reaches its outcome by a different route on a
different provider.* That is a statement about MECHANISM, and it is admissible
across providers precisely because it is not a statement about LEVEL.

**THE PROVENANCE RULE, which does most of the work here.**

    `probe.LEVELS_RULE` — cross-provider LEVEL differences are documentation,
    never findings. Only within-provider deltas and cross-provider coincidence
    of EVENTS are read.

So a pair where one provider ate and the other did not is a RATE difference
wearing a mechanism label, and it is NOT a mechanism flip no matter how
striking it looks. The admissible flip is between two cells that reached the
SAME rate — zero — by different routes: `BINDS_AT_TAKE` against
`BINDS_AT_EAT`, or either against `REACHED_AND_FAILED`. There the level is
identical by construction and only the mechanism differs, so the comparison
survives its own rule.

This module therefore classifies rather than filters, and prints all three
buckets. Hiding the inadmissible rows would leave a reader unable to see that
the interesting-looking differences were the ones that do not count.

**WHAT IT FINDS IS NOTHING, AND THE REASON IS NOT THE ONE EXPECTED.**

The plan called this exhibit "$0 and available now", blocking on nothing. That
is false on the corpus, and this module is how it was found out. Round 21's
cells were served on DeepInfra; their Together-era counterparts for EVERY
overlapping model come from rounds 15 and 16, and both are in
`identity.RETRACTED_SWEEPS`. Twenty-eight model/level pairs are measured on two
providers, and ZERO of them have an admissible cell on both sides.

So Exhibit 1 is not merely empty, it is BLOCKED — and **money does not unblock
it**, which was this module's own first wrong guess. Re-measuring the seven on
Together is not available at any price: six are non-serverless there as of
2026-08-16 (a tiering decision, not a capability one) and Kimi-K3 is servable
but returns EMPTY CONTENT under this programme's pinned request form. Seven of
seven walled, by two different mechanisms.

That is a finding about the plan, and it is recorded rather than worked around,
because an exhibit padded with retracted cells would be worth less than no
exhibit at all.

A second bar sits behind the first even once cells exist: of the 28 pairs, none
is zero-rate on both sides, so all of them would be LEVEL differences anyway.
Both counts are printed, because they are separate obstacles and fixing the
first does not fix the second.

**THE ZERO IS SUBSTRATE PENDING, NOT ABANDONED**, and `SUBSTRATE_PENDING` says
so on the emitted table. The matched-pair fleet is the honest roadmap: Flash
carries the A1 decision cell on BOTH columns daily, so route labels for one
model accrue from probe cells over the coming weeks without buying anything.
That comparison will be PROBE-SOURCED and descriptive, and is labelled as such
rather than being quietly pooled with round measurements.

`test_EXHIBIT_1_IS_BLOCKED_AND_SAYS_WHY` asserts the emptiness and its cause,
so the day a genuine flip lands somebody learns it from a failing test rather
than from a paragraph — the device `FLASH_ENVELOPE_INERT_AT_PIN` uses.
"""

from __future__ import annotations

import collections
import functools

from seahaven.eden import routes as RT
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import identity as ID
from seahaven.eden.outcome import level_state, load_level

__all__ = ["mechanism_rows", "exhibit_1", "ADMISSIBILITY", "SUBSTRATE_PENDING"]

ADMISSIBILITY = (
    "A cross-provider MECHANISM claim is admissible only where the RATE is the "
    "same on both sides. Zero-on-both with different routes is a mechanism "
    "flip: same level by construction, different binding stage. Anything "
    "involving NOT_ZERO is a cross-provider LEVEL difference, which "
    "probe.LEVELS_RULE makes documentation and never a finding. "
    "NEVER_TOOK_UNVERIFIED is neither: it is the absence of the counterfactual "
    "arm that would decide, and it is reported as undecidable rather than "
    "quietly counted as agreement.")


#: **Why the zero is a waiting room and not a grave.**
#: Printed on the table so a reader meeting `ADMISSIBLE MECHANISM FLIPS: 0` is
#: told what would change it, and is not left to infer that the exhibit was
#: tried and given up on.
SUBSTRATE_PENDING = (
    "SUBSTRATE PENDING, NOT ABANDONED. The blocked pairs cannot be recovered "
    "by spending: six of the seven are non-serverless on Together as of "
    "2026-08-16, and Kimi-K3 returns empty content there under this "
    "programme's pinned request form — seven of seven walled, by two different "
    "mechanisms, so no budget reopens them. "
    "The route the design actually leaves open is the MATCHED-PAIR FLEET: "
    "DeepSeek-V4-Flash-0731 carries the A1 decision cell on BOTH columns every "
    "day, so route labels for one model accrue from PROBE cells over the coming "
    "weeks at no additional cost. Any flip that emerges that way is "
    "PROBE-SOURCED and DESCRIPTIVE, and must be labelled so on the table: probe "
    "cells are excluded from the score corpus by construction, and a comparison "
    "drawn from them is not a round measurement.")


@functools.lru_cache(maxsize=1)
def _scan():
    """One pass over the corpus, shared by every reader below.

    `exhibit_1` asked the corpus for the same thing three times — once through
    `mechanism_rows` and twice through `blocked_by_retraction`. The artifact
    leak test renders every registered artifact on every full-suite run, so that
    was three passes over ~2900 cells added to CI in exchange for nothing.

    Cached for the process. The corpus is committed files; nothing in a single
    run writes to it, and the CLI is one-shot.
    """
    out = []
    for path, data in C.iter_cells():
        meta = data.get("meta", {})
        level, arm = meta.get("eden_level"), meta.get("eden_arm")
        if not level or arm not in ("A0", "A1"):
            continue
        got = C.parse_cell_name(path.name)
        if not got or got.get("schema") != "current":
            continue
        #: **Who SERVED, not who was asked for.** #113, and the reason
        #: `served_provider` is now written into every probe cell.
        out.append((
            ID.bare_model(meta.get("served_name") or meta.get("model") or ""),
            level, ID.provider_of(meta), arm, got.get("round"),
            C.episodes(data)))
    return out


def _by_model_level_provider(include_retracted=False):
    """`{(model, level, provider): {arm: episodes}}` from committed cells.

    Model names are BARE — `identity.bare_model` — because a provider is a fact
    about the row and not part of the name. Printed raw, the mechanism table
    once carried `GLM-5:deepinfra` in a column of model names.

    `include_retracted` exists ONLY so the report can count what retraction
    removed. Nothing computes a finding from it.
    """
    out: dict = collections.defaultdict(dict)
    for model, level, provider, arm, rnd, eps in _scan():
        if not include_retracted and rnd in ID.RETRACTED_SWEEPS:
            continue
        out[(model, level, provider)][arm] = eps
    return out


def blocked_by_retraction():
    """Pairs that would be cross-provider comparisons but for a retracted side.

    Returns `[{"model", "level", "live", "retracted"}]`. This is the reason
    Exhibit 1 is empty, and it is computed rather than asserted so that the day
    a live Together cell lands for one of these models the count moves on its
    own.
    """
    live: dict = collections.defaultdict(set)
    for model, level, provider in _by_model_level_provider():
        live[(model, level)].add(provider)

    allrounds: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for model, level, provider, _arm, rnd, _eps in _scan():
        allrounds[(model, level)][provider].add(rnd)

    out = []
    for key, provs in sorted(allrounds.items()):
        if len(provs) < 2 or len(live.get(key, ())) > 1:
            continue
        out.append({
            "model": key[0], "level": key[1],
            "live": sorted(live.get(key, ())),
            "retracted": {p: sorted(r for r in rs if r in ID.RETRACTED_SWEEPS)
                          for p, rs in sorted(provs.items())
                          if any(r in ID.RETRACTED_SWEEPS for r in rs)},
        })
    return out


def mechanism_rows():
    """One row per (model, level) measured on more than one provider.

    `{"model", "level", "labels": {provider: route}, "verdict"}` where verdict
    is FLIP, LEVEL-DIFFERENCE, UNDECIDABLE or AGREES.
    """
    items: dict = {}
    routes: dict = collections.defaultdict(dict)
    for (model, level, provider), arms in _by_model_level_provider().items():
        if "A1" not in arms:
            continue
        if level not in items:
            items[level] = level_state(load_level(f"world_eden_{level}"))["forbidden"]
        #: `a0` is passed even when absent — `route_to_zero` returns
        #: NEVER_TOOK_UNVERIFIED rather than guessing, which is [CORRECTION] 11.
        routes[(model, level)][provider] = RT.route_to_zero(
            arms["A1"], arms.get("A0"), items[level])

    rows = []
    for (model, level), labels in sorted(routes.items()):
        if len(labels) < 2:
            continue
        values = set(labels.values())
        if "NEVER_TOOK_UNVERIFIED" in values:
            verdict = "UNDECIDABLE"
        elif len(values) == 1:
            verdict = "AGREES"
        elif "NOT_ZERO" in values:
            #: The striking-looking ones, and the ones that do not count.
            verdict = "LEVEL-DIFFERENCE"
        else:
            verdict = "FLIP"
        rows.append({"model": model, "level": level,
                     "labels": dict(sorted(labels.items())), "verdict": verdict})
    return rows


def exhibit_1() -> int:
    rows = mechanism_rows()
    flips = [r for r in rows if r["verdict"] == "FLIP"]
    levels = [r for r in rows if r["verdict"] == "LEVEL-DIFFERENCE"]
    undec = [r for r in rows if r["verdict"] == "UNDECIDABLE"]
    agrees = [r for r in rows if r["verdict"] == "AGREES"]

    print("EXHIBIT 1 — MECHANISM FLIP ACROSS PROVIDERS")
    print("=" * 72)
    print("\nThe claim: the same model reaches the SAME rate by a DIFFERENT")
    print("route on a different provider. Admissible across columns precisely")
    print("because it is not a claim about level.\n")
    print("ADMISSIBILITY")
    for line in _wrap(ADMISSIBILITY):
        print(f"  {line}")

    blocked = blocked_by_retraction()

    print(f"\n\nADMISSIBLE MECHANISM FLIPS: {len(flips)}")
    print("-" * 72)
    if not flips:
        print("  NONE ON RECORD, AND THE EXHIBIT IS BLOCKED RATHER THAN QUIET.")
        print()
        print(f"  {len(blocked)} model/level pairs ARE measured on two providers,")
        print("  and every one has a RETRACTED cell on its Together side —")
        print("  rounds 15 and 16, both in identity.RETRACTED_SWEEPS. Round 21")
        print("  supplied the DeepInfra side; nothing live supplies the other.")
        print()
        print("  So this exhibit is NOT $0 and NOT available now, which is what")
        print("  the plan recorded. Padding it with retracted cells would be")
        print("  worth less than leaving it empty.")
        print()
        for line in _wrap(SUBSTRATE_PENDING):
            print(f"  {line}")
    for r in flips:
        print(f"  {r['model']:<44}{r['level']:<6}"
              + "  ".join(f"{p}={lab}" for p, lab in r["labels"].items()))

    print(f"\n\nLEVEL DIFFERENCES — DOCUMENTATION, NEVER FINDINGS: {len(levels)}")
    print("-" * 72)
    print("  Listed so a reader can see what was EXCLUDED and why. Each of")
    print("  these differs across providers in RATE; the route label differs")
    print("  as a consequence of the rate, not independently of it.")
    for r in levels:
        print(f"  {r['model']:<44}{r['level']:<6}"
              + "  ".join(f"{p}={lab}" for p, lab in r["labels"].items()))

    if undec:
        print(f"\n\nUNDECIDABLE — NO COUNTERFACTUAL ARM: {len(undec)}")
        print("-" * 72)
        for r in undec:
            print(f"  {r['model']:<44}{r['level']:<6}"
                  + "  ".join(f"{p}={lab}" for p, lab in r["labels"].items()))

    print(f"\n\nAGREES ACROSS PROVIDERS: {len(agrees)}")
    print("-" * 72)
    for r in agrees:
        lab = next(iter(r["labels"].values()))
        print(f"  {r['model']:<44}{r['level']:<6}{lab}")

    if blocked:
        print(f"\n\nBLOCKED BY RETRACTION: {len(blocked)}")
        print("-" * 72)
        print("  Measured on two providers; excluded because a side is retracted.")
        print("  These are what Exhibit 1 WOULD be computed from.")
        for b in blocked:
            ret = "  ".join(f"{p}=r{'/'.join(r)}"
                            for p, r in b["retracted"].items())
            print(f"  {b['model']:<40}{b['level']:<6}live={b['live']}  {ret}")

    print("\n" + "=" * 72)
    print(f"{len(rows)} admissible model/level pairs on more than one provider; "
          f"{len(blocked)} blocked by retraction.")
    print("Computed from committed cells. Nothing here is hand-maintained.")
    return 0


def _wrap(text, width=68):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
