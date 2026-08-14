"""Ingest the raidex pool and freeze the raidex -> Together mapping.

**THE MAPPING IS THE POINT, NOT THE INGEST.** raidex ids are provider-prefixed —
`openrouter/deepseek/deepseek-v4-pro`, `sambanova/gemma-4-31B-it` — and Together
strings are `deepseek-ai/DeepSeek-V4-Pro`. Measured before any planning:

    43 raidex models, 40 at 9/9 coverage
    73 Together strings on record
    EXACT string matches:                 0
    same model, different provider:      17
    raidex models measured ON Together:   1   (thinkingmachines/Inkling)

**So standing requirement 1 cannot be satisfied as written and the round proceeds
on a NAMED, UNTESTED ASSUMPTION: provider-invariance.** Every correlate this
enables pairs a Together-served behavioural rate with an OpenRouter- or
SambaNova-served RAI score — same weights, different serving stack, possibly
different quantization and sampling defaults.

Round 4 CLOSED the capability route rather than substitute a near-miss variant.
This is a weaker version of that compromise, defensible because it is the same
model rather than a different one, and it is a **limitation of every number the
round produces** rather than a footnote. `measured_on` is carried per record so
no correlation can silently lose which stack produced its x-axis.

Normalisation is deliberately narrow: strip the provider path, lowercase, drop
separators, drop a trailing serving suffix (`instruct`, `turbo`, `it`, `chat`,
`hf`). It does **not** bridge version or size differences — `GLM-5.1` and
`GLM-5.2` stay distinct, which is what round 4 refused to collapse.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "cloudronin/raidex-results"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
BLOB = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT = Path("results/raidex_pool.json")
AVAIL = Path("results/together_availability.json")

#: Served strings the program has ALREADY committed episodes against. Used only
#: to break a tie between two Together listings of the same weights — never to
#: create a match that normalisation did not already find.
from seahaven.eden.round4 import COHORT as _R4  # noqa: E402
COMMITTED = frozenset(_R4)

#: Serving suffixes only. Version and size tokens are NOT stripped.
_SUFFIX = re.compile(r"(instruct|turbo|it|chat|hf)$")


def normalise(model_id: str) -> str:
    """The join key. Narrow on purpose — see the module docstring."""
    s = model_id.split("/")[-1].lower()
    s = re.sub(r"[-_.]", "", s)
    return _SUFFIX.sub("", s)


def together_strings() -> set[str]:
    av = json.loads(AVAIL.read_text())
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("id", "name", "served_name", "model") and isinstance(v, str):
                    out.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(av)
    return out


def fetch() -> list[tuple[str, dict]]:
    tree = json.load(urllib.request.urlopen(TREE, timeout=60))
    files = [x["path"] for x in tree if x["path"].endswith(".json")]
    recs = []
    for f in files:
        d = json.load(urllib.request.urlopen(BLOB + f, timeout=60))
        recs.append((f[:-5].replace("__", "/"), d))
    return recs


def main(argv: list[str] | None = None) -> int:
    # **THE POOL IS A FROZEN AXIS, NOT A CACHE.**
    #
    # Every raidex correlate — round 10's, and round 15's E1 — plots vworld
    # rates against these scores. Re-running this against a changed upstream
    # would move the x-axis under results that are already published, silently,
    # with no pin breaking: the pool is data, not a hashed artifact, so nothing
    # else would notice.
    #
    # So a rebuild is now a deliberate act. Same refusal `corpus fetch` makes
    # about an existing corpus, for the same reason.
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if OUT.exists() and "--force" not in argv:
        have = json.loads(OUT.read_text())
        print(f"REFUSING: {OUT} already exists "
              f"({have.get('n_models')} models, retrieved {have.get('retrieved')}).")
        print("\n  This file is the frozen x-axis of every raidex correlate.")
        print("  Rebuilding it against a changed upstream would move the axis")
        print("  under results already published, and no pin would break —")
        print("  the pool is data, so nothing else would notice.")
        print("\n  Pass --force if you mean to re-freeze it, and re-run every")
        print("  correlate that cites it in the same commit.")
        return 1

    recs = fetch()
    tg = together_strings()
    by_norm: dict[str, list[str]] = {}
    for n in tg:
        by_norm.setdefault(normalise(n), []).append(n)

    # **The exact-match count is recorded, not just the relaxed one.** It is the
    # evidence that the assumption was needed rather than chosen.
    exact = sorted({r for r, _ in recs} & tg)

    pool = []
    for rid, d in recs:
        comp = d.get("composite", {}) or {}
        key = normalise(rid)
        matches = by_norm.get(key, [])
        # **An ambiguity is resolved only by a PRIOR COMMITMENT, never by
        # picking.** `gemma-4-31B-it` is listed on Together twice —
        # `google/gemma-4-31B-it` and `pearl-ai/gemma-4-31b-it`, two providers
        # reselling the same weights. Choosing between them on plausibility is
        # exactly the near-miss standing requirement 1 refuses. The program has
        # served `google/...` for 96 committed episodes, so that commitment
        # disambiguates; anything with no such commitment stays ambiguous and
        # unmapped rather than guessed.
        chosen = matches[0] if len(matches) == 1 else None
        if chosen is None and matches:
            prior = [m for m in matches if m in COMMITTED]
            if len(prior) == 1:
                chosen = prior[0]
        pool.append({
            "raidex_id": rid,
            # The stack that produced the RAI score. Carried so no correlation
            # can lose it.
            "measured_on": rid.split("/")[0],
            "rai_score": comp.get("rai_score"),
            "rai_coverage": comp.get("rai_coverage"),
            "rai_coverage_pct": comp.get("rai_coverage_pct"),
            "dimension_scores": comp.get("dimension_scores", {}),
            "together_served_name": chosen,
            "together_candidates": sorted(matches),
            "disambiguated_by_prior_commitment": bool(
                chosen and len(matches) > 1),
            "exact_string_match": rid in tg,
        })

    full = [p for p in pool if str(p["rai_coverage"]) in ("9/9", "9")]
    mapped = [p for p in full if p["together_served_name"]]
    ambiguous = [p for p in pool if len(p["together_candidates"]) > 1
                 and not p["together_served_name"]]
    resolved = [p for p in pool if p.get("disambiguated_by_prior_commitment")]

    doc = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "retrieved": date.today().isoformat(),
        "n_models": len(pool),
        "n_full_coverage": len(full),
        "n_mapped_full_coverage": len(mapped),
        "exact_string_matches": exact,
        # **The assumption, written into the artifact rather than only the code.**
        "provider_invariance_assumption": (
            "raidex ids are provider-prefixed and Together strings are not, so "
            "EXACT-string matching yields zero models. This pool is joined on "
            "the model identity with serving suffixes stripped, which assumes a "
            "model behaves the same served by OpenRouter/SambaNova as by "
            f"Together. That is UNTESTED. Only {sum(1 for p in pool if p['measured_on'] == 'together_ai')} "
            "of the pool was measured on Together itself. Round 4 closed the "
            "capability route rather than accept a near-miss VARIANT; this is a "
            "weaker version of the same compromise on the same weights, and is "
            "a limitation of every correlate, not a footnote."),
        "normalisation": ("provider path stripped, lowercased, separators "
                          "removed, trailing serving suffix removed "
                          "(instruct|turbo|it|chat|hf). Version and size tokens "
                          "are NOT stripped."),
        "ambiguous_matches": [p["raidex_id"] for p in ambiguous],
        "models": pool,
    }
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"raidex models        : {len(pool)}")
    print(f"9/9 coverage         : {len(full)}")
    print(f"EXACT string matches : {len(exact)}  <- the reason for the assumption")
    print(f"mapped at 9/9        : {len(mapped)}")
    print(f"ambiguous, UNMAPPED  : {len(ambiguous)} "
          f"{[p['raidex_id'] for p in ambiguous]}")
    for p in resolved:
        print(f"tie broken by prior commitment: {p['raidex_id']} -> "
              f"{p['together_served_name']}  (candidates "
              f"{p['together_candidates']})")
    print(f"measured on Together : "
          f"{sum(1 for p in pool if p['measured_on'] == 'together_ai')}")
    print(f"\nwrote {OUT}")
    print(f"\n{'together string':<44}{'rai':>6}  measured_on")
    for p in sorted(mapped, key=lambda x: -(x["rai_score"] or 0)):
        print(f"  {p['together_served_name']:<42}{p['rai_score']:>6.1f}  "
              f"{p['measured_on']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
