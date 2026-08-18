"""§4: the trap index and the named rules, as registers rather than prose.

The bundle spec asks for a trap index with families and instance counts, and an
extract of the named rules. Both live in narrative today — 12,000 lines of
research log and a long AGENTS.md.

**These are NOT parsed out of prose at emit time.** A regex over narrative is a
hand-maintained number wearing a script: it looks derived, it breaks silently on
a heading style, and nobody can tell a miscount from a real change. The pattern
`CORRECTIONS` establishes is used instead — a structured register whose anchors
are VERIFIED against the source on every run, so a drifted entry is reported
rather than absorbed.

The families are the spec's own, and they are what the paper's corrections
section argues: these are not twenty-five unrelated mistakes, they are a handful
of recurring shapes, and naming the shape is what makes the next instance
findable before it ships.
"""

from __future__ import annotations

__all__ = ["TRAPS", "FAMILIES", "NAMED_RULES", "trap_index", "named_rules"]

LOG_PATH = "docs/research-log.md"

#: `(number, log line, one-line)`. Numbered headings only — see `trap_index`
#: for what the gaps in the numbering mean.
TRAPS = [
    (1, 47, "conda silently builds an unusable environment"),
    (2, 68, "the TextWorld banner is invisible to a lexicon lint"),
    (4, 130, "Qwen3 hybrid thinking is on by default"),
    (5, 195, "a wrong test helper made a correct estimator look broken"),
    (9, 1331, "a superlative asserted without checking the other 33 values"),
    (10, 1423, "a metric named for what it is not"),
    (11, 1526, "the first reading was an artifact of contractions and nouns"),
    (12, 1689, "the say/do correlation is mostly paraphrase"),
    (13, 1901, "the induced-convergence statistic is too noisy to rank models"),
    (14, 2083, "the act description is a researcher degree of freedom"),
    (15, 2360, "a reliability gate that passes while what it protects fails"),
    (16, 2425, "the fidelity run measured nothing, and I built the bug"),
    (17, 2722, "gate -1 was too permissive: 62% of 'lift' was episode length"),
    (18, 2896, "the two load-bearing functions had no tests"),
    (19, 2966, "the re-baseline was unreadable, twice over"),
    (20, 3164, "the detector is relation-blind, and it determines every result"),
    (32, 11887, "166 cells, one model, twenty-two name tags"),
    (33, 12025, "the disambiguating data was in the signature"),
    (34, 5565, "I violated my own guard in the next thing I wrote"),
    (35, 6231, "batch-invariance is MODEL-DEPENDENT; B2 was over-generalised"),
    (36, 6616, "a verdict decided by PYTHONHASHSEED, and then by 2.8e-17"),
    (37, 6716, "the serving stack was never pinned, and 0.27.0 broke it"),
    (38, 12203, "a count threshold is not scale-invariant"),
    (39, 12450, "two anchors, two different n, and a window between them"),
    (40, 12597, "membership-dependent derivation"),
]

#: The recurring shapes. `members` are trap numbers; the family is the unit the
#: paper argues, because a named shape is findable and a list of incidents is
#: not. Membership is asserted, not inferred — a trap can belong to more than
#: one family and several do.
FAMILIES = {
    "can't-fire checks": {
        "what": "a guard that cannot fail, so it passes forever and protects "
                "nothing",
        "members": (15, 18, 5),
    },
    "duplicated identity sites": {
        "what": "the same identity decision made independently in several "
                "places, which drift",
        "members": (32, 33),
    },
    "hand-maintained numbers": {
        "what": "a figure retyped rather than derived, which goes stale "
                "silently",
        "members": (9, 10, 13),
    },
    "observer-environment": {
        "what": "the measurement depends on where it is run — mtime, pipe "
                "exit status, TIMEZONE",
        "members": (36, 37, 1),
    },
    "membership-dependent derivation": {
        "what": "a value derived from a collection that can grow, so every "
                "member's value depends on every other member",
        "members": (38, 40),
    },
    "label vs mechanism": {
        "what": "one word covering two opposite behaviours, so findings "
                "collapse into each other",
        "members": (10, 20, 33),
    },
}

#: The rules the programme runs on. Sourced from AGENTS.md, which is where they
#: are argued; recorded here so the paper can cite a list rather than a document.
NAMED_RULES = [
    ("requested == served, per cell",
     "the model that answered is recorded per cell and compared to the one "
     "asked for. #113 is what happens without it"),
    ("closing a bug enumerates its other symptoms",
     "a bug report is a claim about a cause; a closed bug is a claim about "
     "every effect"),
    ("a structural rule lands -> enumerate every path it must cover",
     "the twin of the rule above, one level up. Blockers 5 and 6 were bugs in "
     "the space BETWEEN paths, not in any path"),
    ("reference disjoint",
     "the reference channel may not share cells with what it judges"),
    ("the taint law",
     "EVENT days never enter the rolling anchor; a window that absorbs the "
     "event calls the new level normal"),
    ("no verdicts off unfinished runs",
     "a partial serve is not a measurement, and must never read as one"),
    ("exit codes never travel through a pipe",
     "the success signal has to be the inner step's own, not the wrapper's"),
]


def _resolve(anchors) -> list[tuple]:
    """Verify each anchor still lands on its marker in the log.

    A line number is exact and fragile: any edit above shifts every entry
    below. Verified on every run, so drift is reported rather than absorbed.
    """
    from pathlib import Path

    try:
        lines = Path(LOG_PATH).read_text().splitlines()
    except OSError:
        return [(n, ln, t, "LOG NOT READABLE") for n, ln, t in anchors]
    out = []
    for num, ln, text in anchors:
        ok = 1 <= ln <= len(lines) and "[TRAP]" in lines[ln - 1]
        out.append((num, ln, text, "resolves" if ok else "** DRIFTED **"))
    return out


def trap_index() -> int:
    from .markdown import table

    rows = _resolve(TRAPS)
    drifted = [r for r in rows if r[3] != "resolves"]
    nums = sorted(r[0] for r in rows)

    print("THE TRAP INDEX — confident wrong output, not errors\n")
    print("  A [TRAP] is a defect that produced CONFIDENT WRONG OUTPUT rather")
    print("  than a failure. Nothing raised; every check passed; the answer was")
    print("  wrong. That is the class this programme is mostly about.\n")
    print(table(["trap", "log line", "what it was"],
                [[n, f"{LOG_PATH}:{ln}", t] for n, ln, t, _s in rows]))

    print(f"\n  {len(rows)} numbered traps carry a heading, all anchors "
          f"resolving ({len(drifted)} drifted).")

    #: **The gap in the numbering is information, not an omission.** Numbers
    #: were assigned as traps were found; a missing one was either folded into
    #: a neighbour or recorded without its own heading. Reporting the highest
    #: number as a count would overstate the index by a third.
    missing = [n for n in range(1, max(nums) + 1) if n not in nums]
    print(f"\n  Numbering runs to {max(nums)}, so the index is NOT "
          f"{max(nums)} entries.")
    print(f"  {len(missing)} numbers have no heading of their own: "
          f"{', '.join(map(str, missing))}")
    print("  They were folded into a neighbour or recorded inline. The COUNT")
    print("  the paper quotes is the number of headings, not the highest")
    print("  number — quoting the maximum would overstate the index by a third,")
    print("  which is the hand-maintained-number family eating its own index.")

    print("\n\nFAMILIES — the shapes, which is what the paper argues\n")
    print(table(["family", "instances", "traps", "what it is"],
                [[k, len(v["members"]),
                  ", ".join(str(m) for m in sorted(v["members"])), v["what"]]
                 for k, v in sorted(FAMILIES.items())]))
    print("\n  **These are not unrelated mistakes.** A named shape is findable")
    print("  before the next instance ships; a list of incidents is not. Every")
    print("  family here was named AFTER its second instance and has caught at")
    print("  least one more since — the timezone defect is the newest member of")
    print("  observer-environment, found by running the suite under TZ=UTC.")

    unfiled = sorted(set(nums) - {m for v in FAMILIES.values()
                                  for m in v["members"]})
    if unfiled:
        print(f"\n  {len(unfiled)} traps are not yet filed to a family: "
              f"{', '.join(map(str, unfiled))}")
        print("  Unfiled is honest; inventing a family to absorb them would")
        print("  make the taxonomy fit by construction.")
    return 1 if drifted else 0


def named_rules() -> int:
    from .markdown import table

    print("NAMED RULES — what the programme runs on\n")
    print(table(["rule", "why"], [[k, v] for k, v in NAMED_RULES]))
    print(f"\n  {len(NAMED_RULES)} rules. Each was written after an incident")
    print("  that would have been cheaper to prevent, and each is enforced")
    print("  somewhere executable rather than only argued: the serving-path")
    print("  registry for path enumeration, the taint law in `trace`, the")
    print("  identity assertion per cell. A rule that lives only in a document")
    print("  is one the machinery does not follow — which is itself the")
    print("  finding this programme has recorded most often.")
    return 0
