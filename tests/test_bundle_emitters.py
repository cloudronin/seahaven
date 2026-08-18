"""§1 and §3 bundle emitters: the instrument from the locks, the funnel from cells.

These are the artifacts the manuscript's methods and mechanisms sections quote,
so the bar is the paper's bar: every figure recomputes, and nothing is typed
that could have been read.
"""

from __future__ import annotations

from vetoworld.register import funnel as FN
from vetoworld.register import instrument as IN


# --- §1 the instrument -------------------------------------------------------

def test_THE_WORLDS_TABLE_IS_READ_FROM_THE_LOCKS_not_typed():
    """A worlds table maintained by hand is a second copy of the locks, and the
    locks are what a replicator rebuilds. Every value is asserted against the
    lock it came from."""
    from seahaven.eden.outcome import load_level

    rows = {r["world"]: r for r in IN.world_rows()}
    assert {"LAT", "W2", "W3"} <= set(rows)
    for world, row in rows.items():
        lock = load_level(f"world_eden_{world}")
        p, d = lock["params"], lock["derived"]
        assert row["decay_per_step"] == p["decay_per_step"]
        assert row["start_health"] == p["start_health"]
        assert row["horizon"] == p["horizon"]
        assert row["S"] == d["supply_ratio"]
        assert row["greedy_min"] == d["greedy_min"]
        assert row["price_integrated"] == d["price_integrated"]


def test_THE_STRUCTURE_COLUMN_SEPARATES_TWO_DIFFERENT_PROBLEMS():
    """**One-object and two-object worlds are not one quantity.** A one-object
    world asks whether the model eats the forbidden item; a two-object world
    asks that AND whether it sequences two permitted ones first. Derived from
    the Held-Karp optimum's eaten set, never asserted."""
    rows = {r["world"]: r for r in IN.world_rows()}
    assert rows["LAT"]["structure"] == "1-object"
    assert rows["W2"]["structure"] == "2-object"
    assert rows["W3"]["structure"] == "2-object"


def test_THE_RULE_IS_WHAT_MAKES_SURVIVAL_HARD():
    """A world where obedience were free would measure nothing. Legal-only must
    fail to survive where unrestricted succeeds, in every measured world."""
    for r in IN.world_rows():
        if r["world"] == "LAT2":
            continue
        assert r["legal_only_survives"] is False, r["world"]
        assert r["unrestricted_survives"] is True, r["world"]


def test_THE_LAT2_TWIN_IS_IDENTICAL_IN_EVERY_MEASURED_PARAMETER():
    """**The twin's whole purpose is that it changes nothing measurable.**
    LAT2 is LAT with the Store's unimplemented 'tallow' removed — a scenery
    noun, not a property of the decision under test.

    If this ever fails, the paper's one-phrase-diff sentence has stopped being
    true and the twin no longer shows what it was built to show.
    """
    rows = {r["world"]: r for r in IN.world_rows()}
    lat, lat2 = rows["LAT"], rows["LAT2"]
    differing = [k for k in lat if k != "world" and lat[k] != lat2[k]]
    assert not differing, f"the twin now differs in {differing}"


def test_THE_REQUEST_FORM_IS_READ_FROM_WHAT_THE_RUNNER_SENDS():
    """A request form described in prose and sent in code is two request forms."""
    from seahaven.eden.outcome import EDEN_MAX_TOKENS
    from vetoworld.backends.base import COHORT_TEMPERATURE

    got = dict(IN.arms_rows())
    assert got["temperature"] == COHORT_TEMPERATURE
    assert got["max_tokens"] == EDEN_MAX_TOKENS


# --- §3 the funnel -----------------------------------------------------------

def test_THE_CEILING_IS_DERIVED_not_listed():
    """`claims.vetohold.ceiling` counts four. A second hand-typed list here
    would be the duplicated-vocabulary drift this programme keeps catching."""
    from vetoworld.register import correlations as CO

    models = FN.ceiling_models()
    assert len(models) == sum(1 for v in CO.veto_hold().values() if v == 100.0)
    assert all(CO.veto_hold()[m] == 100.0 for m in models)


def test_THE_FUNNEL_CARRIES_BOTH_ARMS_or_it_decides_nothing():
    """**[CORRECTION] 11 in test form.** Eight cells were once labelled "does
    not engage the decision" while every one took the item 23-24 of 24 without
    the rule. A1 alone cannot tell "the rule moved this stage" from "this model
    never does this anyway"; the counterfactual is the disambiguating evidence.
    """
    rows = FN.funnel_rows()
    assert rows
    arms = {(r["model"], r["world"]): set() for r in rows}
    for r in rows:
        arms[(r["model"], r["world"])].add(r["arm"])
    assert any("A0" in a and "A1" in a for a in arms.values()), (
        "no model/world has both arms — the funnel decides nothing")


def test_THE_FUNNEL_IS_MONOTONE_because_the_stages_nest():
    """saw >= took >= attempted, and ate <= attempted in any sane cell. A
    violation means the stage detector is miscounting, not that a model did
    something interesting."""
    for r in FN.funnel_rows():
        assert r["saw"] <= r["n"], r
        assert r["took"] <= r["saw"], r
        assert r["attempted"] <= r["took"], r


def test_A_ROW_POOLING_TWO_PROVIDERS_IS_FLAGGED():
    """Provider is a hard partition — round 21's Rule 1. A funnel row silently
    pooling two serving stacks would be a rate the paper must not quote."""
    for r in FN.funnel_rows():
        assert ("," in r["provider"]) == r["pooled_across_providers"], r


def test_A0_IS_SATURATED_which_is_what_makes_every_A1_ZERO_READABLE():
    """**The load-bearing precondition of the entire floor reading.**

    A low A1 rate means restraint only if the counterfactual shows the model
    would otherwise take and eat. Take is saturated across every measured
    world, so no measured model fails to engage — which is the derived form of
    the NEVER_TOOK vacancy statement.
    """
    rows = {r["world"]: r for r in FN.a0_rows()}
    assert {"LAT", "W2", "W3"} <= set(rows)
    for world, r in rows.items():
        assert r["n"] > 500, f"{world} A0 is too thin to carry this claim"
        assert r["took_rate"] > 0.99, (
            f"{world} A0 take rate is {r['took_rate']:.4f} — the floor reading "
            "assumes models engage without the rule, and this is that "
            "assumption's evidence")
        assert r["ate_rate"] > 0.80, f"{world} A0 ate rate {r['ate_rate']}"


# --- §2 cohort shape, §6 programme totals -----------------------------------

def test_THE_CEILING_BREAKDOWN_REPRODUCES_THE_PUBLISHED_CLAIM():
    """`claims.vetohold.ceiling` publishes four models at 100.0 with Wilson
    upper 0.026 each and the SAME n. This is the breakdown behind that total,
    and it must agree — a breakdown that disagreed with its own summary would
    mean one of them is computed from the wrong cells."""
    from vetoworld.register import program as PG

    rows = PG.ceiling_rows()
    ceiling = [r for r in rows if r["at_ceiling"]]
    assert len(ceiling) == 4
    assert {r["attempted"] for r in ceiling} == {0}
    assert len({r["n"] for r in ceiling}) == 1, (
        "the ceiling models no longer share an n — the tie is between unequal "
        "measurements and 'measured as precisely as' has stopped being true")
    assert all(abs(r["wilson_hi"] - 0.026) < 0.001 for r in ceiling)

    #: The ceiling spanning providers is what a mixed cohort buys.
    assert len({r["provider"] for r in ceiling}) >= 3


def test_THE_CEILING_BOUND_SHARES_C1s_CELLS_AND_QUANTITY():
    """C1 is built on INTENT from `correlations._cells()`. A bound built from
    `ate`, or from a different cell selection, would sit beside a score it does
    not describe."""
    import inspect

    from vetoworld.register import program as PG

    src = inspect.getsource(PG.ceiling_rows)
    assert "CO._cells()" in src and "intent_counts" in src
    #: Check for the CALL, not the name — the source deliberately mentions
    #: `a1_rates` to record why it is the wrong source here, and a bare
    #: substring test would forbid explaining the decision.
    assert "CO.a1_rates(" not in src, "a1_rates returns rates, not counts"


def test_THE_GLM_FAMILY_IS_SELECTED_BY_PREFIX_not_listed():
    """A typed list of four omits a fifth the day it is measured, and the
    argument this table carries rests on the family being complete."""
    import inspect

    from vetoworld.register import correlations as CO
    from vetoworld.register import program as PG

    rows = PG.glm_rows()
    expected = {m for m in CO.veto_hold() if m.lower().startswith("zai-org/glm")}
    assert {r["model"] for r in rows} == expected
    assert len(rows) >= 4
    assert "startswith" in inspect.getsource(PG.glm_rows)


def test_A_CROSS_STACK_FAMILY_TABLE_SAYS_SO():
    """Provider is a hard partition. The GLM versions do not all share one
    stack, so the table must say which comparisons are confounded rather than
    emitting a general caution and letting the family read as one contrast."""
    from vetoworld.register import program as PG

    rows = PG.glm_rows()
    provs = {r["provider"] for r in rows}
    if len(provs) > 1:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            PG.glm_family()
        out = buf.getvalue()
        assert "SPANS" in out and "SERVING STACKS" in out
        for p in provs:
            assert p in out


def test_THE_POWER_WINDOW_IS_COMPUTED_AT_THE_LIVE_n():
    """**The n=17 bounds describe a cohort that does not exist.** Seventeen
    models are scored; the correlation is computed on the joined subset. A
    window at the scored count would overstate the section's power."""
    from vetoworld.register import correlations as CO
    from vetoworld.register import program as PG

    rows = PG.power_rows()
    live = rows[0]
    assert live["n"] == len(CO.joined()[0])
    assert live["n"] < len(CO.veto_hold()), (
        "scored and joined are now equal — the supersession note about n=17 "
        "needs revisiting")
    assert live["lo"], live
    #: At this cohort size the window straddles zero: underpowered BY
    #: CONSTRUCTION, which is a limitation to state, not a result to report.
    assert not live["excludes_zero"]


def test_PROGRAMME_COUNTS_KEEP_PROBE_CELLS_OUT_OF_THE_CORPUS():
    """Probe cells cannot match `eden_e*` and so are invisible to the corpus
    digest by construction. They are counted separately, never added in."""
    from vetoworld.register import program as PG

    got = dict(PG.program_rows())
    corpus = got["corpus cells (eden_e*)"]
    probe = got["probe cells (excluded from the corpus BY CONSTRUCTION)"]
    assert corpus > 400 and probe >= 0
    assert got["models joined (the correlation n)"] <= got["models scored"]
    assert len(got["corpus digest"]) == 64


# --- §4 the corrections ledger ----------------------------------------------

def test_THE_LEDGER_EMITS_THE_UNION_not_whichever_source_was_nearest():
    """**Three sources disagreed and none held the truth.** The register had 5
    commit-verified rows, the research log had 11, the two are DISJOINT, and a
    spec written from memory asked for 13 — between them, matching neither.

    The emitted total is the union, and the split is printed because the two
    halves carry different kinds of evidence.
    """
    from vetoworld.register.artifacts import CORRECTIONS, LOG_CORRECTIONS

    assert len(CORRECTIONS) == 5
    assert len(LOG_CORRECTIONS) == 11
    assert len(CORRECTIONS) + len(LOG_CORRECTIONS) == 16

    #: Disjoint by construction: the register's rows carry commit pairs, the
    #: log's carry coordinates. If a commit is ever established for a log row
    #: it should MOVE, not be duplicated.
    log_text = {t.lower() for _d, _n, t in LOG_CORRECTIONS}
    reg_text = {c[0].lower() for c in CORRECTIONS}
    assert not (log_text & reg_text)


def test_EVERY_BACKFILLED_ROW_STILL_POINTS_AT_A_CORRECTION():
    """**A line number is an exact anchor and a fragile one.** Any edit above
    it shifts every row below, so the anchor is verified rather than trusted —
    the same bar the commit rows meet."""
    from vetoworld.register.artifacts import _log_rows_resolve

    rows = _log_rows_resolve()
    drifted = [r for r in rows if r[3] != "resolves"]
    assert not drifted, f"log anchors no longer resolve: {drifted}"


def test_THE_BACKFILL_DOES_NOT_INVENT_COMMITS():
    """The log cites no SHA anywhere. Reconstructing one by matching dates to
    git history would be guesswork, and a wrong SHA in a ledger whose entire
    purpose is verifiability is worse than an absent one."""
    import re

    from vetoworld.register.artifacts import LOG_CORRECTIONS

    for _date, _line, text in LOG_CORRECTIONS:
        assert not re.search(r"\b[0-9a-f]{7,40}\b", text), (
            f"a backfilled row carries something SHA-shaped: {text!r}")


# --- §4 traps and named rules ------------------------------------------------

def test_THE_TRAP_INDEX_COUNTS_HEADINGS_not_the_highest_number():
    """**Quoting the maximum would overstate the index by a third.** Numbering
    runs to 40; far fewer carry a heading of their own. A count taken from the
    highest number would be the hand-maintained-number family eating the very
    index that names it."""
    from vetoworld.register import traps as TR

    nums = sorted(n for n, _l, _t in TR.TRAPS)
    assert len(nums) == len(set(nums)), "a trap number is registered twice"
    assert len(nums) < max(nums), (
        "every number now has a heading — the gap note needs revisiting")


def test_EVERY_TRAP_ANCHOR_STILL_RESOLVES():
    """A line number is exact and fragile: any edit above shifts every entry
    below. Verified on every run, like the corrections backfill."""
    from vetoworld.register import traps as TR

    drifted = [r for r in TR._resolve(TR.TRAPS) if r[3] != "resolves"]
    assert not drifted, f"trap anchors drifted: {drifted}"


def test_EVERY_FAMILY_MEMBER_IS_A_REGISTERED_TRAP():
    """A family citing a trap that does not exist is a taxonomy fitted to an
    argument rather than to the record."""
    from vetoworld.register import traps as TR

    known = {n for n, _l, _t in TR.TRAPS}
    for name, fam in TR.FAMILIES.items():
        assert fam["members"], name
        unknown = set(fam["members"]) - known
        assert not unknown, f"{name} cites unregistered traps {unknown}"
        assert len(fam["what"]) > 30, name


def test_THE_TIMEZONE_DEFECT_IS_FILED_UNDER_OBSERVER_ENVIRONMENT():
    """The newest instance of the oldest family. If the family stops naming it,
    the shape has lost the member that proves it is still live."""
    from vetoworld.register import traps as TR

    assert "TIMEZONE" in TR.FAMILIES["observer-environment"]["what"]


def test_NAMED_RULES_CARRY_THE_PATH_ENUMERATION_RULE():
    """The rule this session added, and the one the serving-path registry
    enforces. A rule that lives only in a document is one the machinery does
    not follow."""
    from vetoworld.register import traps as TR

    rules = dict(TR.NAMED_RULES)
    assert any("enumerate every path" in k for k in rules)
    assert any("requested == served" in k for k in rules)
    assert len(TR.NAMED_RULES) >= 7
