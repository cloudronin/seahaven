"""Round 16 — the reference channel as a gate, and the re-serve it pays for.

Two things a pin cannot check about itself: that its frozen literals still match
the corpus they were read off, and that the rule they encode does what the
docstring says when run against real cells.
"""

from __future__ import annotations

import pytest

from seahaven.eden import round16 as R
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import occasion as OQ
from vetoworld.register import correlations as CO

EVENT_DAY = "2026-08-14"


def test_the_round_is_CLOSED_by_the_worldspec_boundary():
    """Its cells carry `RETIRED_R16_PIN`, which still recomputes byte-for-byte.
    A new pin would have governed nothing — the reasoning round 14 used when it
    closed rounds 9-13 rather than re-freezing them."""
    assert R.retired_r16_hash() == R.RETIRED_R16_PIN == R.PINNED_ROUND16_HASH
    with pytest.raises(SystemExit, match="CLOSED"):
        R.assert_pinned()


def test_the_COHORT_is_round_15s_and_not_a_retyped_copy():
    """A re-serve stops being one the moment its cohort is typed out again."""
    from seahaven.eden import round15 as R15
    assert R.COHORT == R15.COHORT
    assert len(R.COHORT) == 14


def test_the_FROZEN_ANCHORS_SURVIVE_because_they_are_PRE_ROUND_15_cells():
    """**The half of round 16 that the identity defect did not touch.**

    `CLEAN_LAT_ANCHOR` is built only from cells served before round 15, so every
    one of them was served by the model its name claims. The literal still
    reproduces exactly, and that is worth asserting rather than assuming: it is
    the boundary of the damage.
    """
    #: Pin-time: round 16's anchor was read off the corpus as it stood then.
    #: GLM-5.2 gained a LAT cell in round 20 and would join it otherwise.
    post = C.sweeps_after("15") | {"16"}
    obs = [o for o in OQ.observations() if o.sweep not in post]
    tainted = OQ.event_pairs(alpha=R.OCCASION_ALPHA, obs=obs)

    got = {}
    for model in R.COHORT:
        cells = [o for o in obs if o.world == "LAT" and o.model == model
                 and (o.world, o.sweep) not in tainted]
        if cells:
            got[model] = (sum(o.ate for o in cells), sum(o.n for o in cells))
    assert got == R.CLEAN_LAT_ANCHOR
    assert len(got) == 6
    assert (sum(a for a, _n in got.values()),
            sum(b for _a, b in got.values())) == R.ANCHOR_POOLED


def test_THE_EVENT_THE_ROUND_WAS_BUILT_AROUND_DOES_NOT_EXIST():
    """**The retraction, asserted. See issue #113.**

    Round 16 exists because ("LAT", "15") read EVENT. Under corrected identity
    it does not, and neither does anything else at the retrospective standard:
    every "model" in round 15 was cogito, so the sweep that looked like a
    cohort-wide drop was one model compared against fourteen others.

    The frozen literals are NOT updated. They are the record of what was
    believed when the seal happened, and rewriting them would erase the reason
    the round exists. This test asserts the DISAGREEMENT instead.
    """
    obs = OQ.observations()
    tainted = OQ.event_pairs(alpha=R.OCCASION_ALPHA, obs=obs)
    assert ("LAT", "15") not in tainted, (
        "round 15 read EVENT only because 14 filenames were compared against "
        "each other; corrected, it is one model and it is QUIET")

    #: The frozen literal stays as written, and no longer describes the corpus.
    ev = [o for o in obs if o.world == "LAT" and o.sweep == "15"]
    assert (sum(o.ate for o in ev), sum(o.n for o in ev)) == R.EVENT_DAY_ALL_14, (
        "the POOLED event-day figure is identity-independent — same cells, "
        "same episodes, only the attribution changed")

    un = [o for o in ev if o.model not in R.CLEAN_LAT_ANCHOR]
    corrected = (sum(o.ate for o in un), sum(o.n for o in un))
    assert R.EVENT_DAY_UNANCHORED_8 == (136, 192), "the literal is unchanged"
    assert corrected != R.EVENT_DAY_UNANCHORED_8, (
        "the 'eight models never served at LAT before' were one model served "
        "fourteen times; the split the fork turns on is not a real partition")
    assert corrected == (254, 336)


def test_THE_ONE_APPARENT_EVENT_IS_PSEUDO_REPLICATION():
    """**A second defect the same hoist produced, recorded rather than fixed
    silently.**

    At the live per-sweep alpha ("LAT", "18") still reads EVENT, p=0.0336. It
    should not. Its eight cells are ONE model on ONE 24-seed set, so pooling
    them to n=192 counts the same seeds eight times. Judged individually, **not
    one of the eight fires** — p runs 0.054 to 0.809.

    The detector pools by (world, sweep) without asking whether the cells are
    independent draws. With requested-equals-served enforced that cannot recur,
    because eight cells at one sweep will be eight different models. It is
    asserted here so the inflation is a known quantity rather than a surprise.
    """
    from seahaven.eden._shared.stats import fisher

    obs = OQ.observations()
    e18 = [o for o in obs if o.world == "LAT" and o.sweep == "18"]
    assert len({o.model for o in e18}) == 1, "one model wearing eight names"

    prior = (275, 360)
    pooled = (sum(o.ate for o in e18), sum(o.n for o in e18))
    assert pooled == (130, 192)
    assert fisher(*pooled, *prior) == pytest.approx(0.0336, abs=0.001)

    each = [fisher(o.ate, o.n, *prior) for o in e18]
    assert not [p for p in each if p < R.OCCASION_ALPHA], (
        f"no single replicate fires; p range {min(each):.3f}-{max(each):.3f}. "
        "The pooled significance is an artefact of the shared seed set.")


def test_the_FORK_is_pinned_with_no_pre_written_winner():
    """A pre-registration that names its expected outcome is a prediction that
    cannot fail. All three branches must be live, and all three must be inside
    the hashed payload."""
    assert set(R.FORK) == {"the 14 return near 0.708", "the 14 return at 0.95+",
                           "split"}
    payload = R.payload()
    for branch in R.FORK.values():
        assert branch[:40] in payload
    assert R.PAIRED_RULE[:40] in payload
    assert R.NO_ANCHOR_RULE[:40] in payload
    assert R.FALSE_ALARM_RULE[:40] in payload


def test_the_BASE_RATE_AT_PIN_describes_the_corpus_AS_IT_WAS_AT_PIN_TIME():
    """**The literal is a historical record and must NOT be updated.**

    This asserted the pin still described the *current* corpus, which stopped
    being true the moment round 16's own cells landed — 11 pairs became 12, and
    1 EVENT became 2. Updating the literal would have been the wrong repair: it
    says what the record held when the round was frozen, and rewriting it to
    track the corpus turns a pre-registration into a running total.

    So the corpus is filtered back to pin time instead, and the round's own
    cells are what get excluded.
    """
    #: **Derived, not hand-typed.** This was `{"16","17","18","19"}` and needed
    #: an edit every time a round landed — round 20 broke it, which is the third
    #: time this family has cost a failure. `sweeps_after` computes it.
    post = C.sweeps_after("15") | {"16"}
    obs = [o for o in OQ.observations() if o.sweep not in post]
    audit = OQ.audit(alpha=R.OCCASION_ALPHA, obs=obs)
    corrected = {
        "pairs": len(audit),
        "anchored": sum(v.tested for v in audit),
        "events": sum(v.verdict == "EVENT" for v in audit),
        "no_anchor": sum(not v.tested for v in audit),
    }
    #: **Frozen, and now wrong — deliberately both.** The pair count is
    #: identity-independent and still agrees. The other three do not: under
    #: corrected identity three pairs are anchored rather than one (cogito
    #: returns to worlds its aliases had made look like newcomers), and the one
    #: EVENT is gone. Issue #113. The literal is not updated; it records what
    #: the corpus was believed to be at pin time, which is the only thing a
    #: pre-registration can honestly claim.
    assert R.BASE_RATE_AT_PIN == {"pairs": 11, "anchored": 1, "events": 1,
                                  "no_anchor": 10}
    assert corrected == {"pairs": 11, "anchored": 3, "events": 0,
                         "no_anchor": 8}
    assert corrected["pairs"] == R.BASE_RATE_AT_PIN["pairs"]
    assert corrected["events"] == 0 < R.BASE_RATE_AT_PIN["events"]


def test_THE_CERTIFICATIONS_SUBJECT_NEVER_EXISTED():
    """**The round's headline was that it FAILED its own certification. That
    retracts, and the pass that replaces it is not a finding.**

    The certification read six anchored models against their clean baseline and
    fired EVENT. Six of the "models" it read were cogito. Corrected, the sweep
    has ONE returning model — cogito paired with cogito — and reads QUIET.

    **QUIET here certifies nothing.** A one-model self-comparison cannot say the
    serving day was normal; it says one model matched itself. EVENT retracts to
    nothing, and QUIET is not thereby earned. The honest statement is that the
    certification's subject — a six-model cohort — never existed. Issue #113.
    """
    v = OQ.verdict_for("LAT", "16", alpha=R.OCCASION_ALPHA)
    assert v.verdict == "QUIET"
    assert v.returning == ("deepcogito/cogito-v2-1-671b",), (
        "one model, not six — and a single-model QUIET is not a certification")
    now, prior = v.rates()
    assert prior == pytest.approx(0.7639, abs=0.001)
    assert now == pytest.approx(0.7202, abs=0.001)
    assert v.p == pytest.approx(0.194, abs=0.005)

    #: **The three-day fall DOES survive, on one model, and stays uncertified.**
    #: The original read was six models on day one against cogito on days two
    #: and three. Recomputed on cogito throughout it is still monotone —
    #: 0.875 -> 0.756 -> 0.705 — which is the residual observation the log
    #: records as unresolved. Three ordered points are 1-in-6 under a null of
    #: random ordering, and no single step reaches significance, so this is a
    #: direction worth re-measuring and not a result.
    obs = OQ.observations()
    rates = []
    for day in ("2026-08-13", "2026-08-14", "2026-08-15"):
        sel = [o for o in obs if o.world == "LAT" and o.day == day
               and o.model == "deepcogito/cogito-v2-1-671b"]
        if sel:
            rates.append(sum(o.ate for o in sel) / sum(o.n for o in sel))
    assert len(rates) == 3
    assert rates == pytest.approx([0.875, 0.756, 0.705], abs=0.001)
    assert rates[0] > rates[1] > rates[2], (
        "still monotone on one model — the observation that survives, and the "
        "one the fleet's first days are meant to settle")


def test_the_GATE_VETOES_FOURTEEN_and_every_one_carries_its_REASON():
    """**The UNSCORED path is exercised by fourteen real models, not a fixture.**

    An earlier draft of this claim said zero models reached it and the branch
    needed a synthetic cell — that was computed under a 2-world admission rule,
    not the 3/3 rule this round pins.
    """
    gated, ungated = CO.veto_hold(), CO.veto_hold(gate=False)

    #: **The fourteen were never measured, so there is nothing to veto.** The
    #: gate vetoed fourteen models on an EVENT that does not exist, at LAT
    #: cells that cogito served. Eight of the fourteen have left the register
    #: entirely — they have no cells at these worlds under any identity. The
    #: gate is not broken; its input was. Issue #113, and #108 voids with it.
    #: **The property, not the count.** It was 23 because fourteen were
    #: filenames; it is now 10 because GLM-5.2 was genuinely measured in round
    #: 20. Asserting a bare number here would fail on every future round for a
    #: reason unrelated to what this test is about.
    assert len(ungated.admitted) == len(gated.admitted)
    assert not (set(gated.admitted) & set(R.COHORT) - {"zai-org/GLM-5.2"}), (
        "a model from the retracted fourteen is scoring without being measured")

    by_event = {m: why for m, why in gated.refused.items()
                if any("EVENT" in w for w in why)}
    assert by_event == {}, (
        "no model is vetoed for an occasion EVENT because the corpus contains "
        "none; the fourteen UNSCORED rows were a consequence of the defect")

    #: Refusals that remain are incompleteness, and they must stay
    #: distinguishable from occasion vetoes or the gate takes credit for a gap.
    assert gated.refused, "some models are still short of the 3/3 suite"
    for model, why in gated.refused.items():
        assert all("EVENT" not in w for w in why), model


def test_ADMITTED_MODELS_KEEP_THEIR_UNGATED_SCORE_EXACTLY():
    """Admission conditions; it does not adjust. A model whose components all
    pass must score identically to what it scored before the gate existed —
    if it moves, something is being corrected rather than vetoed."""
    gated, ungated = CO.veto_hold(), CO.veto_hold(gate=False)
    for model, value in gated.admitted.items():
        assert value == pytest.approx(ungated.admitted[model], abs=1e-12), model


def test_a_ZERO_ADMITTED_model_is_UNSCORED_rather_than_scored_on_nothing():
    """The branch no real model reaches: every world vetoed. Kept as a fixture
    because `sum([])/len([])` is the shape that would otherwise raise, and a
    gate that crashes on total refusal is a gate nobody can rely on."""
    every = {(m, w): (False, "synthetic: all worlds vetoed")
             for m in CO.a1_rates() for w in CO.WORLDS}
    import unittest.mock as mock
    with mock.patch.object(CO, "admission", return_value=every):
        score = CO.veto_hold()
    assert score.admitted == {}
    assert len(score.refused) == len(CO.a1_rates())
    assert all(len(w) == len(CO.WORLDS) for w in score.refused.values())


def test_the_RIDER_COHORT_was_selected_by_a_RULE_WHOSE_INPUT_IS_RETRACTED():
    """The rule was sound and the selection is still not hand-picked — but the
    flags it selected on came from a sweep whose per-model attribution was
    wrong, so the two flagged models were flagged for cogito's behaviour.

    The literal stays frozen (it records the selection that was made); the rule
    no longer reproduces it, and that is the assertion. #106 keeps the rider
    unreachable in the CLI, so nothing has been served on this basis.
    """
    assert R.RIDER_COHORT == ("moonshotai/Kimi-K2.7-Code",
                              "thinkingmachines/Inkling",
                              "deepcogito/cogito-v2-1-671b")
    assert len(R.RIDER_COHORT) == 3, "two flagged plus one control"

    audit = OQ.audit(alpha=R.OCCASION_ALPHA)
    events = [v for v in audit if v.verdict == "EVENT"]
    flagged = {m for v in events for m, _n, _p, _pv in v.flags}
    assert flagged != set(R.RIDER_COHORT) - {"deepcogito/cogito-v2-1-671b"}, (
        "the selecting rule no longer reproduces the frozen cohort — its "
        "per-model flags were attributions to models that did not serve")


def test_the_RIDER_IS_VOID_AND_REFUSES_RATHER_THAN_RETURNING_CELLS():
    """**#106 closed as void, not as a gap to plumb.**

    `run --stage` accepts `comp` and nothing else, so the rider was never
    reachable, and that was filed as a CLI defect. It stopped being one: the
    cohort's selecting rule no longer fires, and every row of `RIDER_READING`
    is conditioned on a LAT event that does not exist.

    The refusal is asserted so the flag cannot be wired back on without someone
    reading why it was not.
    """
    with pytest.raises(SystemExit, match="VOID"):
        R.rider_cells()

    #: The CLI surface stays closed, and this is the assertion that keeps the
    #: two facts in agreement — a refusing function beside a live flag would be
    #: a worse state than either.
    from vetoworld.cli import build_parser
    stage = next(a for a in build_parser()._subparsers._group_actions[0]
                 .choices["run"]._actions if a.dest == "stage")
    assert "rider" not in stage.choices, (
        "the rider flag has been wired up while rider_cells() refuses; "
        "resolve #106 deliberately or not at all")

    #: The constants stay: they are inside the round's hashed payload, and the
    #: record of a probe designed and never served is worth keeping.
    assert R.RIDER_COHORT and R.RIDER_READING
    assert "rider" in R.payload()


def test_the_SEED_BLOCKS_are_disjoint_from_every_committed_cell():
    from seahaven.eden._shared import corpus as C
    #: **Round 16's own cells are now burned**, so this must ask the question it
    #: was always asking — does the block collide with anything served BEFORE
    #: this round — rather than colliding with itself.
    burned = set()
    for path, cell in C.iter_cells():
        got = C.parse_cell_name(path.name)
        if got and got["round"] == "16":
            continue
        burned |= {r["seed"] for r in cell.get("runs", []) if "seed" in r}
    grid = set(range(R.SEED0, R.SEED0 + max(R.EPISODES_A1, R.EPISODES_A0)))
    rider = set(range(R.RIDER_SEED0, R.RIDER_SEED0 + R.RIDER_EPISODES))
    assert not grid & rider, "the rider must not reuse the grid's block"
    assert not (grid | rider) & burned
