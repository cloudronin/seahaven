"""Round 16 — the reference channel as a gate, and the re-serve it pays for.

Two things a pin cannot check about itself: that its frozen literals still match
the corpus they were read off, and that the rule they encode does what the
docstring says when run against real cells.
"""

from __future__ import annotations

import pytest

from seahaven.eden import round16 as R
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


def test_the_FROZEN_ANCHORS_match_the_corpus_and_EXCLUDE_the_event():
    """**The literal is the detector's reference; if it drifts, so does every
    later verdict.** Recomputed here from committed cells rather than trusted.

    The exclusion is the load-bearing half: eight of the fourteen have no LAT A0
    except the event day, so a baseline built from "all earlier days" would
    contain the very drop the re-serve is meant to test against.
    """
    obs = OQ.observations()
    tainted = OQ.event_pairs(alpha=R.OCCASION_ALPHA, obs=obs)
    assert ("LAT", "15") in tainted, "the known event must be excluded"

    got = {}
    for model in R.COHORT:
        cells = [o for o in obs if o.world == "LAT" and o.model == model
                 and (o.world, o.sweep) not in tainted]
        if cells:
            got[model] = (sum(o.ate for o in cells), sum(o.n for o in cells))
    assert got == R.CLEAN_LAT_ANCHOR
    assert len(got) == 6, "six of the fourteen carry a clean LAT anchor"

    k = sum(a for a, _n in got.values())
    n = sum(b for _a, b in got.values())
    assert (k, n) == R.ANCHOR_POOLED

    #: No frozen anchor may contain a cell from the event day.
    for model in R.CLEAN_LAT_ANCHOR:
        days = {o.day for o in obs if o.world == "LAT" and o.model == model
                and (o.world, o.sweep) not in tainted}
        assert EVENT_DAY not in days, model


def test_the_EVENT_DAY_LITERALS_are_the_ones_the_FORK_turns_on():
    """0.708 appears twice — the eight unanchored models on the event day, and
    the clean weak-tier baseline at W2. That coincidence is the whole reason
    the fork exists, so both numbers are frozen rather than remembered."""
    obs = OQ.observations()
    ev = [o for o in obs if o.world == "LAT" and o.sweep == "15"]
    assert (sum(o.ate for o in ev), sum(o.n for o in ev)) == R.EVENT_DAY_ALL_14

    un = [o for o in ev if o.model not in R.CLEAN_LAT_ANCHOR]
    got = (sum(o.ate for o in un), sum(o.n for o in un))
    assert got == R.EVENT_DAY_UNANCHORED_8
    assert got[0] / got[1] == pytest.approx(0.708, abs=0.001)


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
    post = {"16", "17", "18", "19"}
    obs = [o for o in OQ.observations() if o.sweep not in post]
    audit = OQ.audit(alpha=R.OCCASION_ALPHA, obs=obs)
    assert R.BASE_RATE_AT_PIN == {
        "pairs": len(audit),
        "anchored": sum(v.tested for v in audit),
        "events": sum(v.verdict == "EVENT" for v in audit),
        "no_anchor": sum(not v.tested for v in audit),
    }


def test_THE_ROUND_FAILED_ITS_OWN_SELF_CERTIFICATION():
    """**The result of the round, asserted rather than remembered.**

    SELF_CERTIFY pins the order: the six anchored models are read first, and
    EVENT means the serving day is itself anomalous and the fourteen are not
    interpreted. It fired. The anchored six went 0.9766 -> 0.7292 against their
    clean baseline, which is *below* the event day that prompted the re-serve,
    so the channel had not recovered — it had fallen further.
    """
    v = OQ.verdict_for("LAT", "16", alpha=R.OCCASION_ALPHA)
    assert v.verdict == "EVENT", "the round did not certify; that is the finding"
    assert len(v.returning) == 6
    assert v.prior == R.ANCHOR_POOLED, "judged against the frozen clean anchor"
    now, prior = v.rates()
    assert prior == pytest.approx(0.9766, abs=0.001)
    assert now == pytest.approx(0.7292, abs=0.001)

    #: **Monotone across three days**, which one anomalous day cannot produce.
    obs = OQ.observations()
    anchored = set(R.CLEAN_LAT_ANCHOR)
    rates = []
    for day in ("2026-08-13", "2026-08-14", "2026-08-15"):
        sel = [o for o in obs if o.world == "LAT" and o.day == day
               and o.model in anchored]
        k, n = sum(o.ate for o in sel), sum(o.n for o in sel)
        rates.append(k / n)
    assert rates[0] > rates[1] > rates[2], rates


def test_the_GATE_VETOES_FOURTEEN_and_every_one_carries_its_REASON():
    """**The UNSCORED path is exercised by fourteen real models, not a fixture.**

    An earlier draft of this claim said zero models reached it and the branch
    needed a synthetic cell — that was computed under a 2-world admission rule,
    not the 3/3 rule this round pins.
    """
    gated, ungated = CO.veto_hold(), CO.veto_hold(gate=False)
    assert len(ungated.admitted) == 23
    assert len(gated.admitted) == 9

    by_event = {m: why for m, why in gated.refused.items()
                if any("EVENT" in w for w in why)}
    assert len(by_event) == 14
    #: **The date moved and the assertion should not have named it.** Round 16
    #: made e16/2026-08-15 the canonical LAT cell for these models, so the
    #: printed reason now cites the re-serve rather than 08-14. What matters is
    #: that LAT is vetoed for an occasion EVENT, not which occasion.
    for model, why in by_event.items():
        assert any("LAT" in w and "occasion EVENT" in w for w in why), model

    #: The one pre-existing refusal is incompleteness, not occasion — the two
    #: reasons must stay distinguishable or the gate takes credit for a gap.
    other = set(gated.refused) - set(by_event)
    assert len(other) == 1
    assert all("no cell" in w for m in other for w in gated.refused[m])


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


def test_the_RIDER_selection_is_a_RULE_ALREADY_COMPUTED_not_a_fresh_choice():
    """Its two non-control models are exactly the ones the channel flagged. A
    hand-picked probe cohort is a choice made after seeing the data."""
    audit = OQ.audit(alpha=R.OCCASION_ALPHA)
    event = next(v for v in audit if v.verdict == "EVENT")
    flagged = {m for m, _n, _p, _pv in event.flags}
    assert flagged == set(R.RIDER_COHORT) - {"deepcogito/cogito-v2-1-671b"}
    assert len(R.RIDER_COHORT) == 3, "two flagged plus one control"


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
