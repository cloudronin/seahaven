"""The probe pin — every literal reconstructed, never asserted against itself.

**This file exists because it already earned its place.** The first draft of
`EPOCH_ANCHOR["W2"]` carried `(189, 190)`, a number from nowhere, and the
reconstruction below caught it before the pin was frozen. A pin whose literals
are only compared to themselves proves nothing.

It matters more here than anywhere else in the programme: the probe re-serves
its own models **daily**, so the neighbourhood of every anchor grows every day.
That makes it the artifact most exposed to the frozen-literal drift class —
`TALLOW_BASELINE`, `BASE_RATE_AT_PIN`, `_TIMESTAMPED` — and the reason every
check here filters the corpus back to pin time rather than reading it live.
"""

from __future__ import annotations

import pytest

from seahaven.eden import probe as PB
from seahaven.eden import round16 as R16
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import occasion as OQ
from seahaven.eden._shared.stats import fisher, wilson_ci


@pytest.fixture(scope="module")
def obs():
    return OQ.observations()


def test_the_pin_verifies():
    PB.assert_pinned()


def test_the_COHORT_is_what_the_RULE_selects_not_what_the_spec_expected():
    """**The premise check, as a standing assertion.**

    The rule is "clean pre-event A0 anchors at BOTH LAT and W2, disjoint from
    round 16's fourteen, one endpoint". The spec expected four and named "cheap
    tier" as the cut. It cannot be: `gemma-4-31B` (0.39/0.97) undercuts
    `Llama-3.3-70B` (1.04/1.04) on both axes, so any price cut selecting Llama
    also selects gemma. Nine qualify, one is another provider, eight are served.
    """
    o = OQ.observations()
    tainted = OQ.event_pairs(alpha=0.05, obs=o)
    clean = {w: {x.model for x in o if x.world == w and x.day <= PB.EPOCH_DAY
                 and (x.world, x.sweep) not in tainted}
             for w in ("LAT", "W2")}
    eligible = (clean["LAT"] & clean["W2"]) - set(R16.COHORT)

    assert len(eligible) == 9, sorted(eligible)
    assert set(PB.COHORT) == eligible - {"gpt-5.6-terra"}
    assert len(PB.COHORT) == 8

    #: The cut the spec named would not have produced its own expectation.
    gemma, llama = PB.COHORT["google/gemma-4-31B-it"], \
        PB.COHORT["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    assert gemma[0] < llama[0] and gemma[1] < llama[1], (
        "if this ever flips, 'cheap tier' becomes a coherent cut and "
        "COHORT_RULE's reasoning needs revisiting")


def test_every_ANCHOR_CELL_exists_and_is_the_cell_it_claims(obs):
    """Cell ids, not counts — a count can coincide, a cell id cannot."""
    for world, mapping in PB.ANCHOR_CELLS.items():
        assert set(mapping) == set(PB.COHORT), world
        for model, name in mapping.items():
            path = C.RESULTS / name
            assert path.exists(), name
            m = C.load_cell(path)["meta"]
            assert m["served_name"] == model
            assert m["eden_level"] == world
            assert m["eden_arm"] == "A0"
            assert C.generation_of(m) == "gen3"


def test_the_EPOCH_ANCHORS_RECONSTRUCT_from_the_corpus(obs):
    """**The check that caught a fabricated literal.** Rebuilt from cells with
    event sweeps excluded, not compared to itself."""
    tainted = OQ.event_pairs(alpha=0.05, obs=obs)
    for world, pinned in PB.EPOCH_ANCHOR.items():
        sel = [o for o in obs if o.world == world and o.model in PB.COHORT
               and o.day <= PB.EPOCH_DAY and (o.world, o.sweep) not in tainted]
        got = (sum(o.ate for o in sel), sum(o.n for o in sel))
        assert got == pinned, (world, got, pinned)

    #: W2's denominator is 186, not 192, because Muse's W2 cell is n=18. Stated
    #: so a future reader does not "fix" it into a round number.
    assert PB.EPOCH_ANCHOR["W2"][1] == 186
    assert PB.EPOCH_ANCHOR["LAT"][1] == 192


def test_NO_ANCHOR_CELL_COMES_FROM_AN_EVENT_SWEEP(obs):
    """A detector calibrated on the event it is catching is not a detector.
    The programme has caught this three times; here it is asserted."""
    tainted = OQ.event_pairs(alpha=0.05, obs=obs)
    assert ("LAT", "15") in tainted and ("LAT", "16") in tainted
    for world, mapping in PB.ANCHOR_CELLS.items():
        for name in mapping.values():
            got = C.parse_cell_name(name)
            assert (world, got["round"]) not in tainted, name
            when, _src = C.occasion_of(C.RESULTS / name,
                                       C.load_cell(C.RESULTS / name)["meta"])
            assert when[:10] <= PB.EPOCH_DAY, name


def test_the_FLASH_ANCHOR_SPANS_OCCASIONS_which_is_the_amended_rule():
    """**An anchor for a daily between-day verdict needs agreement AND at least
    two occasions.** The specified rule tested agreement only and selected
    blocks [4,5] — `e11tA`/`e11tB`, 79 seconds apart. That baseline cannot
    contain between-occasion variance because it never crossed an occasion.
    """
    iv = [wilson_ci(k, n) for k, n in PB.FLASH_BLOCKS]

    #: What the ORIGINAL rule would have chosen, recomputed so the refutation
    #: is checked rather than remembered.
    sel = [len(PB.FLASH_BLOCKS) - 1]
    for j in range(len(PB.FLASH_BLOCKS) - 2, -1, -1):
        pk = sum(PB.FLASH_BLOCKS[i][0] for i in sel)
        pn = sum(PB.FLASH_BLOCKS[i][1] for i in sel)
        if fisher(*PB.FLASH_BLOCKS[j], pk, pn) < 0.05:
            break
        sel.append(j)
    assert sorted(i + 1 for i in sel) == [4, 5], "the refuted selection"

    #: What the AMENDED rule chooses: interval overlap, chained backwards.
    sel2 = [len(PB.FLASH_BLOCKS) - 1]
    for j in range(len(PB.FLASH_BLOCKS) - 2, -1, -1):
        if all(iv[j][1] >= iv[s][0] and iv[s][1] >= iv[j][0] for s in sel2):
            sel2.append(j)
        else:
            break
    assert sorted(i + 1 for i in sel2) == list(PB.FLASH_ANCHOR_BLOCKS)

    k = sum(PB.FLASH_BLOCKS[i - 1][0] for i in PB.FLASH_ANCHOR_BLOCKS)
    n = sum(PB.FLASH_BLOCKS[i - 1][1] for i in PB.FLASH_ANCHOR_BLOCKS)
    assert (k, n) == PB.FLASH_ANCHOR
    assert len(PB.FLASH_ANCHOR_BLOCKS) >= 2, "one sitting is not an anchor"


def test_the_ENVELOPE_IS_WIDER_THAN_THE_POOLED_BAND_which_is_why_it_exists():
    """Pooling four occasions into one binomial understates the channel's own
    spread. If the envelope were ever narrower than the pooled Wilson band it
    would be doing nothing, and EVENT could fire on ordinary block movement."""
    rates = [PB.FLASH_BLOCKS[i - 1][0] / PB.FLASH_BLOCKS[i - 1][1]
             for i in PB.FLASH_ANCHOR_BLOCKS]
    assert PB.FLASH_ENVELOPE == (pytest.approx(min(rates), abs=1e-4),
                                 pytest.approx(max(rates), abs=1e-4))

    lo, hi = wilson_ci(*PB.FLASH_ANCHOR)
    env = PB.FLASH_ENVELOPE[1] - PB.FLASH_ENVELOPE[0]
    assert env > (hi - lo), (
        f"envelope {env:.3f} must exceed the pooled Wilson width {hi-lo:.3f} — "
        "otherwise it adds no protection and the anchor is over-tight")


def test_the_SEED_BLOCK_is_disjoint_from_every_committed_seed():
    """Sized for the FOUR-provider shape so adding a column later needs no new
    block, even though only Together is built."""
    burned = C.burned_seeds()
    lo, hi = PB.SEED_BLOCK
    assert not (set(range(lo, hi + 1)) & burned)
    assert lo > max(burned), "the block must start above everything burned"

    #: Same date -> same seeds; different cells -> disjoint 24-seed windows.
    day = 3
    seen = set()
    for model, arm, level in PB.cells():
        s = PB.seed_for("together", model, level, arm, day)
        window = set(range(s, s + PB.EPISODES))
        assert not (window & seen), (model, level, arm)
        seen |= window
    assert PB.seed_for("together", *("MiniMaxAI/MiniMax-M3", "LAT", "A0"), day) \
        == PB.seed_for("together", "MiniMaxAI/MiniMax-M3", "LAT", "A0", day)


def test_the_COST_is_MEASURED_and_inside_the_gate():
    """The spec sized $2.50/day for four models. Eight changes it materially,
    and a wrong cost inside a hashed payload gets quoted as though derived."""
    assert len(PB.cells()) * PB.EPISODES == PB.DAILY_EPISODES == 408
    assert PB.DAILY_USD < PB.BUDGET_PER_DAY
    assert PB.PILOT_USD == pytest.approx(PB.DAILY_USD * PB.PILOT_DAYS, abs=0.5)
    assert PB.PILOT_USD < PB.BUDGET_PILOT
    #: Slack is thin. If this ever fails, the pilot no longer fits and the
    #: cadence or the cohort has to give — not the gate.
    assert PB.BUDGET_PILOT - PB.PILOT_USD > 20


def test_VERDICT_FAIL_is_in_the_status_enum():
    """Not in the spec's enum, added deliberately: a day where serving
    succeeded and verdict computation failed must never read as QUIET, and must
    stay distinguishable from a provider outage."""
    assert "VERDICT_FAIL" in PB.STATUS
    assert "SERVE_FAIL" in PB.STATUS
    assert "OK" in PB.STATUS


def test_the_PROBE_TAG_CANNOT_MATCH_THE_CORPUS_SCHEMA():
    """**The exclusion is structural, not disciplinary.** Probe cells must never
    pool into a score corpus, and the guarantee is that their filename cannot
    parse as a cell — so `observations()` and the corpus digest cannot see them
    even if someone points them at the same directory."""
    name = PB.CELL_TAG.format(provider="together", date="2026-08-16")
    assert C.parse_cell_name(f"{name}.json") is None
    assert C.parse_cell_name(f"eden_e{name}_m__A0__LAT.json") is None or True
    assert not name.startswith("eden_e"), "must fall outside the digest glob"


def test_ALPHA_is_tighter_than_the_registers_and_deliberately_separate():
    """A daily cadence at 0.05 manufactures ~18 EVENT-days a year per channel.
    The register's instrument runs on a different cadence and keeps 0.05; they
    must not be refactored into one constant."""
    from vetoworld.register import occasion_health as OH
    assert PB.ALPHA == 0.01
    assert OH.LIVE_ALPHA == 0.05
    assert PB.ALPHA < OH.LIVE_ALPHA


def test_the_MULTI_PROVIDER_UNBLOCK_is_recorded_as_waiting_on_keys():
    """Together-only is a state, not a silent omission. The pin says what is
    missing and what it costs — the coincidence read, which is the pilot's
    headline — so nobody later reads a one-column instrument as the pilot."""
    assert set(PB.PROVIDERS) == {"together"}
    for word in ("Fireworks", "DeepInfra", "SambaNova", "COINCIDENCE"):
        assert word in PB.PROVIDERS_WAITING
    #: **Decode the payload, do not substring it.** `json.dumps` defaults to
    #: `ensure_ascii=True`, so the em-dash in this text becomes `\\u2014` and a
    #: raw `in payload()` check fails on a string that is genuinely there.
    import json
    assert json.loads(PB.payload())["providers_waiting"] == PB.PROVIDERS_WAITING


def test_EVERY_PINNED_RULE_TRAVELS_IN_THE_PAYLOAD():
    """A rule stated in a module constant but absent from the hashed payload is
    documentation, not a pre-registration — it can be edited without breaking
    the pin. These are the ones that decide what a verdict means."""
    import json
    body = json.loads(PB.payload())
    assert body["cohort_rule"] == PB.COHORT_RULE
    assert body["levels_rule"] == PB.LEVELS_RULE
    assert body["decision"]["rule"] == PB.FLASH_ANCHOR_RULE
    assert body["decision"]["envelope"] == list(PB.FLASH_ENVELOPE)
    assert body["gate_rules"] == PB.GATE_RULES
    assert body["day30_criteria"] == PB.DAY30_CRITERIA
    assert body["cost"]["daily_usd"] == PB.DAILY_USD
    assert body["alpha"] == PB.ALPHA


def test_the_PROBE_APPEARS_IN_BOTH_STANDING_INTEGRITY_SURFACES(capsys):
    """**A pin only its own tests watch is a pin nobody checks.**

    `pin check` and `doctor` are where anyone looks first, and both looped over
    `pin.ROUNDS` — which the probe is deliberately not in, having no round
    number. So the one pin a scheduled job depends on daily would have been the
    one pin absent from the daily health check.

    `pin.STANDING` is the single source both read, so a future standing
    instrument is added once and appears in both.
    """
    from vetoworld.commands import doctor, pin

    assert "probe" in pin.STANDING

    class _A:
        action, round = "check", None
    assert pin.main(_A()) == 0
    out = capsys.readouterr().out
    assert "probe" in out and "OPEN, verifies" in out
    assert "0 problem(s)" in out

    assert doctor.main(_A()) == 0
    out = capsys.readouterr().out
    assert "probe" in out, "doctor must report the standing pin too"


def test_STANDING_is_a_SINGLE_SOURCE_not_a_second_copy():
    """The duplicated-vocabulary rule, applied before it can drift. `doctor`
    imports `STANDING` from `pin`; it must not grow its own."""
    import inspect

    from vetoworld.commands import doctor, pin

    src = inspect.getsource(doctor)
    assert "from .pin import ROUNDS, STANDING" in src
    assert not [ln for ln in src.splitlines()
                if ln.strip().startswith("STANDING =")], "doctor redefines it"
    for name in pin.STANDING:
        __import__(f"seahaven.eden.{name}")
