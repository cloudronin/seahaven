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
    The programme has caught this three times; here it is asserted.

    **The events it was excluding no longer exist (#113), and the exclusion is
    still the right rule.** ("LAT","15") and ("LAT","16") were EVENT because
    fourteen filenames were compared against each other. Corrected, the only
    sweep that still trips the live alpha is ("LAT","18"), and that one is
    pseudo-replication — eight cells, one model, one seed set.

    What matters for the pin is unchanged and is what this asserts: **no anchor
    cell comes from any sweep the channel flags**, whatever that set turns out
    to be. Writing the sweep names into the assertion would have made this test
    a record of one day's verdicts instead of a property.
    """
    tainted = OQ.event_pairs(alpha=0.05, obs=obs)
    assert ("LAT", "15") not in tainted, "retracted; see #113"
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
    """**Both columns, both measured.** The spec sized $2.50/day for four
    models; eight changed it, and the second provider changed it again. A wrong
    cost inside a hashed payload gets quoted later as though it were derived.

    The DeepInfra figure comes from round 21's committed billing, not a rate
    card — the plan assumed ~$2.50 a serving day and the truth is $6.17 without
    Kimi-K3 and $13.59 with him.
    """
    assert len(PB.cells()) * PB.EPISODES == PB.DAILY_EPISODES == 408
    assert PB.DAILY_USD < PB.BUDGET_PER_DAY

    projected = (PB.DAILY_USD * PB.PILOT_DAYS
                 + PB.DEEPINFRA_DAY_USD * PB.DEEPINFRA_SERVING_DAYS)
    assert PB.PILOT_USD == pytest.approx(projected, abs=1.0)
    assert PB.PILOT_USD < PB.BUDGET_PILOT
    #: Slack is thin. If this fails, the CADENCE or the COHORT gives — not the
    #: gate. A gate raised to fit a projection is not a gate.
    assert PB.BUDGET_PILOT - PB.PILOT_USD > 20

    #: **The excluded model is excluded for a stated, checkable amount.**
    #: Keeping Kimi-K3 would breach even the amended gate, and that arithmetic
    #: is the whole justification for dropping him.
    with_kimi = projected + PB.DEEPINFRA_KIMI_WOULD_ADD * PB.DEEPINFRA_SERVING_DAYS
    assert with_kimi > PB.BUDGET_PILOT, (
        "Kimi-K3 now fits inside the gate, so the cost reason for excluding "
        "him from the daily fleet has evaporated — put him back or restate why")
    assert "moonshotai/Kimi-K3" not in PB.DEEPINFRA_COHORT


def test_the_BUDGET_AMENDMENT_IS_PREREGISTERED_not_discovered():
    """**$280 -> $350, stated before day two.**

    A budget raised after the spend cannot be told apart from a budget raised
    to excuse it. So the amendment carries its date, its reason, and the three
    figures it is answerable to — and day one's superseded pin is kept, because
    the 17 cells served under it cite it in their `probe_pin`.
    """
    assert PB.BUDGET_PILOT == 350.00
    a = PB.BUDGET_AMENDMENT
    assert "280" in a and "350" in a
    assert "BEFORE DAY TWO" in a
    assert "Kimi-K3" in a
    assert "cannot be told apart" in a

    #: The superseded pin stays recomputable, like a retired round's.
    assert PB.SUPERSEDED_PINS
    old = "956f9059871c87961495d4c861c367c7578c9821f4dc0bf709e851931e845471"
    assert old in PB.SUPERSEDED_PINS
    assert old != PB.PINNED_PROBE_HASH

    #: And day one's cells really do cite it, or the boundary is not legible.
    import json
    from pathlib import Path
    served = sorted(Path("results").glob("probe-together-*.json"))
    if served:
        pins = {json.loads(p.read_text())["meta"].get("probe_pin")
                for p in served}
        assert pins <= set(PB.SUPERSEDED_PINS) | {PB.PINNED_PROBE_HASH}, (
            f"a served probe cell cites a pin that is neither current nor "
            f"recorded as superseded: {pins}")


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
    #: **DeepInfra no longer waits.** The HF router serves it on an HF_TOKEN,
    #: so the second column was taken without a DeepInfra account — weeks
    #: earlier than the plan assumed. Two errands remain.
    assert set(PB.PROVIDERS) == {"together", "deepinfra"}
    for word in ("Fireworks", "SambaNova", "COINCIDENCE"):
        assert word in PB.PROVIDERS_WAITING
    assert "DeepInfra NO LONGER WAITS" in PB.PROVIDERS_WAITING
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


# --- the second column, and what adding it nearly broke ---------------------

def test_ADDING_A_PROVIDER_MUST_NOT_MOVE_AN_EXISTING_COLUMNS_SEEDS():
    """**Caught before the second column was added, not after.**

    `seed_for` derived the provider offset from `sorted(PROVIDERS).index()`.
    "deepinfra" sorts BEFORE "together", so adding it would have moved
    Together's offset 0 -> 1 and shifted every seed the column derives: day one
    served `seed0=100864` for MiniMax-M3/LAT and would have re-derived 101272.
    The served cells would have become unreproducible from the code that made
    them, silently, with nothing failing.

    An alphabetical index looks stable and is not — it is a function of the
    whole set, so every member depends on every other. Same shape as [TRAP] 38,
    in a place where it corrupts provenance rather than a verdict.
    """
    assert PB.PROVIDER_SLOT["together"] == 0, (
        "Together's slot moved; every seed it has ever derived moved with it")
    #: Appending must never renumber an existing column.
    assert PB.PROVIDER_SLOT == {"together": 0, "deepinfra": 1,
                                "fireworks": 2, "sambanova": 3}

    #: The derivation must use the frozen slot, not the sorted position.
    import inspect
    #: **Executable lines only.** The comment beside the fix NAMES the old form
    #: so a reader knows what was wrong, so a whole-source substring check would
    #: fire on the explanation rather than on the code.
    src = inspect.getsource(PB.seed_for)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "PROVIDER_SLOT[provider]" in code
    assert "sorted(PROVIDERS).index" not in code


def test_DAY_ONES_SEEDS_STILL_DERIVE_TO_WHAT_WAS_SERVED():
    """The property the slot freeze exists for, asserted against the committed
    cells rather than against the constant."""
    import json
    from pathlib import Path

    for p in sorted(Path("results").glob("probe-together-2026-08-16_*.json")):
        m = json.loads(p.read_text())["meta"]
        got = PB.seed_for("together", m["served_name"], m["eden_level"],
                          m["eden_arm"], 1)
        assert got == m["seed0"], (
            f"{p.name}: served seed0={m['seed0']} but the code now derives "
            f"{got}. The cell is no longer reproducible from the pin.")


def test_the_TWO_COLUMNS_TAKE_DISJOINT_SEEDS():
    #: **Each column over its OWN grid.** This iterated `cells()` for both,
    #: which is Together's grid — and once `seed_for` learned to refuse a model
    #: outside the column's cohort, the test refused too. It was asking a
    #: question that cannot be asked: what seed does DeepInfra give a model it
    #: does not serve.
    from seahaven.eden import probe as P
    a = {P.seed_for("together", m, lv, ar, d)
         for m, ar, lv in P.cells("together") for d in range(1, 31)}
    b = {P.seed_for("deepinfra", m, lv, ar, d)
         for m, ar, lv in P.cells("deepinfra") for d in range(1, 31)}
    assert not (a & b), "the columns share seeds; they are not independent"
    lo, hi = P.SEED_BLOCK
    assert max(a | b) + P.EPISODES - 1 <= hi, (
        "a 30-day two-column pilot runs past the reserved block")


def test_the_ROUTED_COLUMN_IS_DEFINED_BY_THE_HEADER_not_the_request():
    """A router can reroute. What makes the column trustworthy is that the
    runner refuses a cell whose `x-inference-provider` does not match the pin —
    #113's lesson applied before the fact rather than after it."""
    assert "x-inference-provider" in PB.ROUTER_ATTESTATION
    assert "REFUSES" in PB.ROUTER_ATTESTATION
    assert PB.PROVIDERS["deepinfra"]["model_suffix"] == ":deepinfra"
    assert PB.PROVIDERS["deepinfra"]["key_env"] == "HF_TOKEN"

    import inspect
    from vetoworld.commands import run
    src = inspect.getsource(run)
    assert "SERVED BY THE WRONG PROVIDER" in src
    assert "NO PROVIDER ATTESTATION" in src


def test_PHASE_E_IS_RESOLVED_WITH_FACTS_not_left_as_INVESTIGATE():
    """The plan left HF Jobs cron as `[INVESTIGATE]` and said to resolve it
    before building. These are the facts it was resolved with."""
    f = PB.SCHEDULE_FACTS
    assert "create_scheduled_job" in f
    assert "CRON" in f and "@daily" in f
    assert "POSITIVE CREDIT BALANCE" in f, "the tier question must be answered"
    assert "THIRTY MINUTES" in f, (
        "the default timeout is the trap and must be stated: a two-column day "
        "exceeds it and would be killed mid-serve")
    assert "secrets=" in f

    s = PB.SCHEDULE_SHAPE
    assert "ONE scheduled job" in s
    assert "BY CONSTRUCTION" in s, (
        "the reason for sequential-in-one-window is that coincidence needs a "
        "shared occasion; without that the choice is arbitrary")
    assert "hung provider" in s, "the accepted cost must be named"


def test_the_FORK_IS_VOID_and_no_occasion_can_reopen_it():
    """#108's precedent exactly. The gate returns VOID-SUBJECT before it looks
    at any verdict, because occasion was never what was missing."""
    assert "VOID-SUBJECT" in PB.GATE_RULES["fork-reopen"]
    assert "#113" in PB.FORK_VOIDED
    assert "no subject" in PB.FORK_VOIDED
    assert "NOT reopenable" in PB.FORK_VOIDED
    assert "round 21" in PB.FORK_VOIDED, (
        "the residual question's actual answer must be pointed at, or voiding "
        "reads as abandoning it")

    #: The live surface must exit NONZERO and must not consult the day.
    import inspect
    from vetoworld.commands import occasion as OC
    src = inspect.getsource(OC)
    assert "VOID_SUBJECT = 3" in src
    body = src.split('if purpose == "fork-reopen":')[1].split("return")[0]
    assert "QUIET" not in body.split("#:")[0] or "VOID-SUBJECT" in body, (
        "the fork branch still reads the day's verdict; voiding after "
        "consulting the occasion implies the occasion mattered")


def test_ROUND_19s_SEALED_TEXT_IS_NOT_EDITED():
    """The seal is a retired pin — restored, never re-frozen. It stands as the
    record of what was believed when it was written, and the live gate is where
    the correction goes."""
    from seahaven.eden import round19 as R19

    assert "have never been read" in R19.SEALED_ROUND16_FORK["what"], (
        "the sealed text was edited to match the void; it should stand as "
        "written and the LIVE gate should carry the correction")
    assert "0.708" in R19.SEALED_ROUND16_FORK["what"]
    #: The seal still says a QUIET sweep reopens it. That is now FALSE, and it
    #: is exactly why the text stays in the retired pin rather than being
    #: corrected in place: it records what was believed, not what is true.
    assert "QUIET" in R19.SEALED_ROUND16_FORK["reopens_on"]


def test_the_UTC_HOUR_IS_PINNED_and_the_reason_is_the_confound():
    """15:00 UTC is 08:00 Pacific, so a failure surfaces at breakfast. That is
    the operational reason and NOT why it is in the payload.

    A fixed hour exists to remove the hour-of-day confound: a fleet serving at
    15:00 one day and 03:00 the next folds diurnal load variation into every
    between-day verdict, and the instrument's whole claim is that a between-day
    difference is about the endpoint. Consistency is the requirement; 15:00 is
    the choice.
    """
    assert PB.SCHEDULE_UTC_HOUR == 15
    assert PB.SCHEDULE_CRON == "0 15 * * *"
    assert str(PB.SCHEDULE_UTC_HOUR) in PB.SCHEDULE_CRON

    import json
    p = json.loads(PB.payload())
    assert p["schedule_utc_hour"] == 15, "the hour must travel in the pin"
    assert p["schedule_cron"] == PB.SCHEDULE_CRON


def test_the_CADENCE_IS_A_FUNCTION_not_a_sentence():
    """'DeepInfra on Mon/Wed/Fri' written only in prose is a cadence someone
    has to remember. `serves_today` is one the job obeys."""
    #: Monday=0 ... Sunday=6
    assert [d for d in range(7) if PB.serves_today("deepinfra", d)] == [0, 2, 4]
    assert [d for d in range(7) if PB.serves_today("together", d)] == list(range(7))

    #: 13 DeepInfra serving days in 30, which is what the budget assumed.
    import datetime as _dt
    start = _dt.date(2026, 8, 17)
    n = sum(PB.serves_today("deepinfra", (start + _dt.timedelta(days=i)).weekday())
            for i in range(PB.PILOT_DAYS))
    assert n == PB.DEEPINFRA_SERVING_DAYS, (
        f"the cadence yields {n} serving days but the budget assumed "
        f"{PB.DEEPINFRA_SERVING_DAYS} — the projection and the schedule "
        "disagree, and the projection is what the gate was amended against")


def test_the_JOB_SPEC_DOES_NOT_INHERIT_THE_30_MINUTE_DEFAULT():
    """The default kills a two-column day mid-serve and would present as a
    PARTIAL day of unknown cause. The timeout must be explicit."""
    job = PB.SCHEDULE_JOB
    assert "timeout" in job and job["timeout"] != "30m"
    assert job["flavor"] == "cpu-basic"
    assert set(job["secrets"]) == {"TOGETHER_API_KEY", "HF_TOKEN"}
    #: Order matters: the columns run sequentially in one window, and the
    #: anchored column goes first so a DeepInfra outage cannot delay it.
    assert job["columns_in_order"][0] == "together"


def test_EVERY_SUPERSEDED_PIN_SAYS_WHETHER_IT_GOVERNED_CELLS():
    """A superseded hash a reader finds in git history is a question. Each one
    answers it: day one's governed 17 cells, the other governed none."""
    assert len(PB.SUPERSEDED_PINS) >= 2
    for h, why in PB.SUPERSEDED_PINS.items():
        assert len(h) == 64
        assert h != PB.PINNED_PROBE_HASH
        assert ("SERVED under it" in why) or ("NO CELLS WERE SERVED" in why), (
            f"{h[:12]} does not say whether any cell was served under it")


# --- defining a rule is half the work: three that were not wired -----------

def test_EACH_COLUMN_SERVES_ITS_OWN_COHORT():
    """`cells(provider)` took the provider and IGNORED it, so the DeepInfra
    column would have served Together's eight models through the router —
    models DeepInfra may not host at all — while `DEEPINFRA_COHORT` sat defined
    and unread. Found when a dry run printed 408 episodes for a 288-episode
    grid."""
    tg, di = PB.cells("together"), PB.cells("deepinfra")
    assert {m for m, _a, _l in tg} - {PB.DECISION_MODEL} == set(PB.COHORT)
    assert {m for m, _a, _l in di} == set(PB.DEEPINFRA_COHORT)
    assert len(tg) * PB.EPISODES == 408
    assert len(di) * PB.EPISODES == 288

    #: The decision channel is Together's. A DeepInfra Flash cell would be a
    #: different served artifact judged against a Together anchor.
    assert PB.DECISION_MODEL in {m for m, _a, _l in tg}
    assert PB.DECISION_MODEL not in {m for m, _a, _l in di}


def test_SEED_FOR_INDEXES_WITHIN_THE_PROVIDERS_OWN_GRID():
    """It indexed into `cells()` — Together's grid — so the first DeepInfra
    cell raised `ValueError: not in list`. The scheduled job would have died on
    its first MWF day."""
    assert PB.seed_for("deepinfra", "zai-org/GLM-5", "LAT", "A0", 2) > 0

    #: And a model outside the column's cohort is refused, not silently given
    #: another column's seed.
    with pytest.raises(SystemExit, match="not in deepinfra's grid"):
        PB.seed_for("deepinfra", "google/gemma-4-31B-it", "LAT", "A0", 2)


def test_THE_CADENCE_GATE_IS_CALLED_not_merely_defined():
    """`serves_today` existed and nothing called it. One job runs daily and each
    column decides its own days; without the check the MWF column serves every
    day — $185 over the pilot instead of $80, breaching the gate amended for it
    hours earlier."""
    import inspect

    from vetoworld.commands import probe as CMD
    src = inspect.getsource(CMD._daily)
    assert "serves_today" in src, (
        "the cadence is defined in the pin and not consulted by the verb — a "
        "rule the machinery does not read is a rule it does not follow")
    #: And it must gate BEFORE the grid is priced or served.
    assert src.index("serves_today") < src.index("todo = [")


def test_THE_TWO_COLUMNS_STILL_TAKE_DISJOINT_SEEDS_AFTER_THE_GRID_SPLIT():
    a = {PB.seed_for("together", m, lv, ar, d)
         for m, ar, lv in PB.cells("together") for d in range(1, 31)}
    b = {PB.seed_for("deepinfra", m, lv, ar, d)
         for m, ar, lv in PB.cells("deepinfra") for d in range(1, 31)}
    assert not (a & b)
    lo, hi = PB.SEED_BLOCK
    assert max(a | b) + PB.EPISODES - 1 <= hi


def test_THE_PIN_DID_NOT_MOVE_because_no_RULE_moved():
    """**The wiring fixes changed no pinned constant, and that is the point.**

    The cohorts, cadences, slots and gates were all correct and hashed before
    this build. What was wrong was that three of them had no caller. A pin
    records what was decided; it cannot record whether the code obeys it, which
    is what these tests are for.
    """
    assert PB.PINNED_PROBE_HASH.startswith("b380cc61")
    PB.assert_pinned()
