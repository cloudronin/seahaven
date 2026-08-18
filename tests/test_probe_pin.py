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


def test_the_COST_is_MEASURED_where_it_can_be_and_LABELLED_where_it_cannot():
    """**One column measured, one estimated, and the difference is named.**

    Together's $7.86 is measured from committed billing on these exact models
    at these exact worlds. DeepInfra's is NOT, and cannot be yet: the matched
    pair's prices are `(0, 0)` until day one measures them, so the only figure
    available is that grid priced off TOGETHER's rate card. That is an estimate
    wearing another column's prices, and the constant says so in its name.

    **This test previously asserted "both columns, both measured"** and
    multiplied by `DEEPINFRA_DAY_USD` — a name that made a round-21 billing
    figure look like this cohort's measured cost. Different models entirely; it
    could never have carried over. A wrong cost inside a hashed payload gets
    quoted later as though it were derived.
    """
    assert len(PB.cells()) * PB.EPISODES == PB.DAILY_EPISODES == 408
    assert PB.DAILY_USD < PB.BUDGET_PER_DAY

    projected = (PB.DAILY_USD * PB.PILOT_DAYS
                 + PB.DEEPINFRA_DAY_USD_TOGETHER_PRICED
                 * PB.DEEPINFRA_SERVING_DAYS)
    assert PB.PILOT_USD == pytest.approx(projected, abs=1.0)
    assert PB.PILOT_USD < PB.BUDGET_PILOT
    #: Slack is thin. If this fails, the CADENCE or the COHORT gives — not the
    #: gate. A gate raised to fit a projection is not a gate.
    assert PB.BUDGET_PILOT - PB.PILOT_USD > 20

    #: **The unqualified name must not come back.** `DEEPINFRA_DAY_USD` read as
    #: this cohort's measured cost and was another round's; the qualifier is
    #: the entire point of the rename.
    assert not hasattr(PB, "DEEPINFRA_DAY_USD"), (
        "the unqualified name is back — a Together-priced estimate must not be "
        "spellable as though it were this column's measurement")

    #: **The estimate is not allowed to become a measurement by typing.** The
    #: pair carries (0, 0) deliberately and nothing may price a cell from it.
    assert set(PB.DEEPINFRA_COHORT.values()) == {(0.0, 0.0)}, (
        "a DeepInfra price is non-zero. If day one MEASURED it, amend "
        "PILOT_USD and this test together at a pin boundary; if it was typed "
        "off a rate card, that is precisely what the (0, 0) exists to prevent")

    #: Kimi-K3 is still absent, but no longer for a cost reason: he is absent
    #: because he is not in the exact-variant intersection of the two catalogs.
    #: The arithmetic that used to justify his exclusion retired with the
    #: six-model cohort it belonged to — and an assertion kept past the reason
    #: for it is how a test starts protecting the wrong thing.
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
    #: **DeepInfra no longer waits.** The HF router served it on an HF_TOKEN,
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

    #: **The matched pair's selection rule, which was documentation until
    #: 2026-08-17.** Its load-bearing clause is "SELECTED BEFORE ANY
    #: CROSS-COLUMN TRACE EXISTED" — a claim whose whole value is that it
    #: cannot be edited once the traces arrive. Stated in a module constant and
    #: absent from the payload, it could have been, and no pin would have
    #: broken. The same argument covers the EXHIBITS, which fix what the pilot
    #: claims before the data can suggest which claim is easiest to support.
    assert body["matched_pair_rule"] == PB.MATCHED_PAIR_RULE
    assert "BEFORE ANY CROSS-COLUMN TRACE" in body["matched_pair_rule"]
    assert body["exhibits"] == list(PB.EXHIBITS)
    assert body["round21_column_stands"] == PB.ROUND21_COLUMN_STANDS

    #: And the one constant standing between a DeepInfra verdict and Together's
    #: epoch. Unhashed, the fix for blocker 1 could be undone silently.
    assert body["anchor_owner"] == PB.ANCHOR_OWNER


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


def test_the_COLUMN_IS_DEFINED_BY_EVIDENCE_ON_THE_PATH_THAT_SERVES_IT():
    """**This test used to grep the wrong module, and passed for it.**

    It was named `..._DEFINED_BY_THE_HEADER_not_the_request` and asserted that
    `SERVED BY THE WRONG PROVIDER` appears in `vetoworld.commands.run`. It does
    — and the probe column is served by `probe._daily`, which never imports a
    round and never reached any of those guards. The test protected a path that
    was never at risk while the path that was went unguarded, which is the
    "defining the rule is half the work" family living inside a witness.

    The column now serves DIRECT, so the evidence is the endpoint host rather
    than a routing header, and the guard is asserted where it has to run.
    """
    assert PB.PROVIDERS["deepinfra"]["base_url"] == \
        "https://api.deepinfra.com/v1/openai"
    assert PB.PROVIDERS["deepinfra"]["key_env"] == "DEEPINFRA_API_KEY"
    assert "model_suffix" not in PB.PROVIDERS["deepinfra"], (
        "dead config that LOOKS live: the suffix is applied only in the round "
        "path via R.served_id and was never read here")

    #: The rule in force names its own weakness rather than glossing it.
    assert "who ANSWERED" in PB.TRANSPORT_RULE
    assert "who was ASKED" in PB.TRANSPORT_RULE
    assert "WEAKER THAN THE HEADER" in PB.TRANSPORT_RULE
    #: And the router rule is KEPT, because round 21's cells answer to it.
    assert "x-inference-provider" in PB.ROUTER_ATTESTATION
    assert "REFUSES" in PB.ROUTER_ATTESTATION

    #: **The guard is asserted on the PROBE path this time.**
    import inspect

    from vetoworld.commands import probe as CMD
    src = inspect.getsource(CMD._daily)
    assert "REFUSING TO SERVE" in src
    assert "served_provider_for" in src
    assert "catalogued" in src
    #: It must refuse BEFORE anything is served and paid for.
    assert src.index("REFUSING TO SERVE") < src.index("for model, arm, level in todo")


def test_THE_PAIR_IS_AN_INTERSECTION_and_NEAR_MISSES_ARE_REFUSED():
    """**The pair is what the rule selects, not what was convenient.**

    Servable on BOTH columns at EXACT variant. The refusal that matters is
    `nvidia/nemotron-3-ultra-550b-a55b` against DeepInfra's
    `NVIDIA-Nemotron-3-Ultra-550B-A55B` — a different string, therefore a
    different model, which is round 4's generation-qualifier lesson and #113's
    wire-id lesson in one. Matching case-insensitively here would have put a
    model in the pair on the strength of its spelling.
    """
    assert set(PB.DEEPINFRA_COHORT) == {
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo"}
    #: Both are drawn from Together's own fleet, which is what makes them a
    #: PAIR rather than two unrelated columns.
    assert set(PB.DEEPINFRA_COHORT) <= set(PB.COHORT) | {PB.DECISION_MODEL}

    #: The near-miss is refused and stays refused.
    assert "nvidia/nemotron-3-ultra-550b-a55b" in PB.COHORT
    assert "nvidia/nemotron-3-ultra-550b-a55b" not in PB.DEEPINFRA_COHORT
    assert "NVIDIA-Nemotron-3-Ultra-550B-A55B" not in PB.DEEPINFRA_COHORT
    assert "nemotron" in PB.MATCHED_PAIR_RULE or "near-miss" in \
        PB.MATCHED_PAIR_RULE.lower(), "the refusal must be recorded, not just made"

    #: One volatile probe and one damped control. Without the control a
    #: divergence cannot be attributed to the provider rather than the method.
    assert PB.DECISION_MODEL in PB.DEEPINFRA_COHORT
    assert "control" in PB.MATCHED_PAIR_RULE
    assert "BEFORE ANY CROSS-COLUMN TRACE" in PB.MATCHED_PAIR_RULE


def test_A_DIRECT_ENDPOINT_CELL_PARTITIONS_TO_ITS_OWN_COLUMN():
    """**Blocker 5.** The probe path wrote `provider` from INTENT with no
    `served_provider` and no `base_url`, so `identity.provider_of` fell through
    to `DIRECT_PROVIDER` and a DeepInfra probe cell partitioned as TOGETHER —
    pooling two providers in the one place round 21's Rule 1 forbids it."""
    from seahaven.eden._shared import identity as ID

    di = PB.PROVIDERS["deepinfra"]["base_url"]
    assert PB.served_provider_for(di) == "deepinfra"
    assert ID.provider_of({"base_url": di}) == "deepinfra"
    assert ID.provider_of({"served_provider": "deepinfra"}) == "deepinfra"

    #: The two maps must agree, or the serving path and the corpus reader
    #: disagree about which column a cell belongs to.
    assert PB.HOST_PROVIDER == ID.HOST_PROVIDER

    #: An unknown host is refused outright rather than attributed by intent.
    assert PB.served_provider_for("https://api.example.com/v1") is None
    #: And it must not quietly become Together, which is what falling through
    #: to DIRECT_PROVIDER did.
    assert ID.provider_of({"base_url": "https://api.example.com/v1"}) \
        == "api.example.com"


def test_THE_DEEPINFRA_HOST_IS_CATALOGUED_or_verification_is_SKIPPED_SILENTLY():
    """**Blocker 7.** Absent from `CATALOGUED_HOSTS`, `catalogued` is False,
    `list_models()` returns None, `resolve_model()` hands back the requested
    string verbatim, and the probe's `resolved != model` check compares a
    string to itself and can never fail."""
    from vetoworld.backends.base import EndpointSpec

    assert "api.deepinfra.com" in EndpointSpec.CATALOGUED_HOSTS
    for name, cfg in PB.PROVIDERS.items():
        spec = EndpointSpec(name=name, base_url=cfg["base_url"],
                            key_env=cfg["key_env"], model="x")
        assert spec.catalogued, (
            f"{name} serves from an uncatalogued host, so its model strings "
            "are never checked against a published record")


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
    """A cadence written only in prose is one someone has to remember.
    `serves_today` is one the job obeys.

    **The cadence changed with the re-scope, and the reason is the read.**
    Mon/Wed/Fri was sized for a six-model $6.17/day column. The matched pair
    costs a third of that, and DIFFERENTIAL STABILITY is a day-over-day
    variance comparison: matched cadence is what makes the two traces
    comparable, and same-day coincidence is only observable on days both
    columns serve. 13 of 30 would have thrown away more than half the primary
    read to save about $30.
    """
    #: Monday=0 ... Sunday=6. Both columns daily — which is now the point, and
    #: is the one thing a reader is most likely to assume is still MWF.
    assert [d for d in range(7)
            if PB.serves_today("deepinfra", d)] == list(range(7))
    assert [d for d in range(7)
            if PB.serves_today("together", d)] == list(range(7))

    #: The MWF cadence is KEPT as a definition, because a later column may want
    #: it — but nothing rides it, and that has to be visible rather than
    #: assumed. A cadence with no subscriber that still looks live is how the
    #: prose and the function drift apart again.
    assert PB.CADENCES["mon_wed_fri"] == (0, 2, 4)
    assert not [pv for pv, spec in PB.PROVIDERS.items()
                if spec["cadence"] == "mon_wed_fri"], (
        "a column is on mon_wed_fri again — if that is deliberate, the budget "
        "projection and DEEPINFRA_SERVING_DAYS move with it")

    #: 30 DeepInfra serving days in 30, which is what the budget assumed.
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
    assert set(job["secrets"]) == {"TOGETHER_API_KEY", "HF_TOKEN",
                                   "DEEPINFRA_API_KEY"}, (
        "the direct second column needs its own key, and HF_TOKEN is not "
        "vestigial: it pushes the day's rows and, since read_rows went live, "
        "reads the prior ones back for the rolling window and the earn pool")
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
    assert len(di) * PB.EPISODES == 120

    #: **The decision channel is on BOTH columns now — that is the matched pair
    #: working, and it is the change this test used to forbid.** What must not
    #: travel with the cell is Together's BASELINE: DeepInfra's Flash A1 gets no
    #: anchor and no envelope until it earns them, rather than inheriting the
    #: ones the pre-event corpus happened to produce on another provider.
    #:
    #: The old objection — "a DeepInfra Flash cell would be judged against a
    #: Together anchor" — was correct, and was an argument against the
    #: INHERITANCE, not against the cell. `epoch_for`/`envelope_for` removed the
    #: inheritance, so the cell can be served.
    dec = (PB.DECISION_MODEL, PB.DECISION_ARM, PB.DECISION_LEVEL)
    assert dec in tg and dec in di
    assert PB.epoch_for("together", "LAT", "A1") == PB.FLASH_ANCHOR
    assert PB.envelope_for("together", "LAT", "A1") == PB.FLASH_ENVELOPE
    assert PB.epoch_for("deepinfra", "LAT", "A1") is None, (
        "DeepInfra's decision cell resolved to an epoch it has not earned — "
        "ANCHOR_OWNER exists for exactly this, and a verdict computed here "
        "would Fisher-test DeepInfra's rate against Together's corpus")
    assert PB.envelope_for("deepinfra", "LAT", "A1") is None

    #: **Together keeps its decision cell by ANCHOR_OWNER, not by cohort
    #: membership.** Flash is the decision channel, not one of the anchored
    #: eight, so `DECISION_MODEL in COHORT` is False — gating the append on
    #: membership alone drops Together from 17 cells to 16 and makes day one
    #: underivable from the code that served it. Caught while writing the fix.
    assert PB.DECISION_MODEL not in PB.COHORT
    assert PB.DECISION_MODEL in PB.DEEPINFRA_COHORT
    assert PB.ANCHOR_OWNER == "together"


def test_SEED_FOR_INDEXES_WITHIN_THE_PROVIDERS_OWN_GRID():
    """It indexed into `cells()` — Together's grid — so the first DeepInfra
    cell raised `ValueError: not in list`. The scheduled job would have died on
    its first serving day."""
    assert PB.seed_for(
        "deepinfra", "deepseek-ai/DeepSeek-V4-Flash-0731", "LAT", "A0", 2) > 0
    #: The decision cell too — it is in DeepInfra's grid now, and a seed for it
    #: must come from DeepInfra's own range.
    assert PB.seed_for(
        "deepinfra", PB.DECISION_MODEL, "LAT", "A1", 2) > 0

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


def test_THE_STRIDE_REFUSES_A_COLUMN_IT_CANNOT_HOLD():
    """**SEED_STRIDE says "across four providers" and holds two.**

    864 reserves 36 cell-slots a day; four columns at Together's 17-cell grid
    need 68. Slots 0 and 1 fit; slots 2 and 3 run past the stride into the NEXT
    day's range, starting with Together's own. Latent, because those columns
    have no keys — and silent when it fires, because nothing downstream would
    notice two days sharing seeds.

    **The guard refuses rather than widening.** Day one is day=1 and derives
    from `100000 + 1*864`; moving SEED_STRIDE would make the 17 committed cells
    unreproducible from the code that served them, which is exactly what
    freezing PROVIDER_SLOT was for.
    """
    #: The live columns are unaffected, and day one still derives.
    assert PB.seed_for("together", "MiniMaxAI/MiniMax-M3", "LAT", "A0", 1) == 100864
    assert PB.seed_for("deepinfra", PB.DECISION_MODEL, "LAT", "A1", 2) > 0

    #: A third column is refused at the seed, before it can serve a cell.
    with pytest.raises(SystemExit, match="SEED_STRIDE"):
        PB.seed_for("fireworks", "google/gemma-4-31B-it", "LAT", "A0", 1)

    #: And the arithmetic the refusal rests on, so a later reader can check it
    #: rather than take it.
    slots = PB.SEED_STRIDE // PB.EPISODES
    assert slots == 36
    assert len(PB.cells("together")) * len(PB.PROVIDER_SLOT) == 68 > slots


def test_THE_TWO_COLUMNS_STILL_TAKE_DISJOINT_SEEDS_AFTER_THE_GRID_SPLIT():
    a = {PB.seed_for("together", m, lv, ar, d)
         for m, ar, lv in PB.cells("together") for d in range(1, 31)}
    b = {PB.seed_for("deepinfra", m, lv, ar, d)
         for m, ar, lv in PB.cells("deepinfra") for d in range(1, 31)}
    assert not (a & b)
    lo, hi = PB.SEED_BLOCK
    assert max(a | b) + PB.EPISODES - 1 <= hi


def test_THE_PIN_MOVED_because_the_RULES_moved_and_the_OLD_ONE_IS_KEPT():
    """**This test asserted the opposite until 2026-08-17, and the inversion is
    itself the record.**

    Its previous name was `THE_PIN_DID_NOT_MOVE_because_no_RULE_moved`, and it
    was right at the time: the wiring fixes changed no pinned constant, because
    what was wrong was that three constants had no caller. A pin records what
    was decided; it cannot record whether the code obeys it, which is what the
    rest of this file is for.

    The matched-pair re-scope is the other case. Cohort, cadence, cost, serving
    days and grid all moved, so the pin MUST move — a re-scope that left the
    hash alone would prove the payload was not carrying the design.
    """
    assert PB.PINNED_PROBE_HASH.startswith("8bfb3da9")
    PB.assert_pinned()

    #: **The pin it replaced is kept, and says it governed nothing.** A reader
    #: who finds b380cc61 in git history — it is the head of the red-tree
    #: commit — needs to know which day's cells answer to it. None do.
    old = "b380cc610df78fbd684a482cdd2e528c4a3be7758f77ebfc31c00fa938e8a08f"
    assert old in PB.SUPERSEDED_PINS
    assert "NO CELLS WERE SERVED" in PB.SUPERSEDED_PINS[old]

    #: And the invariant that makes superseding safe at all: every served cell
    #: still cites a pin that is either current or recorded. Day one's 17 cells
    #: cite 956f9059 and must keep verifying against it forever.
    import json
    from pathlib import Path
    served = sorted(Path("results").glob("probe-*.json"))
    pins = {json.loads(pth.read_text())["meta"].get("probe_pin")
            for pth in served}
    assert pins <= set(PB.SUPERSEDED_PINS) | {PB.PINNED_PROBE_HASH}, (
        f"a served cell cites a pin that is neither current nor recorded as "
        f"superseded: {pins - set(PB.SUPERSEDED_PINS) - {PB.PINNED_PROBE_HASH}}")
