"""Round 21 — the second column, and the attestation that makes it readable.

Written before a single cell was served. The round measures seven models on
DeepInfra through the HuggingFace router, and the thing that makes those cells
worth anything is that each one records **who actually answered**, taken from
the wire rather than from what was requested.
"""

from __future__ import annotations

import pytest

from seahaven.eden import round15 as R15
from seahaven.eden import round20 as R20
from seahaven.eden import round21 as R
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import identity as ID


def test_the_PIN_RECOMPUTES_and_the_round_is_OPEN():
    assert R.current_hash() == R.PINNED_ROUND21_HASH
    R.assert_pinned()


def test_the_COHORT_IS_EXACTLY_WHAT_TOGETHER_WOULD_NOT_SERVE():
    """Seven models, and every one of them is in round 20's EXCLUDED for a
    stated reason. A model that Together *could* serve has no business here:
    it would be measured on a second provider for no reason, and the register
    would gain a mixed-provider row it did not need."""
    assert set(R.COHORT) == set(R20.EXCLUDED)
    assert len(R.COHORT) == 7
    assert set(R.COHORT) <= set(R15.COHORT)

    #: Six commercial, one technical — the distinction round 20 recorded.
    tiered = [m for m, w in R20.EXCLUDED.items() if "non-serverless" in w]
    assert len(tiered) == 6
    assert "moonshotai/Kimi-K3" in R.COHORT


def test_PRICES_ARE_DEEPINFRAS_not_Togethers():
    """The same model at a different provider is a different line item.
    Carrying round 15's Together prices would mis-state the spend, and the
    programme has already had one round bill at prices nothing was served at."""
    assert R.COHORT["moonshotai/Kimi-K3"] == (2.85, 14.25)
    assert R15.COHORT["moonshotai/Kimi-K3"] == (1.00, 3.00)
    differ = [m for m in R.COHORT if R.COHORT[m] != R15.COHORT[m]]
    assert differ, "no price differs from Together's — check they were re-read"


def test_the_PROVIDER_IS_PINNED_IN_THE_WIRE_ID_not_left_to_auto():
    """`auto` picks the fastest available provider per request. A round whose
    provider varies request to request is not one measurement."""
    assert R.PROVIDER == "deepinfra"
    assert R.served_id("moonshotai/Kimi-K3") == "moonshotai/Kimi-K3:deepinfra"
    for m in R.COHORT:
        assert R.served_id(m).endswith(R.PROVIDER_SUFFIX)


def test_BARE_MODEL_strips_the_routing_suffix_and_nothing_else():
    """**The single place that strips, asserted on its edges.**

    The router echoes `org/model` for a request of `org/model:provider`, so a
    raw `requested == served` comparison fails on every routed cell. The
    tempting inline fix is a `startswith`, which would be a second and looser
    identity rule beside the strict one — how the first one rotted.
    """
    assert ID.bare_model("zai-org/GLM-5:deepinfra") == "zai-org/GLM-5"
    assert ID.bare_model("zai-org/GLM-5") == "zai-org/GLM-5"
    assert ID.bare_model("gpt-4o") == "gpt-4o"
    assert ID.bare_model(None) is None

    #: It must not eat a colon that is part of the org or an earlier segment.
    assert ID.bare_model("weird:org/model") == "weird:org/model"
    #: And two different models must not collapse onto one another.
    assert ID.bare_model("a/b:x") != ID.bare_model("a/c:x")


def test_THE_RUNNER_REFUSES_A_CELL_WITH_NO_PROVIDER_ATTESTATION():
    """**The guard that makes this column readable, asserted on the source.**

    A round that pins a provider must record one that ANSWERED, not one that
    was asked for. Round 21's whole value rests on that: without it the cells
    would assert DeepInfra on the strength of the request, which is the exact
    shape of #113 — a claim about who served, resting on intent.
    """
    import inspect

    from vetoworld.commands import run
    src = inspect.getsource(run)
    assert "NO PROVIDER ATTESTATION" in src
    assert "SERVED BY THE WRONG PROVIDER" in src
    assert "x-inference-provider" in src
    assert "ID.bare_model(resolved) != ID.bare_model(model)" in src
    assert '"served_provider"' in src and '"pinned_provider"' in src

    #: The endpoint must actually capture the header, or the guard above can
    #: only ever see None and would refuse every routed cell.
    import seahaven.fidelity.endpoint as EP
    esrc = inspect.getsource(EP)
    assert "last_provider" in esrc
    assert "x-inference-provider" in esrc


def test_THE_ROUTER_HOST_IS_CATALOGUED_so_verification_is_not_skipped():
    """`catalogued` was `"together" in base_url or "openai.com" in base_url`.
    A second provider would have fallen through to RECORD-AND-PIN silently —
    skipping catalogue verification without saying so."""
    from vetoworld.backends.base import EndpointSpec

    spec = EndpointSpec(name="hf", base_url=R.BASE_URL, key_env="HF_TOKEN",
                        model="zai-org/GLM-5:deepinfra")
    assert spec.catalogued, "the router publishes availability; use it"
    assert "router.huggingface.co" in EndpointSpec.CATALOGUED_HOSTS


def test_the_SEED_BLOCKS_are_fresh_and_disjoint():
    burned = C.burned_seeds()
    grid = set(range(R.SEED0, R.SEED0 + max(R.EPISODES_A1, R.EPISODES_A0)))
    comp = set(range(R.COMP_SEED0, R.COMP_SEED0 + R.COMP_EPISODES))
    assert not grid & burned and not comp & burned
    assert not grid & comp
    assert R.SEED0 != R20.SEED0 and R.COMP_SEED0 != R20.COMP_SEED0
    from seahaven.eden import probe as PB
    assert max(grid | comp) < PB.SEED_BLOCK[0]


def test_BOTH_RULES_ARE_INSIDE_THE_PAYLOAD_not_only_in_the_docstring():
    """A rule that lives in prose is a rule that travels with the reader's
    memory. These are hashed, so they travel with the cells."""
    payload = R.payload()
    assert R.PROVENANCE_RULE[:60] in payload
    assert R.KIMI_K3_DIAGNOSIS[:60] in payload
    assert R.CANNOT_SETTLE[:40] in payload
    assert R.STANDING[:30] in payload
    assert "deviations" in payload and "SUPPRESSION NOT HONOURED" in payload
    assert "provider" in payload and "deepinfra" in payload


def test_MINIMAX_IS_ADMITTED_WITH_ITS_DEVIATION_STATED():
    """**The Terra precedent: recorded, admitted, and carried.**

    It ignores `enable_thinking=False` and emits reasoning that every other
    cell in the corpus suppresses. A flagged row beats a permanent blank — but
    only if the flag says what differs, so a reader can discount it without
    reconstructing why.
    """
    assert "MiniMaxAI/MiniMax-M2.7" in R.COHORT
    dev = R.DEVIATIONS["MiniMaxAI/MiniMax-M2.7"]
    assert "SUPPRESSION NOT HONOURED" in dev
    assert "BILLED" in dev
    assert "discount" in dev
    #: Only the models that actually deviate carry one.
    assert set(R.DEVIATIONS) == {"MiniMaxAI/MiniMax-M2.7"}


def test_the_ROUND_SAYS_IT_SETTLES_NOTHING_ABOUT_TOGETHER():
    """Registered before serving, because after serving is when a
    cross-provider comparison becomes tempting."""
    assert "anything about Together" in R.CANNOT_SETTLE
    assert "provider AND occasion" in R.CANNOT_SETTLE
    assert "ONE-OFF" in R.STANDING
    assert "not the decision" in R.STANDING
    assert "MAY NOT be compared" in R.PROVENANCE_RULE
