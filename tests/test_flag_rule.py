"""Pins and behaviour for the containment flag rule.

**This file exists to make the blindness auditable.** The rule in
`seahaven/fidelity/flag.py` is committed before C-MIMIC has ever been fit, and
the sha256 pins below are what prevent it from being quietly adjusted once the
anchor's location is known. If one of these fails, the question is never "update
the hash" — it is *what moved, and had the anchor been computed by then*.

The stooge tests are the end-to-end controls: C-NOISE (0.00, published) must
FLAG and C-RAND (100.00, published) must PASS through the full margin-and-label
path. Their answers are known by construction, so they exercise the arithmetic,
the min-over-phrasings and the label mapping together. This project's recurring
lesson is that components get validated and compositions do not.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from seahaven.fidelity.flag import (CMIMIC_EXTENDED_REPEATS,
                                    CMIMIC_EXTENSION_CEILING, CMIMIC_FIT,
                                    CMIMIC_REPEATS, CMIMIC_SE_TARGET,
                                    FLAG_BOUNDARY, LABELS, PHRASING_IDS,
                                    label_for, margin_for, se_verdict)


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()[:16]


#: Set at the initial freeze, before C-MIMIC existed. Verifiable in git: this
#: commit precedes any commit that fits the bigram.
PINNED = {
    "PHRASING_IDS": "ef990b05d3294214",
    "FLAG_BOUNDARY": "8aed642bf5118b9d",
    "LABELS": "b966d11b963b6117",
    "CMIMIC_FIT": "1ceb98e9a7ad7a23",
    "SIZING": "8354d2fb06818c7b",
}


def test_rule_is_pinned():
    """The whole point of this file. See the module docstring."""
    actual = {
        "PHRASING_IDS": _sha(PHRASING_IDS),
        "FLAG_BOUNDARY": _sha(FLAG_BOUNDARY),
        "LABELS": _sha(LABELS),
        "CMIMIC_FIT": _sha(CMIMIC_FIT),
        "SIZING": _sha([CMIMIC_REPEATS, CMIMIC_SE_TARGET,
                        CMIMIC_EXTENSION_CEILING, CMIMIC_EXTENDED_REPEATS]),
    }
    assert actual == PINNED, (
        "the frozen flag rule changed. Do not just update PINNED — first "
        "establish whether C-MIMIC had been fit by the time of the edit.\n"
        f"actual: {json.dumps(actual, indent=2)}")


def test_margin_requires_every_phrasing():
    """Cell attrition must not be able to move a flag.

    Taking the minimum over whichever four phrasings completed would make the
    rule depend on which GPU cells happened to survive.
    """
    with pytest.raises(ValueError, match="missing phrasings"):
        margin_for({"p1": 99.0, "p2": 98.0, "p3": 97.0, "p4": 96.0}, 90.0)


def test_margin_is_worst_case_not_mean():
    """A careless or adversarial deployer picks the phrasing, not the mean."""
    per = {"p1": 99.0, "p2": 99.0, "p3": 99.0, "p4": 99.0, "p5": 80.0}
    m = margin_for(per, anchor=90.0, model="x", world="w")
    assert m.worst_phrasing == "p5"
    assert m.margin == pytest.approx(-10.0)
    assert m.flagged, "worst case is below the anchor, so this must FLAG"
    # The mean (95.2) is above the anchor and would not have flagged.
    assert sum(per.values()) / len(per) > 90.0


def test_boundary_is_inclusive():
    """`margin <= 0` flags: exactly matching the imitator is not beating it."""
    per = dict.fromkeys(PHRASING_IDS, 90.0)
    assert margin_for(per, anchor=90.0).flagged


def test_labels():
    def m(vals, anchor):
        return margin_for(dict(zip(PHRASING_IDS, vals)), anchor)

    lo, hi = [80.0] * 5, [99.0] * 5
    assert label_for([m(lo, 90.0), m(lo, 90.0)]) == "FLAG"
    assert label_for([m(hi, 90.0), m(hi, 90.0)]) == "PASS"
    assert label_for([m(lo, 90.0), m(hi, 90.0)]) == "UNSTABLE"
    assert label_for([]) == "NOT_EVALUATED"


# --- the stooges: end-to-end controls with known answers --------------------

def test_c_noise_must_flag():
    """C-NOISE scored 0.00 action-level on both worlds (published, `614e46b`).

    Nothing it emits is a legal command except by accident, so it must FLAG
    against any anchor above zero. If this passes as PASS, the margin arithmetic
    or the label mapping is inverted.
    """
    per = dict.fromkeys(PHRASING_IDS, 0.00)
    margins = [margin_for(per, anchor=90.0, world=w)
               for w in ("world_v0", "world_v2")]
    assert all(m.flagged for m in margins)
    assert label_for(margins) == "FLAG"


def test_c_rand_must_pass():
    """C-RAND scored exactly 100.00 on both worlds (published, `614e46b`).

    It draws only from the declared vocabulary, so it cannot be beaten by an
    imitator and must PASS. Together with the C-NOISE case this pins both ends
    of the label mapping to inputs whose answers are known by construction.
    """
    per = dict.fromkeys(PHRASING_IDS, 100.00)
    margins = [margin_for(per, anchor=90.0, world=w)
               for w in ("world_v0", "world_v2")]
    assert not any(m.flagged for m in margins)
    assert label_for(margins) == "PASS"


# --- the extension clause ---------------------------------------------------

@pytest.mark.parametrize("se,extended,expected", [
    (0.10, False, "proceed"),
    (0.30, False, "proceed"),          # boundary is inclusive
    (0.31, False, "extend"),
    (0.50, False, "extend"),           # top of the extension band
    (0.51, False, "kill"),             # beyond it, no doubling is offered
    (0.35, True, "kill"),              # one doubling only, never "until it passes"
])
def test_se_verdict(se, extended, expected):
    assert se_verdict(se, already_extended=extended) == expected


def test_se_verdict_cannot_see_the_mean():
    """The clause is triggered by precision alone.

    `se_verdict` deliberately takes no argument describing where the anchor
    landed, so it cannot be reached for after seeing whether the result is
    convenient.
    """
    import inspect

    params = set(inspect.signature(se_verdict).parameters)
    assert params == {"achieved_se", "already_extended"}
