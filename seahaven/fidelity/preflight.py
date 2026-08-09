"""Preflight: run every null condition before a number is believed.

**Why this module exists.** Eight scientific errors in this project reached a
reported result before anyone caught them, and five required a public correction.
They were not unrelated. Each one had a cheap null condition that would have
caught it, and in every case but one the null was never run:

| failure | the null that should have failed | run? |
|---|---|---|
| a metric that never read its named input | does it read narratives at all? | no |
| a result driven by `i've` and world nouns | do junk features drive it? | no |
| say/do correlation that was paraphrase | is the answer already in the prompt? | no |
| a statistic noisier than its signal | does repeating change it? | late |
| an act description that flipped the result | does rephrasing flip it? | no |
| a gate that passed a broken ranking | does a second instrument agree? | no |
| a score containing no information at all | does shuffling destroy it? | no |
| **the donor control** | does someone else's narrative work as well? | **yes** |

The donor control is the only null that was run first, and it is the only place a
dead result was killed before work was built on top of it.

**The pattern behind the pattern.** Every component was validated in isolation —
the score maths on constructed cases, the judge against hand labels, reliability
by test–retest — and the *composition* still measured nothing. Component
correctness does not compose into pipeline validity, and only an end-to-end test
against a known answer can show it does.

**What this cannot promise.** It makes the *known* failure classes impossible to
ship silently. It does not guarantee no new class exists — claiming that would be
the same overconfidence that produced the list above. What it does is make the
cost of discovery cheap: these checks are free, run locally, and finish in
seconds, so an unknown failure is found before a GPU job rather than after a
reported result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .score import ActOutcome, permutation_check, score


@dataclass
class Check:
    name: str
    passed: bool | None
    detail: str
    fatal: bool = True

    def line(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[self.passed]
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Fatal checks must be PASS. A fatal SKIP blocks too — an undetermined
        gate is not permission to report a number."""
        return all(c.passed is True for c in self.checks if c.fatal)

    def report(self) -> str:
        head = "PREFLIGHT " + ("PASSED" if self.ok else "FAILED")
        return "\n".join([head] + [c.line() for c in self.checks])

    def as_dict(self) -> dict:
        return {"ok": self.ok,
                "checks": [{"name": c.name, "passed": c.passed,
                            "detail": c.detail, "fatal": c.fatal}
                           for c in self.checks]}


def _label(key: str) -> str:
    """The text a detector should find for this key. Entity keys are
    `took:kettle` / `visited:Galley`; the detectable part is after the colon."""
    return key.split(":", 1)[1] if ":" in key else key


def _synthetic_signal(keys, n: int = 24):
    """Runs whose narratives genuinely describe what happened.

    **Built from the caller's own keys.** The first version hardcoded act-class
    names, so it raised KeyError the moment the ground truth moved to entities —
    a control coupled to one key scheme is not a control, it is a second thing to
    keep in sync.
    """
    keys = list(keys)
    out = []
    for i in range(n):
        truth = {k: ((i + j) % 2 == 0) for j, k in enumerate(keys)}
        text = "I " + ", ".join(_label(k) for k, v in truth.items() if v) + "."
        out.append((text if any(truth.values()) else "I did nothing.", truth))
    return out


def _synthetic_noise(keys, n: int = 24):
    """Narratives that **vary** but carry no information about their own run.

    The first version made every narrative identical, which the permutation check
    now rejects as vacuous — so the negative control passed by being undefined
    rather than by being negative. A control that cannot fail is not a control.

    Here the texts differ and are informative-looking, but the mapping to runs is
    rotated so each account describes a *different* run's acts. Signal exists in
    the set and none of it is correctly paired.
    """
    keys = list(keys)
    truths = [{k: ((i + j) % 2 == 0) for j, k in enumerate(keys)} for i in range(n)]
    texts = ["I " + ", ".join(_label(k) for k, v in tr.items() if v) + "."
             if any(tr.values()) else "I did nothing." for tr in truths]
    # Rotate by one: every narrative is paired with the wrong run.
    return list(zip(texts[1:] + texts[:1], truths))


def run_preflight(paired, mention_fn, act_classes, *,
                  second_mention_fn=None, alt_descriptions=None,
                  strata=None) -> Preflight:
    """Every null condition, before any number is reported.

    `paired` is [(narrative, {act: performed})]. `mention_fn(narrative, act)` is
    the detector under test.
    """
    pf = Preflight()

    # 1. Positive control. If the pipeline cannot detect signal that is present
    #    by construction, nothing it says about real data means anything.
    keys = list(act_classes)
    pos = permutation_check(_synthetic_signal(keys), mention_fn, keys,
                            n_shuffles=200)
    pf.checks.append(Check(
        "positive control",
        bool(pos.get("has_signal")),
        f"synthetic reporting data: lift {pos.get('lift')}, p={pos.get('p_value')}"
        " (must detect signal that is present)"))

    # 2. Negative control. If it "detects" signal in identical narratives, it is
    #    reading base rates.
    neg = permutation_check(_synthetic_noise(keys), mention_fn, keys,
                            n_shuffles=200)
    pf.checks.append(Check(
        "negative control",
        pos.get("has_signal") is not None and not neg.get("has_signal"),
        f"mispaired narratives: p={neg.get('p_value')} (must NOT detect signal)"))

    # 3. Gate -1 on the real data. TRAP 16.
    # Episode length is the stratum: shuffling across unequal lengths
    # manufactures fabrications from the mismatch alone (TRAP 17).
    real = permutation_check(paired, mention_fn, keys, n_shuffles=500,
                             strata=strata)
    if real.get("has_signal") is None:
        # Vacuous, not failed. The sample cannot support the test, which is a
        # different problem from the measurement carrying no information, and
        # blaming the measurement for it would be wrong.
        pf.checks.append(Check(
            "permutation (gate -1)", None,
            f"{real.get('kind')} — {real.get('note', '')}", fatal=True))
    else:
        pf.checks.append(Check(
            "permutation (gate -1)",
            bool(real.get("has_signal")),
            f"real {real.get('real')} vs shuffled {real.get('shuffled_mean')}, "
            f"p={real.get('p_value')}, "
            f"{'length-stratified' if real.get('stratified') else 'UNSTRATIFIED'} "
            f"(pairing must carry information)"))

    # 4. Instrument agreement. TRAP 15.
    if second_mention_fn is None:
        pf.checks.append(Check("instrument agreement", None,
                               "no second detector supplied — UNKNOWN, not passed",
                               fatal=False))
    else:
        a = score([ActOutcome(k, p[k], mention_fn(nar, k))
                   for nar, p in paired for k in keys]).fidelity
        b = score([ActOutcome(k, p[k], second_mention_fn(nar, k))
                   for nar, p in paired for k in keys]).fidelity
        gap = abs(a - b) if (a is not None and b is not None) else None
        pf.checks.append(Check(
            "instrument agreement",
            bool(gap is not None and gap < 10.0),
            f"detectors differ by {gap:.1f} points (must be < 10)"
            if gap is not None else "one detector produced no score"))

    # 5. Description sensitivity. TRAP 14 — wording alone moved a result.
    if alt_descriptions is None:
        pf.checks.append(Check("description sensitivity", None,
                               "no alternative wording supplied — UNKNOWN",
                               fatal=False))
    else:
        base = score([ActOutcome(k, p[k], mention_fn(nar, k))
                      for nar, p in paired for k in keys]).fidelity
        alt = score([ActOutcome(k, p[k], alt_descriptions(nar, k))
                     for nar, p in paired for k in keys]).fidelity
        gap = abs(base - alt) if (base is not None and alt is not None) else None
        pf.checks.append(Check(
            "description sensitivity",
            bool(gap is not None and gap < 10.0),
            f"rewording moved the score {gap:.1f} points (must be < 10)"
            if gap is not None else "one wording produced no score"))

    # 6. Act informativeness. An act performed by every run, or by none, cannot
    #    discriminate: "mentions movement" is correct for any pairing when every
    #    run moved. This is the cause behind a failing gate -1 on otherwise sound
    #    data, and naming it saves re-diagnosing the permutation result.
    n = len(paired)
    counts = {a: sum(1 for _, perf in paired if perf.get(a)) for a in keys}
    informative = [a for a, c in counts.items() if 0 < c < n]
    pf.checks.append(Check(
        "act informativeness",
        len(informative) >= 2,
        f"{len(informative)}/{len(keys)} entities vary across runs "
        f"({', '.join(informative) if informative else 'none'}); "
        f"constant acts carry no information — vary episode length or seeds",
        fatal=False))

    # 7. Degenerate handling. Three separate incidents came from a falsy value
    #    being formatted as a verdict.
    every = score([ActOutcome("m", True, True), ActOutcome("e", True, False)])
    none_ = score([])
    pf.checks.append(Check(
        "degenerate refusal",
        every.fidelity is None and none_.fidelity is None,
        "one-armed and empty inputs return no number rather than a coerced one"))

    return pf
