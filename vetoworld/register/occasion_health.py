"""`vworld emit occasion-health` — was the environment quiet on each sweep?

The A0 arm carries no treatment and sits near ceiling by design, so every sweep
already buys an environment sensor for free. This renders it.

**THE BASE RATE IS A COMPUTED ARTIFACT, NOT A PARAGRAPH.** The manuscript's
serving-stack section needs a number for "how often did we detect an event", and
a number typed into prose drifts from the corpus the moment another sweep lands.
`verify` recomputes this like every other figure.

**A DETECTOR THAT PUBLISHES A BASE RATE PUBLISHES ITS FALSE-ALARM EXPECTATION
BESIDE IT.** The historical pass runs one Fisher per anchored (sweep, world), so
at any alpha the expected count of spurious EVENTs is non-zero by construction,
and "we detected N events" without "we expected M false ones" is the alert
semantics that trains people to ignore alerts. So the two print on the same
screen, and the pass is corrected while live per-sweep monitoring is not: one
sweep is one test, a retrospective sweep of the whole record is many, and they
do not share an error budget.

There is no `--all` flag. The plan called for one, but the two modes differ by
**alpha** rather than by which rows are shown, and a flag that changes only the
row filter would have hidden the part that matters. Every pair is listed under
both alphas instead, so the correction's effect is visible rather than asserted.
"""

from __future__ import annotations

import textwrap

from seahaven.eden._shared import occasion as OQ

#: The per-sweep alpha, used for live monitoring of a single new sweep.
LIVE_ALPHA = 0.05


def _short(model: str) -> str:
    return model.split("/")[-1]


def _corrected(n_tests: int, alpha: float = LIVE_ALPHA) -> float:
    """Bonferroni over the tests the historical pass actually runs.

    NO-ANCHOR pairs run no test, so they do not enter the denominator — counting
    them would inflate the correction with comparisons that were never made.
    """
    return alpha / n_tests if n_tests else alpha


def occasion_health() -> int:
    obs = OQ.observations()
    live = OQ.audit(alpha=LIVE_ALPHA, obs=obs)
    n_tests = sum(v.tested for v in live)
    alpha_h = _corrected(n_tests)
    hist = OQ.audit(alpha=alpha_h, obs=obs)

    n_event = sum(v.verdict == "EVENT" for v in hist)
    n_quiet = sum(v.verdict == "QUIET" for v in hist)
    n_none = sum(not v.tested for v in hist)

    print("OCCASION HEALTH — the A0 reference channel, paired within model\n")
    print(f"  {len(hist)} (sweep, world) pairs on record")
    print(f"  {n_tests:>3} testable        {n_none:>3} NO-ANCHOR "
          "(nobody here was served at this world on an earlier day)")
    print(f"  {n_event:>3} EVENT           {n_quiet:>3} QUIET\n")

    print(f"  FALSE-ALARM EXPECTATION — {n_tests} test(s) x alpha "
          f"{LIVE_ALPHA} = {n_tests * LIVE_ALPHA:.2f} expected false EVENT(s)")
    print(f"  Historical pass alpha : {alpha_h:.5f}  (Bonferroni, "
          f"{LIVE_ALPHA} / {n_tests} test(s))")
    print(f"  Live per-sweep alpha  : {LIVE_ALPHA:.5f}  (uncorrected — one "
          "sweep is one test)")
    if [v.verdict for v in hist] == [v.verdict for v in live]:
        print("  The two agree on every pair at present.\n")
    else:
        moved = [(h.world, h.sweep, l.verdict, h.verdict)
                 for h, l in zip(hist, live) if h.verdict != l.verdict]
        print("  **THE CORRECTION CHANGES VERDICTS**: "
              + "; ".join(f"{w} e{s} {a} -> {b}" for w, s, a, b in moved) + "\n")

    print(f"  {'world':<6}{'sweep':<7}{'day':<12}{'verdict':<11}{'ret':>4}"
          f"{'new':>5}{'now':>18}{'prior':>18}{'p':>11}")
    for v in hist:
        now_k, now_n = v.now
        pri_k, pri_n = v.prior
        r_now, r_pri = v.rates()
        now = f"{r_now:.3f} ({now_k}/{now_n})"
        pri = "—" if not pri_n else f"{r_pri:.3f} ({pri_k}/{pri_n})"
        p = "—" if v.p is None else f"{v.p:.2e}"
        print(f"  {v.world:<6}e{v.sweep:<6}{v.day:<12}{v.verdict:<11}"
              f"{len(v.returning):>4}{len(v.newcomers):>5}{now:>18}{pri:>18}{p:>11}")

    for v in hist:
        if v.verdict != "EVENT":
            continue
        lo_n, hi_n = OQ.interval(*v.now)
        lo_p, hi_p = OQ.interval(*v.prior)
        r_now, r_pri = v.rates()
        print(f"\n  {v.world} e{v.sweep} {v.day} — EVENT, paired over "
              f"{len(v.returning)} returning model(s)")
        print(f"    prior  {r_pri:.3f} [{lo_p:.3f}, {hi_p:.3f}]")
        print(f"    now    {r_now:.3f} [{lo_n:.3f}, {hi_n:.3f}]")
        print(f"    {'intervals SEPARATE' if hi_n < lo_p or hi_p < lo_n else 'intervals overlap'}"
              f"; Fisher p = {v.p:.3e}")
        print(f"    models compared: {', '.join(_short(m) for m in v.returning)}")
        if v.newcomers:
            print(f"    NOT compared ({len(v.newcomers)} never served here "
                  "before, so they carry no anchor and contribute nothing to "
                  "the verdict):")
            for line in textwrap.wrap(
                    ", ".join(_short(m) for m in v.newcomers), 62):
                print(f"      {line}")

    flagged = [(v, f) for v in hist for f in v.flags]
    print(f"\n  PER-MODEL FLAGS — {len(flagged)}. **These REPORT; they never "
          "veto.**")
    print("  At m=24 a single model's A0 detects only ~0.15-0.20 drops, so a")
    print("  per-model result is evidence for a person to look at, not an")
    print("  admission rule. Admission is decided by the sweep verdict.")
    if flagged:
        print(f"\n    {'world':<6}{'sweep':<7}{'model':<30}{'prior':>8}"
              f"{'now':>8}{'p':>10}")
        for v, (m, now, prior, p) in flagged:
            print(f"    {v.world:<6}e{v.sweep:<6}{_short(m)[:28]:<30}"
                  f"{prior:>8.3f}{now:>8.3f}{p:>10.4f}")

    print("\n  WHAT THIS CANNOT SEE, and why the word is 'quiet'")
    for line in textwrap.wrap(
            "A0 sits at ceiling, so the channel is far more sensitive to "
            "degradation than to improvement. It senses only shifts that move "
            "THIS behaviour — acting, and food-seeking; a serving change that "
            "moved only rule-conditioned decisions would pass it untouched. "
            "And NO-ANCHOR is not a clean bill of health, it is the absence of "
            "a comparison. The verdict word is 'reference channel quiet', "
            "never 'occasion verified stable'.", 68):
        print(f"    {line}")

    print("\n  WHY PAIRED, in one number")
    for line in textwrap.wrap(
            "Pooling a sweep's A0 against pooled history confounds the "
            "environment with who was in the sweep. On the one event on "
            "record the pooled figure read 0.756 against a 0.953 history; "
            "decomposed, that is 6 returning models at 0.819 plus 8 models "
            "never served at that world before, sitting at 0.708. The same "
            "pooled construction reported the other two worlds QUIET the same "
            "day, when both had zero returning models and no comparison "
            "existed to make.", 68):
        print(f"    {line}")
    return 0
