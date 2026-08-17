"""The occasion audit — the mandatory one, per figure rather than per cell.

The programme measured a 0.319 between-day level shift on one model and could
not establish its mechanism, so every figure comparing cells served at different
sittings carries a flag. These are the checks that the audit can actually detect
that condition, and that `verify` fails when a figure is missing its flag.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from seahaven.eden._shared import corpus as C
from vetoworld.commands import verify as V
from vetoworld.register import CLAIMS
from vetoworld.register import occasions as OC

#: **Corpus-wide tests: not unit tests, not timed like them (#112).** These
#: recompute every figure over the whole corpus, so they grow with it. The
#: project default of 120s caught that growth once, as an unexplained flake
#: rather than as the arithmetic it was. Measured at 425 cells, the slowest here
#: is ~53s.
pytestmark = pytest.mark.timeout(600)



def _claim(fid):
    return next(c for c in CLAIMS if c.fid == fid)


def test_the_read_set_is_RECORDED_not_DECLARED():
    """**The design commitment.** `Claim.cells` is prose; if the audit trusted
    it, a figure could change what it reads and the audit would keep describing
    the old inputs. Wrapping the loader means it cannot."""
    row = OC.audit_claim(_claim("cogito.worlds"))
    names = sorted(p.name for p in row.cells)
    assert len(names) == 3
    assert any("e12" in n and "LAT" in n for n in names)
    assert any("e11" in n and "W2" in n for n in names)
    assert any("e11" in n and "W3" in n for n in names)


def test_a_figure_that_reads_NOTHING_DECLARED_is_still_caught():
    """The point restated as a failure the old design would have missed: a claim
    whose `cells` string names one sweep while its lambda reads two."""
    lying = replace(_claim("cogito.worlds"), cells="e11 only, honest")
    assert OC.audit_claim(lying).sweeps == ("11", "12")


def test_MTIME_can_never_produce_a_same_occasion_verdict():
    """mtime is when a file was last WRITTEN — for a gap-filled cell, its last
    attempt. Two agreeing mtimes are not evidence of a shared sitting, so the
    verdict is `unknown-mtime` however well the dates line up."""
    for row in OC.audit():
        srcs = {s for _v, s in row.labels}
        if len(row.cells) > 1 and srcs != {"wall_start_epoch"}:
            assert row.verdict == "unknown-mtime", row.fid


def _timestamped_sweeps() -> set[str]:
    """The sweeps whose cells recorded a real serving timestamp — **derived from
    the corpus, never declared.**

    Round 13's spec required one; round 14 got one because `vworld run --round`
    now writes it for every cell. Everything earlier has only mtime,
    permanently: a timestamp cannot be recovered after the fact.

    This was a hand-typed literal, `{"13", "14", "11tA", "11tB"}`, and it went
    stale the moment round 15 landed — all 98 e15 cells carry
    `wall_start_epoch` and the literal did not know it, so the clean-figure
    assertion below was checking a claim about four sweeps against a corpus
    containing five. **That is the duplicated-vocabulary drift already seen in
    doctor's `ROUNDS` and in `ARTIFACTS`/`fn`**, in a file whose own docstring
    says it asserts facts about the corpus. A fact about the corpus should be
    read off the corpus.

    A sweep counts only if EVERY one of its cells is timestamped. Today `all`
    and `any` select the same five sweeps, but a partially-instrumented sweep is
    not one whose figures can be shown clean, and the strict reading is the one
    the assertion below needs.
    """
    total: dict[str, int] = {}
    stamped: dict[str, int] = {}
    for path, cell in C.iter_cells():
        got = C.parse_cell_name(path.name)
        if not got or got["schema"] != "current":
            continue
        rnd = got["round"]
        total[rnd] = total.get(rnd, 0) + 1
        if cell.get("meta", {}).get("wall_start_epoch"):
            stamped[rnd] = stamped.get(rnd, 0) + 1
    return {r for r, n in total.items() if stamped.get(r, 0) == n}


_TIMESTAMPED = _timestamped_sweeps()


def test_the_TIMESTAMPED_derivation_DISCRIMINATES():
    """The cost of deriving a set instead of declaring it is that a broken
    derivation returns everything, or nothing, and every assertion built on it
    passes vacuously. So: it must be non-empty, and it must exclude somebody."""
    every = {got["round"] for path, _c in C.iter_cells()
             if (got := C.parse_cell_name(path.name))
             and got["schema"] == "current"}
    assert _TIMESTAMPED, "no sweep reads as timestamped — derivation is broken"
    assert _TIMESTAMPED < every, "every sweep reads as timestamped — no discrimination"


def test_a_figure_is_OCCASION_CLEAN_only_if_every_cell_it_reads_is_timestamped():
    """Asserted as a fact about the corpus rather than about the code: no figure
    reading an e10, e11 or e12 cell can be shown clean, however its mtimes fall,
    because those sweeps recorded no serving time and never will."""
    clean = {r.fid for r in OC.audit() if r.verdict == "yes"}
    assert clean, "some figure should be clean"
    for row in OC.audit():
        if row.verdict == "yes":
            assert set(row.sweeps) <= _TIMESTAMPED, (row.fid, row.sweeps)
        elif set(row.sweeps) - _TIMESTAMPED and len(row.cells) > 1:
            assert row.verdict == "unknown-mtime", (row.fid, row.verdict)


def test_MTIME_UNDER_DETECTS_which_is_why_the_sweep_column_exists():
    """Rounds 11, 12 and 13 were three separate sweeps whose files share one
    mtime day. A timestamps-only audit would call those comparisons same-day —
    the exact error the audit exists to prevent — so the round tag is reported
    beside it, and it is recorded provenance rather than a filesystem artifact.
    """
    row = OC.audit_claim(_claim("derivation.checks"))
    assert len(row.days) == 1, "the premise: mtime alone sees one day here"
    assert row.sweeps == ("11", "12"), "the sweep column sees two sittings"
    assert row.needs_flag


def test_A_FIGURE_SPANNING_SWEEPS_WITHOUT_A_FLAG_FAILS_VERIFY(capsys,
                                                              monkeypatch):
    """**The register regression.** Strip the flag off the figure that reads
    across four rounds and `verify` must exit nonzero AND name it."""
    stripped = [replace(c, occasion="") if c.fid == "break.gone" else c
                for c in CLAIMS]
    import vetoworld.register.claims as CL
    monkeypatch.setattr(CL, "CLAIMS", stripped)

    assert "break.gone" in OC.unflagged()
    rc = V.main()
    out = capsys.readouterr().out
    assert rc == 1, "verify passed with an unflagged cross-occasion figure"
    assert "COMPARE CELLS ACROSS SERVING OCCASIONS WITH NO FLAG" in out
    assert "break.gone" in out.split("NO FLAG:")[1]


def test_the_committed_register_has_NO_unflagged_figures():
    assert OC.unflagged() == []


def test_the_FOUR_KNOWN_MEMBERS_are_confirmed_by_the_WALK(capsys):
    """The closeout named four comparisons it expected to be cross-occasion and
    said to CONFIRM them by walking the corpus rather than asserting them. Each
    is registered, so each is walked."""
    rows = {r.fid: r for r in OC.audit()}
    for fid in ("round8.gen1v2.LAT", "round3.halves", "cogito.worlds",
                "flash.blocks"):
        assert rows[fid].needs_flag, f"{fid} should carry an occasion flag"
        assert _claim(fid).occasion, f"{fid} has no flag text"


def test_the_ROUND_3_HALVES_are_invisible_to_BOTH_signals():
    """**The audit's own blind spot, asserted rather than discovered later.**

    A top-up reuses its round tag, so the sweep column reads `e3` for both
    halves, and neither cell carries a serving time. The machinery cannot see
    this split at all — it is known from the round's design, and the flag says
    so. An audit that hid its blind spots would be worse than none.
    """
    row = OC.audit_claim(_claim("round3.halves"))
    assert row.sweeps == ("3",), "the sweep column cannot separate a top-up"
    assert {s for _v, s in row.labels} == {"git_add(reconstructed)"}
    assert "Neither signal detects it" in _claim("round3.halves").occasion


def test_emit_occasions_prints_the_SOURCE_and_never_launders_mtime(capsys):
    from vetoworld.commands import emit

    class _A:
        artifact = "occasions"
    assert emit.main(_A()) == 0
    out = capsys.readouterr().out
    assert "wall_start_epoch" in out and "mtime" in out
    assert "mtime is NOT a serving date" in out
    assert "UNDER-DETECTS" in out
    # every consumed cell prints a source beside its timestamp
    body = out.split("PROVENANCE OF THE")[1]
    stamps = [ln for ln in body.splitlines() if ln.strip().startswith("2026-")]
    assert stamps
    for ln in stamps:
        assert any(k in ln for k in ("mtime", "wall_start_epoch",
                                     "git_add(reconstructed)")), ln


def test_occasion_labels_come_from_the_CORPUS_LAYER_not_a_second_detector():
    """One definition of an occasion label, in the layer that owns cell
    metadata. A second one here would be free to disagree with `read`'s."""
    p = next(iter(OC.audit_claim(_claim("terra.floor.total")).cells))
    assert OC._label(p) == C.occasion_of(Path(p),
                                         C.load_cell(p).get("meta", {}))


def test_NO_PROSE_HARDCODES_A_LIVE_CELL_COUNT():
    """**Inverted, because the corpus now grows daily.**

    This asserted four files state the CURRENT "N of M" split of timestamped
    cells, so a sweep that recorded timestamps failed rather than quietly making
    three files wrong. That was right when sweeps were occasional. The daily
    probe fleet turns it into a four-file treadmill every twenty-four hours, and
    a guard that must be satisfied daily gets satisfied carelessly.

    So the rule flips: prose may not carry a live count at all. The count is
    computed and printed by `emit occasions`, which cannot go stale because it
    reads the corpus every time. Prose that states no number cannot state a
    wrong one.
    """
    import re
    from pathlib import Path

    tot = sum(1 for _p, _d in C.iter_cells())
    root = Path(__file__).resolve().parents[1]

    #: "N of M" where M is plausibly a corpus size. Small ratios such as
    #: "2 of 3 worlds" or "5 of 6" are not cell counts and stay legal.
    pat = re.compile(r"\b(\d+) of (\d{3,})\b")
    #: **And the bare form, `N cells`.** The regex above only caught ratios, so
    #: the corpus card carried "**357 cells**" and a manifest line reading
    #: "(259 cells)" while the corpus held 481 — a published document with three
    #: stale counts, in front of a test written to stop exactly that. A guard
    #: that checks one spelling of a fact is a guard that the other spelling
    #: walks past.
    #: `figures` joins `cells` because the register grows too — the card said
    #: "all 17 figures recompute" while `verify` reported 21.
    bare = re.compile(r"\*{0,2}(\d{2,}) (?:cells?|figures?)\b")
    bad = []
    for rel in ("vetoworld/register/occasions.py", "vetoworld/commands/emit.py",
                "vetoworld/commands/run.py", "docs/vetoworld-corpus-card.md"):
        src = re.sub(r"\s+", " ", (root / rel).read_text().lower())
        bad += [f"{rel}: {g.group(0)!r}" for g in pat.finditer(src)
                if int(g.group(1)) <= int(g.group(2))]
        #: The 166 relabelled cells are a CLOSED historical set — it cannot
        #: grow, so naming it is a fact, not a live count.
        bad += [f"{rel}: {g.group(0)!r}" for g in bare.finditer(src)
                if g.group(1) not in ("161", "166")]
    assert not bad, (
        "prose hardcodes a live cell count, which the daily probe makes wrong "
        f"within a day: {bad}. State the fact, not the figure — "
        "`emit occasions` prints the current split.")


def test_emit_occasions_STILL_PRINTS_the_live_split(capsys):
    """The other half of the inversion. Removing counts from prose is only safe
    while the artifact still reports them, or the number stops existing."""
    from vetoworld.commands import emit

    class _A:
        artifact = "occasions"
    assert emit.main(_A()) == 0
    out = capsys.readouterr().out
    #: The split is still reported; the vocabulary widened from two sources to
    #: three when 249 mtime labels were reconstructed ([CORRECTION] 11). What
    #: must hold is that the artifact states how many figures rest on a
    #: MEASURED timestamp versus a derived one — removing counts from prose is
    #: only safe while the artifact still reports them.
    assert "measured," in out and "derived" in out
    assert "by source:" in out
