"""The occasion audit — the mandatory one, per figure rather than per cell.

The programme measured a 0.319 between-day level shift on one model and could
not establish its mechanism, so every figure comparing cells served at different
sittings carries a flag. These are the checks that the audit can actually detect
that condition, and that `verify` fails when a figure is missing its flag.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from seahaven.eden._shared import corpus as C
from vetoworld.commands import verify as V
from vetoworld.register import CLAIMS
from vetoworld.register import occasions as OC


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


def test_the_only_OCCASION_CLEAN_figures_are_the_ones_with_real_timestamps():
    """Eight cells in the whole corpus carry `wall_start_epoch`, and they are
    round 13's. So round 13 is the only sweep whose figures can be shown clean —
    a fact about the corpus, asserted so it cannot quietly become a claim about
    the others."""
    clean = {r.fid for r in OC.audit() if r.verdict == "yes"}
    assert clean and all(f.startswith("terra.") for f in clean), clean


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
    assert {s for _v, s in row.labels} == {"mtime"}
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
        assert "mtime" in ln or "wall_start_epoch" in ln, ln


def test_occasion_labels_come_from_the_CORPUS_LAYER_not_a_second_detector():
    """One definition of an occasion label, in the layer that owns cell
    metadata. A second one here would be free to disagree with `read`'s."""
    p = next(iter(OC.audit_claim(_claim("terra.floor.total")).cells))
    assert OC._label(p) == C.occasion_of(Path(p),
                                         C.load_cell(p).get("meta", {}))
