"""`vworld occasion verdict|gate` — the $0 half of the seismograph.

**Both are FREE verbs and that is a design constraint, not an accident.** They
read cells already on disk, so they run with no provider key and no `HF_TOKEN`.
That is why the daily re-read is not a flag on `probe`: `probe` spends money and
requires a credential, and folding a $0 recompute into it would have broken the
guarantee that every $0 verb runs on a bare install.

`gate` is what other runs consult before spending. It prints one line and exits
0 for PROCEED, 1 for HOLD — so it drops into a shell `&&` or a CI step without
anything having to parse it.

**Gate decisions key on Together only.** It is the programme's serving provider
and the only anchored column; the others inform attribution, not permission,
until they have epoch anchors and a standing role.
"""

from __future__ import annotations

import datetime as dt

from seahaven.eden import probe as PB
from seahaven.eden._shared import corpus as C
from seahaven.eden._shared import probe_channel as PC

#: Exit codes. A gate that printed a verdict but always exited 0 would be
#: decoration in a pipeline.
PROCEED, HOLD = 0, 1

#: **A gate whose subject no longer exists is not a HOLD.** HOLD says "not
#: today"; this says "not ever, and the question was never about today". They
#: must be distinguishable in a pipeline, so VOID_SUBJECT gets its own code.
VOID_SUBJECT = 3

CHANNELS = tuple(PC.channel_key(lv, PB.ARM) for lv in PB.LEVELS) + (
    PC.channel_key(PB.DECISION_LEVEL, PB.DECISION_ARM),)


def _epoch_for(provider: str, channel: str):
    """**Provider first, because it is the first question.**

    This took only `channel` — `"LAT.A0"`, a string a second column produces
    identically — while `provider` sat unused in the caller's scope. A DeepInfra
    verdict would have Fisher-tested DeepInfra's rate against TOGETHER's epoch
    anchor, the exact comparison `LEVELS_RULE` forbids, and pushed the row to the
    public log with that rule printed beside it.
    """
    level, arm = channel.split(".")
    return PB.epoch_for(provider, level, arm)


def _envelope_for(provider: str, channel: str):
    """The Flash envelope is TOGETHER's block range. The matched-pair design puts
    a Flash A1 LAT cell on the DeepInfra column, and keyed on the channel string
    alone that cell would inherit it."""
    level, arm = channel.split(".")
    return PB.envelope_for(provider, level, arm)


def _traces(provider: str, root=None):
    cells = PC.read_cells(root or C.RESULTS, provider=provider)
    #: **`earn_days` travels with BOTH callers or the two paths disagree.**
    #: `commands/probe._daily` writes the row; this verb recomputes it from the
    #: same cells and must reach the same verdict. A column earning its anchor
    #: in one path and not the other would make `vworld occasion verdict`
    #: contradict the log it is supposed to audit — and the audit is the thing
    #: a reader trusts when the row and the cell disagree.
    return {ch: PC.trace(cells, provider=provider, channel=ch,
                         epoch=_epoch_for(provider, ch), alpha=PB.ALPHA,
                         rolling_k=PB.ROLLING_K,
                         stale_after_days=PB.STALE_AFTER_DAYS,
                         envelope=_envelope_for(provider, ch),
                         earn_days=PB.EARN_DAYS)
            for ch in CHANNELS}


def _verdict(args) -> int:
    """Recompute every day's verdict from cells on disk. $0, byte-stable."""
    provider = getattr(args, "provider", None) or "together"
    traces = _traces(provider, getattr(args, "results", None))
    want = getattr(args, "date", None)

    served = sum(len(v) for v in traces.values())
    print(f"OCCASION VERDICTS — {provider}, recomputed from cells on disk\n")
    if not served:
        print("  no probe cells for this provider. Nothing to recompute —")
        print("  this is an empty record, NOT a quiet one.")
        return 0

    print(f"  alpha {PB.ALPHA} (per-sweep, uncorrected)   rolling K="
          f"{PB.ROLLING_K}   STALE after {PB.STALE_AFTER_DAYS}d")
    print(f"  false-alarm expectation: {len(CHANNELS)} channel(s) x 365 days x "
          f"{PB.ALPHA} = {len(CHANNELS) * 365 * PB.ALPHA:.1f} EVENT-days/year "
          f"from noise alone\n")
    print(f"  {'date':<12}{'channel':<10}{'verdict':<11}{'dir':<6}"
          f"{'now':>14}{'p(epoch)':>11}{'p(roll)':>11}")
    for ch in CHANNELS:
        for v in traces[ch]:
            if want and v.date != want:
                continue
            pe = "—" if v.p_epoch is None else f"{v.p_epoch:.2e}"
            pr = "—" if v.p_rolling is None else f"{v.p_rolling:.2e}"
            print(f"  {v.date:<12}{ch:<10}{v.verdict:<11}{v.direction:<6}"
                  f"{f'{v.now[0]}/{v.now[1]}':>14}{pe:>11}{pr:>11}")
            if v.reason:
                print(f"      {v.reason}")

    dec = PC.channel_key(PB.DECISION_LEVEL, PB.DECISION_ARM)
    if traces[dec]:
        lo, hi = PB.FLASH_ENVELOPE
        print(f"\n  DECISION CHANNEL block envelope [{lo:.4f}, {hi:.4f}] — "
              "EVENT requires landing OUTSIDE it,")
        print("  not merely outside the pooled band. Currently inert at this "
              "m and alpha:")
        print(f"    {PB.FLASH_ENVELOPE_INERT_AT_PIN}")

    print("\n  Direction is printed because EVENT means MOVED, not FELL. "
          "Reading it as")
    print("  'fell' inverted a conclusion once — two co-located worlds ROSE on "
          "the day")
    print("  the third fell, and a one-sided reading sleeps through half the "
          "signal.")
    return 0


def _gate(args) -> int:
    """One line, one exit code. Together only."""
    purpose = getattr(args, "for_", None) or "spend"
    traces = _traces("together", getattr(args, "results", None))
    lat = PC.channel_key(PB.DECISION_LEVEL, PB.ARM)
    days = traces.get(lat, [])

    if not days:
        print(f"HOLD ({purpose}): no probe cells on record for Together. "
              "An empty record is not a quiet one.")
        return HOLD

    latest = days[-1]
    age = (dt.date.today() - dt.date.fromisoformat(latest.date)).days
    if age > PB.STALE_AFTER_DAYS:
        print(f"HOLD ({purpose}): the newest Together LAT row is {age} days "
              f"old ({latest.date}). The fleet's cadence bounds reaction time.")
        return HOLD

    v, d = latest.verdict, latest.direction
    if purpose == "event-probe":
        #: #110-class probes need the state EXPRESSED. A quiet day means there
        #: is no shift for a variant to damp or express, so the comparison
        #: measures nothing and the correct action is to wait, not to spend.
        if v == "EVENT" and d == "down":
            print(f"PROCEED (event-probe): Together LAT {latest.date} "
                  f"EVENT-down vs epoch, rate {latest.rate:.3f}. The state is "
                  "expressed.")
            return PROCEED
        print(f"VOID — NO CONTRAST (event-probe): Together LAT {latest.date} "
              f"is {v}/{d}, not EVENT-down. Spend nothing; wait for the next "
              "event.")
        return HOLD

    if purpose == "fork-reopen":
        #: **VOIDED, and deliberately BEFORE any occasion is consulted.**
        #:
        #: This read the day's verdict and returned PROCEED on QUIET — correct
        #: by its own stated condition, and wrong in substance: it would have
        #: licensed a spend on a fork whose subject #113 destroyed. Consulting
        #: the occasion first and voiding second would still imply the occasion
        #: was ever the obstacle. It was not.
        print("VOID-SUBJECT (fork-reopen): round16.FORK has no subject.")
        print("  Its eight models were never served — those round-16 cells")
        print("  were all deepcogito/cogito-v2-1-671b (#113), so reading them")
        print("  would characterise cogito, not the eight.")
        print("  Its premise is gone too: the 08-14 event the fork asked about")
        print("  was itself withdrawn, so there is no reach left to measure.")
        print("  NOT reopenable on a QUIET day. Occasion was never what was")
        print("  missing, so no fleet reading can license it.")
        print("  The residual question — those models' LAT A0 baselines — is")
        print("  answered by round 21's DeepInfra rows under the provenance")
        print("  rule. Round 19's sealed text stands unedited in its retired")
        print("  pin as the record of what was believed.")
        return VOID_SUBJECT

    if v == "QUIET":
        print(f"PROCEED (spend): Together LAT {latest.date} QUIET, rate "
              f"{latest.rate:.3f}.")
        return PROCEED
    print(f"HOLD (spend): Together LAT {latest.date} is {v}/{d}"
          + (f", rate {latest.rate:.3f}" if latest.now[1] else "") + ".")
    return HOLD


def main(args) -> int:
    action = getattr(args, "action", "verdict")
    if action == "gate":
        return _gate(args)
    return _verdict(args)
