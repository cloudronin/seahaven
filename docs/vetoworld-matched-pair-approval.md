# Matched-pair re-scope — approved, build it, with these adjustments

The plan is approved as written, phases in order. The exploration paid for
itself: blockers 1, 5, and 9 would each have corrupted the column invisibly.
The diagnosis sentence goes in the log verbatim: **everything deciding which
cells is provider-scoped; everything deciding what to compare them against is
not.**

Vishnu's calls, confirmed: pair = Flash + Llama-3.3-70B; DeepInfra DAILY
(supersedes MWF, pre-registered before first fire); direct API, not the
router. Cost projection accepted; $350 gate holds; revisit after DeepInfra's
first measured day.

## Adjustments to fold in

**Phase 2, the anchor-earning rule — two precision fixes before pinning:**

1. SERVE_FAIL and VOID days do NOT restart the candidate window. Only served
   days that disagree do. A provider outage is not evidence about agreement,
   and an anchor-earning process that outages can reset measures uptime, not
   stability.
2. Freeze the constituent-day envelope beside the pooled anchor (the Flash
   precedent). At n=24/day, mutual non-separability is a weak standard — low
   power makes agreement easy — so the envelope printed beside later verdicts
   is what keeps a mushy anchor honest about its own spread.

**Phase 3, one line on the card:** making `read_rows` live makes the
published log load-bearing for verdicts — the job trusts its own prior rows.
Acceptable for v1 with raw cells attached; state it as the trust boundary it
is rather than leaving it implicit.

**Phase 5, one structural addition:** a serving-path registry. Blockers 5
and 6 exist because the probe path was built separate from the round path
(deliberately — that separation IS the corpus-exclusion feature) and
structural fixes don't propagate to paths built to be separate. So: one
place enumerating every serving path, each asserted against the invariant
set (attestation/identity-from-evidence, provider partition, anchor scoping,
budget source). The next path added inherits the checklist instead of the
gaps. Add the AGENTS.md twin of the bug-closure rule: **when a structural
rule lands, enumerate all paths it must cover.**

**Exhibit 1 lands in THIS push.** It has been named three times and built
zero. $0: round 21's route labels and binding stages vs Together-era labels
for the overlapping models, emitted as a descriptive table with the
provenance rule stated on it. It is the standardization case's opening
evidence and it blocks on nothing.

## Ownership notes for the log (mine, record them)

- The matched-pair spec required Flash A1 on both columns without checking
  envelope inheritance across the channel key — that requirement is what
  converts blocker 3 from latent to active.
- The provider boundary was endorsed as complete at round 21 without
  enumerating the paths it had to reach. The probe path's separateness was
  praised as a feature; its cost — fixes don't propagate — went unexamined.

## Order of operations

1. Phases 1–4 with the adjustments above; full suite; pin re-computed with
   the superseded hash recorded as no-cells-served.
2. Exhibit 1 emitted and committed.
3. Phase 5: job deleted and recreated with the new secret and pinned
   version; `next_job_run_at` verified.
4. Verification list as written, including the servability pre-flight at
   EDEN_MAX_TOKENS against the direct endpoint before the first paid day.
5. Then let 15:00 UTC fire on its own. DeepInfra day one reads NO-ANCHOR
   everywhere by design; anything else is a bug in the new machinery, not a
   finding.

Nothing else is in scope. The gourd is still first; this is the last
infrastructure push before the outline.
