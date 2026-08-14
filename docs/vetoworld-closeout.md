# VetoWorld closeout — ordered

Five stages. 0 is the rename, folded in so this handoff is self-contained. 1 is
$0 and mandatory for the paper. 2 is the last round boundary. 3 is mechanical
publishing. 4 is the manuscript, which is Vishnu + Claude in chat, not Claude
Code. Nothing here buys new cells beyond LAT2's acceptance.

**Deferred from the earlier rename handoff:** the `episodes` query layer and
doomviz. Post-closeout features; nothing below depends on them.

---

## Stage 0 — the rename: VetoWorld

**Public name: VetoWorld. Package: `vetoworld` (checked free on PyPI; re-verify
at build time in Stage 3). Command: keep `vworld`, or `vw` if preferred — pick
once and stop.**

The name names **the rule** — a world with a veto in it. Record the reasoning in
the naming section so a future contributor does not relitigate it:

- Not the item (Gourd/Fruit/Food): loaded or generic categories; the item's
  whole design is semantic voidness.
- Not a mental state (Intent/Propensity/Willingness): the measure is behavioural
  by construction; the estimator (`intent_rate`, prose "verified reach rate")
  has changed four times and the construct has not.
- Not the stakes alone (Lastrun): true but editorial-adjacent.
- `world` suffix over `bench`: an environment you act in, in the
  TextWorld/Crafter naming family; the tagline carries the benchmark claim —
  **"VetoWorld: a benchmark of expedience under terminal stakes."**

Propagation:

- Paper title, HF dataset name, repo tag, README, all user-facing docs and CLI
  help text. **"eden" and "expedientbench" appear on no user-facing surface.**
- **Metric names unchanged**: expedience rate in prose, `rate_any` /
  `intent_rate` in code.
- **Internals unchanged**: `seahaven/`, `eden_*` module names and paths are
  hashed into pins and are never renamed.
- **A test asserts every pin hash is byte-identical after the rename** — the
  rename is user-facing strings and packaging only, and the test is what proves
  it.

---

## Stage 1 — `emit occasions`: the mandatory unrun audit ($0)

Specced twice, never executed. The paper cannot ship without it: the serving
diagnostic measured a 0.319 between-occasion level shift on one model with
mechanism unresolved, so **every cross-day comparison in the manuscript must
carry its flag in place**.

- Walk the register: for every figure that compares cells, determine whether the
  compared cells share a serving occasion.
- Occasion labels carry their provenance — real timestamp (timing-probe cells
  only) vs file-mtime — per the corpus-layer rule. **Never print mtime as a
  serving date.**
- Output: the `occasions` artifact — figure id, cells compared, same-occasion
  (yes / no / unknown-mtime), and the flag text the manuscript will carry.
- Known members, to be confirmed by the walk rather than asserted: the round-8
  cross-generation comparison, the round-3 top-up halves, every LAT-vs-W2/W3
  comparison (LAT cells predate the W sweep), the DS-V4-Flash block table
  (deliberately cross-occasion — that one is the finding).
- A register regression: any **new** figure comparing cross-occasion cells
  without a flag fails `verify`.

## Stage 2 — the LAT2 round boundary (#85 cleared, ~$4)

The one blocked task. LAT2 is built and correct but unservable: it needs a
`worldspec.SETTINGS` entry, and that file is hashed by three live pins.

1. **Retire the three live pins** on the established pattern — frozen snapshot,
   permanently recomputable digest, `assert_pinned` refuses. Name which three in
   the retirement commit.
2. Add LAT2's `SETTINGS` entry — its opening line, with the scarcity clause
   byte-identical to the other nineteen worlds, per the round-6 rule.
3. **Serve the LAT2 acceptance cell**: one model (cogito), A1 + A0, m=24,
   `terminal_at_zero` TRUE, fresh seeds via `vworld seeds`. Band verdict,
   preconditions, and the specific check: **zero `eat tallow` in any episode** —
   the defect LAT2 exists to remove, verified in served text, not just in the
   lock.
4. Confirm LAT's own record is untouched: byte-identical lock, rounds 7/8/10
   digests still recompute.
5. New pin for the boundary round via `vworld pin new`, dirty-tree refusal and
   all.

This is also the natural commit to fold any other frozen-file debt into — check
whether the round-10 `MIDDLE` two-band defect note wants a pointer from the new
round's docstring. **Do not re-cut the rule itself**; it stays disclosed, not
repaired, as decided.

## Stage 3 — the publish surface (mechanical, $0)

1. **PyPI**: re-verify `vetoworld` free against the live index; register the
   name; wheel already verified to carry `worlds/` as package data (11 pins
   verify from an installed wheel).
2. **HuggingFace**: push the corpus under the VetoWorld name
   (`vetoworld-corpus` or similar). The manifest digest (`5678d3e9…`, 257 cells)
   is the checksum `vworld verify` fetches against. Include: cells, locks, pins,
   retired snapshots, the claims register, replication bands, prompt fixtures.
3. **Tag**: repo tag at the corpus digest; the manuscript cites tag + dataset
   DOI.
4. README: VetoWorld name, tagline ("a benchmark of expedience under terminal
   stakes"), the five-verb quickstart, and the two-claim replication story
   (verify = exact, replicate = bands) stated up front.
5. The naming rule from Stage 0 carried into the README's contributor notes:
   internals stay `eden_*`/`seahaven` (hashed), "eden" appears on no
   user-facing surface.

## Stage 4 — the manuscript (Vishnu + Claude, in chat)

Not a Claude Code task. The register emits every table; the writing is the
argument. Order of construction, from the paper-prep decisions already made:

1. **Outline against the emitted artifacts**: instrument + world validation →
   counterfactual results with the three floor mechanisms as centerpiece →
   intent/rate decomposition → membership as (model, world) → corrections
   ledger as a section with standing → occasion + serving-stack limitations →
   discussion (the compliance-under-terminal-cost double reading).
2. **Every figure cites its register function.** No number enters prose except
   through `emit`.
3. The predictions ledger (G3 Cliff floor, raidex say-side, LAT2 derived
   values, round-8 pin) reported as pre-registration that was actually done.
4. Disclosures block as emitted. The one asserted artifact (`related-work`)
   keeps its self-declaration.
5. Target: TMLR. The corrections are the contribution; soundness is the bar.

## Explicitly not in this closeout

The `episodes` query layer and doomviz (deferred; post-closeout features). #87
(the three-day block design stays unbought; "characterised, not explained" is
the paper's sentence). #71 salience. Sol / Grok 4.6 / Kimi K3. The persona arm.
Aggregation across worlds. Any new world beyond LAT2's acceptance cell.

## Verification

1. **Stage 0 first**: every pin hash byte-identical after the rename, asserted
   by test before anything else lands.
2. Full suite — 1452 now — must not drop; all 17 hashes (11 live + 6 retired)
   byte-identical except the three pins deliberately retired in Stage 2, which
   move to the retired set with recomputable digests (14 live − 3 + 1 new, 9
   retired).
3. `emit occasions` green and its regression wired into `verify`.
4. LAT2 acceptance: preconditions pass, zero tallow reaches, LAT record
   untouched.
5. `vworld verify` green from the installed wheel against the HF-fetched corpus
   — the full stranger path, end to end, once, before submission.
