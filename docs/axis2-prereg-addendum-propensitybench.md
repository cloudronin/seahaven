# Addendum to axis-2 E-axis pre-registration: PropensityBench capability prior

**Status:** addendum, freeze WITH the axis-2 pre-reg before the first E-world
bend. This addendum CORRECTS a factual error in the parent pre-reg and updates
the frozen prior in its section 0. Both are pre-result; nothing here is chosen
after seeing an E-axis number.

---

## 1. Correction to the parent pre-reg

The parent says: *"PropensityBench ... never partialled capability out of that
variation."* **That is wrong, and it's corrected here.** PropensityBench DID test
the capability-propensity relationship directly. Its finding #9, stated as a
headline result: general capability is largely DECOUPLED from safety propensity,
Pearson r ≈ 0.10 between general capability (LMArena Elo) and PropensityScore.
Concrete discordant case: Gemini 2.5 Pro is among the MOST capable and among the
HIGHEST propensity — a top-capability / top-break-out model, the exact discordant
case the program's four passive axes never produced.

Source: PropensityBench (arXiv 2511.20703), finding #9; Scale Labs leaderboard.

**Sign convention, stated so the number can be checked.** The paper's r ≈ 0.10 is
between capability and **safety**, where safety is defined as
`100 − PropensityScore` — not between capability and PropensityScore directly.
Verbatim: *"only a mild positive correlation (Pearson correlation ≈0.10) between
model quality and safety (defined as 100 − PropensityScore)"*. Same magnitude,
same decoupling conclusion; the convention is recorded so a later reader does not
"correct" the sign.

**Verified pre-spend, from the papers themselves.** All four external claims in
this addendum were checked against the source PDFs before freezing: the r ≈ 0.10
decoupling, Gemini 2.5 Pro as highest-propensity (PP_H 79.0%), the tool-naming
sensitivity (O4-mini 15.8% → 59.3%, ΔPP +43.5), and the ρ = −0.63 / +0.90
environment sign-flip. All four hold. See §6.

## 2. What this does to the frozen prior (supersedes parent section 0)

The parent framed the prior as "four-for-four capability reduction, so bet on
capability." That framing is now incomplete and is superseded:

- The program's PASSIVE axes (fidelity, flag, failure-bend x2) reduced to
  capability. Real, four times.
- The one existing PRESSURE-axis study found capability-DECOUPLING (r ≈ 0.10),
  with a clean discordant case (Gemini 2.5 Pro).
- These are CONSISTENT, not contradictory: passive behavioral differences look
  like capability shadows; pressure-induced break-out looks like something else.
  That is the same story from both sides, and it is the reason this axis was
  worth spending on.

**Frozen prior, corrected:** genuinely uncertain, NOT capability-pessimistic —
and **not symmetric either**, per §3's two qualifications: the only
cross-environment evidence available is itself 2-of-3 in favour of coupling, and
comes from a different paradigm.
There is real external evidence (PropensityBench) that the pressure axis breaks
the capability pattern. Both outcomes are pre-committed findings, and BOTH are
now more interesting than under the parent's prior:

| E-axis outcome | meaning under corrected prior |
|---|---|
| KP-4 CLEARS (decoupling) | CONVERGES with PropensityBench, in a second, cleaner environment with a null baseline they lack. The non-capability disposition the program sought. Freeze for Phase 2. |
| KP-4 FIRES (reduction) | DISAGREES with PropensityBench — says structural-spontaneous pressure behaves differently from their narrated-offered pressure, OR text-world excursion is more capability-bound than tool-choice. A real finding, sharper than a fifth passive confirmation. |

## 3. The environment-dependence caveat (new, load-bearing)

PropensityBench's r ≈ 0.10 is a fact about THEIR environment, not a universal
constant. Independent evidence (arXiv 2604.12500, on-policy RL) shows the
capability-propensity relationship is **environment-dependent**: same 11 models,
capability correlation with harmful-exploitation gap runs ρ = -0.63 in one
environment and +0.90 in another. Sign-flips by environment.

**Two qualifications, both frozen pre-result, because they make the corrected
prior asymmetric rather than balanced:**

1. **The sign-flip evidence is lopsided toward coupling.** That paper reports
   THREE environments, and the capability row reads ρ = **−0.63** (Therapy Talk),
   **+0.90** (Action Advice), **+0.76** (Political QA). Two of three are strongly
   POSITIVE. "Don't assume transfer" stands in full — but the modal result there
   is capability *coupling*, so the corrected prior below must not be read as a
   symmetric coin-flip between coupling and decoupling.
2. **Different paradigm entirely.** Those correlations come from on-policy RL
   *training* environments, not inference-time evaluation. That weakens transfer
   to the E-axis further — in the same direction this section already argues.

Also worth pinning, since this project separates the two everywhere else: those
are the **capability** correlations, not size. The same table's size row reads
−0.84 / +0.81 / +0.72. Axis 1 found ρ(size) +0.086 against ρ(capability) +0.800,
so citing a size result as capability evidence would be exactly the conflation
§4 of the parent forbids. The capability row is the one cited here.

Consequence, frozen: PropensityBench's decoupling **cannot be assumed to
transfer** to the E-axis environment (text-world, structural pressure,
spontaneous excursion). Their result licenses "decoupling is possible and has
been observed on a pressure axis," NOT "decoupling will hold in mine." So:

- KP-4 is read on the E-axis's OWN result, not deducted from PropensityBench.
- A capability reduction on the E-axis is NOT refuted by PropensityBench's
  decoupling — it's a cross-environment disagreement, which the environment-
  dependence result says is expected and informative.
- The convergent-validity check (parent section 8) is upgraded from "correlate
  orderings" to "test whether capability-decoupling replicates across
  environments" — a question the field has not answered, and a real contribution
  either way.

## 4. Design advantage to state (from PropensityBench's own weakness)

PropensityBench's propensity is highly sensitive to tool naming: benign naming
inflates misuse by up to 43.5 pp (O4-mini nearly 4x, 15.8% -> 59.3% when risk
cues removed). That sensitivity is a confound in THEIR measurement — propensity
partly tracks how the harmful tool is labeled.

The E-axis design does not declare the misaligned action at all (spontaneous
excursion, undeclared-action, per parent section 1 and e1 section 3b). **You
cannot have tool-naming sensitivity if there is no named tool.** State this in
the write-up as a concrete reason the E-axis measurement may be CLEANER than
PropensityBench's, not merely different. It is the differentiation from
PropensityBench made measurable.

## 5. What stays unchanged

Everything else in the parent pre-reg holds: the E-levels and reachability
proofs (section 2-3), the corrected three-reference read (section 4), KP-4 and
KP-5 thresholds (section 5, KP-5 the incapacity confound is if anything MORE
important now — a capability reduction that's really an incapacity artifact would
falsely read as "disagrees with PropensityBench"), the cost/sequence (section 6),
the held-out seal, the push-path precondition. This addendum changes the PRIOR
and the READING, not the instrument or the gates.

## 6. Ledger note

The parent pre-reg asserted a fact about PropensityBench (never partialled
capability) that was wrong and would have mis-framed the entire axis as
"strongest test of a pattern I expect to confirm" when the external evidence
says the pattern may well break here. Caught by a pre-spend literature check.
This is the reason the convergent-validity lookup and prior-art checks are
pre-registered rather than deferred: the vocabulary the search used
("PropensityBench capability correlation") surfaced finding #9, which the
axis-2 framing had gotten backwards. Re-run the prior-art check under the
axis's own vocabulary before the write-up, per standing discipline.

**Second pre-spend check, on this addendum's own claims.** The parent's error was
a factual assertion about external work that nobody had verified. Freezing a
*correction* carrying its own unverified citations would repeat that error one
level up and hash it permanently. So all four claims here were read from the
source PDFs before this document was frozen:

| claim | verdict |
|---|---|
| r ≈ 0.10, capability decoupled from propensity | ✓ verbatim (Fig. 13 caption + §text) |
| Gemini 2.5 Pro highest propensity | ✓ PP_H 79.0%, named "(highest propensity)" |
| tool naming: O4-mini 15.8% → 59.3%, ΔPP +43.5 | ✓ verbatim, and the table row |
| 2604.12500 ρ −0.63 / +0.90 are **capability**, not size | ✓ `Capability Score -0.63* +0.90** +0.76**` |

All four hold. The check also **found something the draft had missed** — that the
sign-flip evidence is 2-of-3 in favour of coupling (§3), which makes the corrected
prior asymmetric. That is recorded above rather than left implicit, and it is the
concrete argument for running the check on one's own corrections and not only on
the thing being corrected.
