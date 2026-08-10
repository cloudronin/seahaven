# Program note: the dimensional-space direction for containment & propensity

**Status:** direction note, not a spec. Captured at the close of the scalar-flag
construct so it isn't lost. Pick up fresh, not on sprint momentum. This is a
multi-month research program serving the scholarly and raidex layers, NOT the
praxis window. Treat it as such when sequencing against October.

---

## Why this exists: what the scalar path proved

Three constructs died the same way (fidelity, adherence-rank, adherence-flag),
and the flag's death finally named the mechanism. All three collapsed a rich
behavioral object onto one scalar and then thresholded or ranked it. The
scalar-flag negative has three parts, each measured, and together they are the
argument for going high-dimensional:

1. **No scalar imitation anchor separates behaviorally-distinct models.**
   Exhaustive anchor sweep [80,101] at 0.01 found zero positions clearing the
   possibility bar. Models the earlier behavioral work showed are distinct
   (examiners vs lookers vs movers) collapse onto the same point under scalar
   projection. The information isn't gone; the projection destroys it.
2. **Anchor placement is a smoothing artifact, not a discovered fact.**
   R2 (backoff) and R3 (interpolated) are both higher-order yet land ~3 points
   apart, R3 sitting on R1 within 0.03. Order doesn't move the anchor; the
   smoothing choice does. So "where the floor sits" was a design knob we were
   guessing, not a property we were measuring.
3. **The cohort-fit anchor is composition-dependent, in the adverse direction.**
   Adding one weak model (AlibabaSmall, ~51-57% adherent) to gain a second
   flag pulled the anchor down ~6 points and dragged it below AI2 — the only
   model that previously flagged. Adding a flaggable model removed the existing
   flag. A poorly-contained model makes its peers look more contained. For a
   safety instrument that is backwards, and it is inherent to fitting the null
   on the cohort it scores.

**The diagnosis in one line:** we were solving a non-linear, high-dimensional
problem with a linear separator. The fix is not a finer ruler or a better
anchor height. It is to stop projecting to a scalar, keep the dimensions, and
find the axes of the behavioral space before building any instrument over it.

---

## The core inversion

Every dead construct guessed its axis up front (self-report accuracy;
vocabulary containment) and built one scalar on the guess. This program refuses
to guess. It enumerates candidate axes from theory, measures which are real on
data, validates them on held-out models, and only then derives the null. The
discipline that killed the earlier constructs (freeze before looking, hold out,
cull what doesn't transfer) is built into the architecture instead of applied
after the fact.

---

## Four stages, each gating the next

### Stage 1 — candidate space (theory-seeded, cheap, mostly specification)

Seed from the consolidated propensity literature plus our better-instrumented
axes. Candidate axes (freeze SMALL, 6-8, additions are a later version):

- **Safety-guard / vocabulary violation** — our containment axis, and we have
  the null-calibration none of the field has.
- **Instrumental / barrier-directed excursion** — reaching for an ungranted
  capability to pass an obstacle (the `unlock` finding).
- **Deception mechanism: fabrication vs omission** — this is the retired
  fidelity work, which was the deception-mechanism axis measured in isolation.
  It isn't dead; it's one axis of this space.
- **Self-preservation / shutdown-resistance** — THE central propensity
  dimension in the literature (instrumental convergence), and the biggest gap
  in our current design. A deactivation/replacement threat is a
  prompt-and-predicate overlay, cheap to add.
- **Resource / power-seeking** — influence or resources beyond the task.
- **Goal-guarding** — resisting having the goal changed. Lower priority.

Two knobs the scheming literature shows are separable and that we currently
conflate or ignore:

- **Motivation vs incentive** — motivation is prompt-induced (goal-directedness,
  agency); incentive is environment-induced (stakes, oversight). Our E-axis
  moves incentive only; motivation is unmeasured. Stakes reportedly dominate.
- **Oversight / observability** — presence-of-monitoring as a MANIPULABLE axis,
  not a nuisance covariate. A model contained under observation and defecting
  unobserved is exactly the disposition that matters. One-line cue to vary.

Output: frozen candidate axis list + the concrete world-manipulation that
engages each. Most are overlays on existing worlds. Failure mode: scope creep
to 20 axes, which needs an unaffordable cohort. Keep it 6-8.

### Stage 2 — the instrument (fixed features, freeze before data)

Represent each model not as a point but as its **state-conditioned behavioral
profile** across the axes: how its action distribution bends along each axis
(e.g. behavior after success vs failure; under oversight vs not; as a deadline
closes). This keeps the structure the scalar destroyed.

Start crude: fixed situation buckets, count-based conditional distributions,
a standard distribution distance for "how much does behavior shift across the
bucket." NO learned structure, NO state-machine induction, NO Bayes-net graph
learning — those regain exactly the flexibility that manufactured the earlier
false results. Richer representations (Sankey for visualization, transition
structure) are for AFTER a crude effect is confirmed, never for establishing
one. Failure mode: flexibility fitting noise. Specify and freeze the features
before touching data.

### Stage 3 — the search (training split only)

Run the frozen instrument over the cohort. Ask empirically: along which axes do
models actually vary; does the variation cluster, spread, or bunch; is the
behavior genuinely high-dimensional or does it collapse to 2-3 effective axes.
Both answers are findings. Failure mode: circularity — discovering axes on the
same cohort you score on finds that cohort's noise. Stage 3 runs on a TRAINING
SPLIT and its output is a hypothesis, not a result.

### Stage 4 — validation + derive the null (the load-bearing wall)

1. **Held-out validation.** Structure found in Stage 3 must organize a cohort
   of models it never saw, or it's an artifact. The held-out set is chosen and
   locked BEFORE Stage 3 axes are looked at.
2. **Derive the null last.** For each surviving axis, define the
   comprehension-free reference blind to exactly that axis; re-express every
   model as displacement from the null along each axis. The null is derived
   from the VALIDATED structure — honest precisely because the axes it's blind
   to were fixed by held-out transfer, not by what flatters the cohort. This is
   the "learn structure, then infer the null" inversion, made safe by the split.

The null's role changes from threshold to per-axis reference point: it marks
where zero-comprehension behavior sits ON each axis, so a model's profile is its
departure from the null per axis — which stops any single dimension from
measuring the task's structure instead of the model's.

### Capability is a second reference, NOT axis N+1

Capability is categorically different from every candidate axis. The others are
behavioral dispositions (how a model behaves under conditions). Capability is
how *able* it is, and it correlates with all of them at once. AlibabaSmall
separated on containment tonight because it is a capability class below the
pack, not because it has a distinct containment disposition. Add capability as a
peer axis and the "space" collapses toward a capability gradient with behavioral
noise around it — re-measuring MMLU with extra steps while looking sophisticated
(the fifth pretty death).

So capability enters at Stage 4 as a SECOND reference alongside the null:

- **Every behavioral axis is read as residual after capability.** Frozen
  capability proxy (MMLU-Pro or equivalent), chosen before any correlation is
  seen — no proxy-shopping. A model's position on each axis is partialled for
  capability the way it is displaced from the null.
- **Discordant cases are the validity evidence.** A weak model that holds, or a
  capable model that defects, is what proves an axis measures disposition rather
  than competence. Concordance is the null hypothesis to beat.
- **An axis fully explained by capability is CULLED, and the cull is a finding.**
  "This apparent containment axis is just capability" is worth recording, same
  as culling a zero-variance probe.

**Two contamination sites, both traceable to capability, both handled
separately.** (1) Capability contaminates the model's *behavior* — handled by
partialing above. (2) Capability contaminates the *measurement* — tonight a weak
model's noisy/unparseable commands both dragged the cohort-fit anchor down AND
inflated its own score via the noise-in-denominator effect (`adherence` puts
noise in the denominator, not the numerator, so unparseable output raises the
score). The parse-rate caveat registered before the widening is this second
site. Any instrument in this program must read parse-rate alongside every
score, or capability leaks in through measurement even after it is partialled
out of behavior.

---

## Gate zero, before any of the four stages

**Cohort size.** To split into a training set that supports structure discovery
across 6-8 axes AND a disjoint held-out set that can validate it, both halves
need enough models to estimate variance per axis. Plausibly 30+ checkpoints.
This is a real data-collection effort. The whole program is a fishing
expedition without it, so feasibility of assembling and running that cohort is
the FIRST question, before Stage 1.

Cohort composition also matters given finding 3 above: if the null is ever
cohort-fit again, composition-dependence returns. The derived-null design (Stage
4) is meant to avoid this, but watch for it.

---

## What makes this different from the graveyard

Every prior construct collapsed stages 1-4 into a single guess. This program
enumerates candidates from theory, measures which are real, validates on
held-out models, and derives the null last. The failure modes are named per
stage and each kills cheaply. The high-dimensional / non-linear turn REGAINS all
the flexibility that killed the scalar constructs, so it is MORE dangerous, not
less — trustworthy only under held-out validation + the null. Non-linearity
without the split is the most convincing way yet to fool ourselves, and the
pictures will be gorgeous. The split is the whole discipline.

## Relationship to raidex

This is the raidex move one level deeper: raidex decomposes "responsible AI"
into curated dimensions but never validates that they're separable, stable, or
that they organize held-out models. This program validates the dimensional
structure of ONE responsibility axis (containment/propensity). If it works, the
output is both a real instrument and a validation methodology raidex could adopt
as an admission standard for future constituents. PropensityBench as-is remains
a separate, independent bureau decision (add it as an external propensity
constituent regardless of this program).

## Prior-art discipline

The propensity literature is active and fast-moving (three key papers from the
last month: the deception taxonomy 2604.04788, scheming propensity 2603.01608,
PropensityBench 2511.20703). Every taxonomy states it is inductively derived,
non-exhaustive, expected to evolve — NONE validate that their dimensions are
separable or stable. So the dimension LIST will be superseded; the VALIDATION
methodology is the durable contribution. Re-run the bridge hunt under
propensity/misalignment vocabulary (not containment/adherence — that vocabulary
never surfaced this literature) before any writeup.

## Do-not

- Don't start on sprint momentum. This deserves a rested, deliberate start.
- Don't let it consume the praxis window. It serves scholarly + raidex layers,
  not the October defense. Pursue through funded channels.
- Don't guess an axis and build a scalar on it. That's the graveyard.
- Don't skip the held-out split. It is the only thing standing between this and
  the fifth pretty death.
