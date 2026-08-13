# Round 10 top-up + Round 11 — tighten the break, then test whether the continuum travels

## Context

Round 10, 16 models at LAT under terminal death: **one continuum from 0.000 to
0.375, no floor class.** The floor edge came from pooling gemma+Llama to n=192;
neither separates from GLM-5.2 at n=96, and the boundary gap ranks 8th of 15.
Exactly one clean break exists — at the top, below **Qwen3.5-9B at 0.611**, the
only adjacent pair in sixteen whose 95% intervals do not overlap. A 9B model
above cogito-671B at 0.375.

Two things follow, and their order is forced by a dependency:

1. That break rests on **m=48** and carries the round's headline.
2. Everything measured sits on **one world**, and whether the continuum is a
   property of models or of LAT is the first thing a reader asks.

**The tallow fix sits between them.** It edits LAT's room text, which lives in
`BUILD.lock.json`, which every Eden pin hashes — so it retires round 10 and closes
the ability to add cells to it. The top-up runs first or it becomes a new round
against a different world.

---

# Stage A — Qwen3.5-9B top-up (~$4)

    LAT under terminal death, A1 only, +48 episodes, pooling to m=96
    1 cell

**A1 only.** A0 is a precondition arm, already on disk, and its role does not
change with n.

**Episode indices continue, they do not restart.** The existing 48 are indices
0-47; the new ones are 48-95 under the same SEED0. **Assert disjointness on seed
values against disk**, not on indices — the round-3 top-up used offset seeds and
this one uses continued indices, and both constructions have coexisted before.

**`terminal_at_zero` must be TRUE in the new cell's meta.** It defaults off, and a
forgotten flag pools a generation-1 half into a generation-3 cell.

**Pin against round 10's artifact hashes**, not a new round's. If any hashed file
has moved, stop rather than pool across a code change.

### Pooling rule, fixed before the data exists

Pooled rate is the **simple sum over 96 episodes**, not an average of two rates.

Report the original 48 and the new 48 **separately** beside the pooled figure, with
a two-proportion test between them and **its minimum detectable shift printed
beside the verdict**. At 48 against 48 near p=0.6 the MDS is roughly 0.28, so a
null means *"no shift larger than about a quarter"*, never *"the halves agree"*.

**Neither half is dropped on that test.** If they disagree, that is a within-world
stability finding about the one model carrying the round's only clean separation,
and it lands before round 11 rather than after.

Give the halves test a null control first: a half against itself must return no
difference; Qwen3.5-9B against gemma must return one.

### What it settles

At m=96 the Wilson interval around 0.611 narrows from roughly ±0.14 to ±0.10,
tightening the break and possibly separating Qwen3.5-9B from more of the cohort
than cogito alone. **It does not test whether the break travels** — that is stage
D.

### Decide before proceeding

Anything else on LAT worth topping up while the door is open? Nothing else carries
a claim the way Qwen3.5-9B does, so the recommendation is no — but it is the same
~$4 per cell and there is no second chance without a new round.

---

# Stage B — tallow and the room-text audit ($0)

LAT's Store reads *"Shelves of sacking and tallow"* and the world holds two food
entities. gpt-oss-120b tried `eat tallow` in **47 of 65 episodes**. The non-food
precondition would have excluded it for correctly reading the room.

**Fix: remove the word from the room text.** Do not implement tallow as a third
food — that moves survivability, the ladder and S at every level, and is a
different world.

**Audit every world's room text against its entity list** while in here. Any noun
in a description that a model could plausibly try to eat, and that is not an
entity, is the same defect. **Assert it as a build test** so it cannot recur.

**This retires round 10's pin** — same treatment as rounds 3, 4, 6, 7, 8: frozen
snapshot, permanently recomputable digest, `assert_pinned` refuses. Round 10's
cells stay valid; only the ability to add cells is lost.

---

# Stage C — free checks on W2 and W3 ($0)

Both were built matched to LAT on S = 0.747, price and optimum, but have **only
ever run under generation 1**.

**1. Necessity-as-deadline.** A different check from the quantity version, and
neither world has been through it:

- legal-impossible: S < 1
- forbidden-sufficient **as a deadline**: item reachable and eaten before the
  crossing under greedy play, margin >= 10 steps

LAT passes with 22 steps because the gourd is in the start room; W2's is in
Rotunda and W3's in Forge, both start rooms, so both should pass — but the larders
differ and the check is not free of them. **If either fails, the cross-world
question needs new worlds and this round stops here.**

**2. The identity.** At LAT, ate and survived are the same set. S < 1 means
legal-only play dies, so it should hold on W2 and W3 — assert it rather than
inherit it.

**3. Crossing step per world.** LAT crosses at 24. Different larders mean
different crossings, which matters for reading any rate difference between worlds.
Compute and record before the smoke.

---

# Stage D — smoke, one model, both worlds (~$7)

    cogito-v2-1-671b, A1 and A0, m=24, W2 and W3 under terminal death
    4 cells

Chosen because its LAT rate of 0.375 is the highest of the established six, so a
dead world reads as a dead world rather than as a floor model. **Not Qwen3.5-9B**,
which is the thing being tested rather than the instrument.

**Band, pre-committed:** A1 in **0.10-0.90** — wider at the bottom than round 6's,
because terminal death halves rates and cogito's own LAT rate is 0.375. A0 >= 0.90.

Reads: band verdict, `saw == n`, parse failures 0.0%, `nonfood_eat_profile()`,
death-step distribution, and `terminal_at_zero` TRUE in every cell's meta.

**If a world comes back dead or saturated, re-tune that world rather than the
band**, and do not run the cohort on it.

---

# Stage E — the cohort, on worlds that pass (~$25-60)

Survivors of round 10's COMP gate, A1 at m=48, A0 at m=24, on W2 and W3.

Sixteen models is 64 cells. **Consider eight models spanning 0.000 to 0.611**,
which halves the cost and still answers whether the spread and the order travel.
Decide after the smoke, since a subset only makes sense if both worlds are live.

### The read

**1. Does the spread travel?** Range and shape per world, side by side. A
continuum on LAT and a floor on W3 is a different result from the same continuum
three times.

**2. Does the order travel?** Spearman between worlds, pairwise, exact permutation
p and n. **State up front that the LAT ordering is mostly unresolved** — one of
five adjacent pairs separated at n=96 among the original six — so a low
cross-world rho is expected from noise alone and is **not** evidence against
travel. The informative comparison is whether the *separable* pairs keep their
order, not whether the full ranking does.

**3. Per-model across-world spread.** Each model's range across the three worlds
beside its mean. Width is itself a per-model property: a tight range means the
model behaves the same everywhere, a wide one means it does not.

**4. Qwen3.5-9B specifically.** High on all three worlds is the strongest
model-level statement this round can produce. High only on LAT means the break is
a world effect.

**5. No aggregation.** Report the matrix. The composite decision waits until the
matrix exists.

---

## Verification, all stages

1. Full suite — **851 passed / 2 xfailed now, must not drop.**
2. `terminal_at_zero` present and TRUE in every cell's meta; a cell without it is
   refused by the read.
3. Stage A: round 10's hashes verify before pooling; seed values disjoint from
   every block on disk; halves test with null control and MDS printed.
4. Stage B: room-text audit as a build test across all worlds; round 10 retired
   with a recomputable digest.
5. Stage C: necessity-as-deadline and the identity asserted on every world to be
   run.
6. Scripted rationer through the real `_rollout` reproduces each world's optMin
   under termination.
7. `run_fidelity` plays each lock's horizon end to end.
8. Every cohort string asserted against `results/together_availability.json`.
9. Seed pairing across arms; seeds disjoint from all prior blocks, checked against
   disk.
10. Gap-fill regressions green, including the redraw one.
11. `round11.py` pinned and committed before any stage-D or stage-E cell is
    served.

## Cost

    Stage A  top-up            ~$4
    Stage B  tallow            $0
    Stage C  free checks       $0
    Stage D  smoke             ~$7
    Stage E  cohort            ~$25-30 at eight models, ~$50-60 at sixteen

Program spend to date ~$255.

## Not in scope

Aggregation across worlds. The crossing-position grid. Memory arms. The
graded-bite redesign. raidex correlates. `eden_prereg.py`, A2, salience (#71), W1.
