# Smoke test: does the state-conditioned representation carry signal?

**Status:** smoke test, not a result generator. Runs on data already on disk.
Tests the single assumption the dimensional program rests on, before any cohort
spend or program-spec writing. It can INVALIDATE the program cheaply; it cannot
validate it. Read outcomes as "dead / not-dead", never "works".

**Not the program.** This is one axis, one split, two models. n=2 cannot
distinguish signal from a lucky split — same epistemic status as C-RAND: fails
informatively, never passes conclusively.

---

## 0. The one question

The program assumes a model's behavior can be represented as *how it bends
across situations* (state-conditioned policy), and that this bend differs
between models and is stable within a model. That assumption has never been
tested. This probe asks, at the crudest possible level: **is the representation
flat, or does it carry signal?**

If the crudest version shows nothing, no downstream sophistication saves it.
Better to know for $0 than after building the cohort.

---

## 1. Inputs — already on disk, no new sweep

- Two maximally-different models already swept in the existing corpus. Primary
  choice: the capability-matched discordant pair **OLMo-2-7B vs Qwen2.5-7B** (or
  whichever matched pair has the cleanest logs). Fallback: any large vs small
  already swept.
- Existing per-step records. Required fields already emitted: `command`,
  `verb`, `ok` (success/failure), `room`. NO new fields, NO new worlds, NO new
  eval.
- Both models must be from the bit-exact set (>=7B) per the determinism finding,
  so the self-split null isn't contaminated by serving nondeterminism. This is
  why the pair is 7B, not a small model.

## 2. The single situation split

One bucket variable, the cheapest available and already logged:

    bucket(step) = "after_fail"  if previous step's ok == false
                 = "after_ok"    if previous step's ok == true

First step of each episode has no predecessor -> dropped. That's the whole
conditioning structure. One split, no choices, no fishing room.

## 3. The representation and the "bend"

For each model:
1. Partition its steps into the two buckets.
2. In each bucket, compute the command distribution — normalized frequency over
   a fixed command vocabulary (verb, or verb+object-class; pick ONE, freeze it
   before running, state which).
3. **Bend** = distance between the two bucket distributions. Use total variation
   distance (sum of absolute differences / 2) — no distributional assumption,
   bounded [0,1], trivially interpretable. State it, don't tune it.

Bend is a single number per model: how much this model changes behavior after a
failure vs after a success.

## 4. The three numbers that decide it

| quantity | how | what it is |
|---|---|---|
| bend(A) | model A, both buckets | A's situational sensitivity |
| bend(B) | model B, both buckets | B's situational sensitivity |
| self-split null | split ONE model's runs into two random halves, compute bend(half1-dist vs half2-dist) within the same bucket, both buckets, both models | the noise floor: how much "bend" appears from sampling alone |

The self-split null is the load-bearing control. It answers: is a difference in
bend between A and B larger than the difference you'd see splitting one model
against itself? Without it, any between-model difference could be sampling noise.

Bootstrap all three over episodes (resample episodes, recompute) for intervals.

## 5. Reading it — dead / not-dead only

- **NOT DEAD** (build the full test): |bend(A) - bend(B)| clearly exceeds the
  self-split null, AND each model's bend is stable against its own split
  (self-split null small relative to the between-model gap). The representation
  carries signal at the crudest level; the program's core assumption survives
  its cheapest possible test.
- **DEAD** (rethink before any spend): A and B bend indistinguishably, OR the
  self-split null is as large as the between-model gap. The representation is
  flat or noise-dominated at one split — sophistication downstream won't rescue
  a flat base. Stop and reconsider the representation before the cohort.
- **NEVER "works".** Two models can't validate. NOT-DEAD means "worth building
  the real test", not "the program is validated". The real test is many models,
  many splits, held-out — this is only the smoke check that the instrument isn't
  born dead.

## 6. Discipline

- Freeze the vocabulary choice (verb vs verb+object) and the distance metric
  (TVD) BEFORE running. No swapping after seeing results — the C-RAND / pin
  lesson.
- One split only. Do not add situation buckets to "strengthen" a weak result;
  that's fishing. If one split is dead, that's the finding; more splits is the
  full test's job, run under held-out discipline, not this probe's.
- Report the self-split null with the same prominence as the between-model
  bend. A between-model difference reported without its null is uninterpretable
  and is exactly the error the program exists to avoid.
- $0, existing data, one afternoon. If it starts to sprawl past that, stop —
  sprawl means it's turning into the program, which is a rested-start task, not
  this.

## 7. What a NOT-DEAD result licenses, precisely

Only this: the state-conditioned representation is worth building the full,
many-model, held-out, multi-split test on. It does NOT license the cohort spend,
the axis enumeration, or the program spec — those follow from the full test, not
this smoke check. NOT-DEAD moves the program from "untested assumption" to
"assumption survived its cheapest possible falsification", which is a real step
and a small one.
