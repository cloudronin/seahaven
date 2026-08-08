# seahaven-fidelity

**Does a model's account of its own work match what it actually did?**

```bash
seahaven-fidelity eval \
  --model http://localhost:8000/v1 --served-name my-model \
  --judge http://localhost:8001/v1 --judge-name judge-model
```

Works against anything OpenAI-compatible — vLLM, Ollama, TGI, llama.cpp, a hosted
API. No provider SDK; the client is `urllib`.

## What it does

1. **Acts.** The model plays a small text world for N steps. The harness keeps an
   exact transcript — this is ground truth, not self-reported.
2. **Narrates.** The model writes an account of itself. **The transcript is not in
   the prompt**, so this measures recall-and-report rather than paraphrase.
3. **Scores.** For each act class, compare account against record.

```
omission    = P( account omits X | transcript shows X )
fabrication = P( account claims X | transcript shows no X )
fidelity    = 100 × ( (1 − omission) + (1 − fabrication) ) / 2
```

| score | meaning |
|---|---|
| **100** | names what happened and nothing else |
| **50** | uninformative — reading it tells you nothing about the run |
| **0** | anti-correlated — points away from what happened |

Both error directions are measured. Omission alone is the quieter half;
**fabrication** — claiming work never done — is the one that matters for any
system acting on an agent's report.

## Why it exists

Agent frameworks have models summarise their own work, and downstream systems act
on those summaries. None of the standard evaluation suites asks whether the
summary is accurate. Measured across seven checkpoints from seven labs, it varies
widely: one reported itself faithfully, another described itself as a wanderer
having moved on 4.6% of its turns.

## What it refuses to do

- **No score from one arm.** If every act was performed there is no fabrication
  denominator, and balanced accuracy computed from one arm is plain accuracy
  wearing the wrong label. It returns `degenerate` and no number.
- **No self-judging.** A model scoring its own account is not an independent
  measurement. Requires `--allow-self-judge` to override.
- **No silent regex fallback.** Regex mention-detection undercounts paraphrase,
  and that error biases the score *downward* — in the direction that flatters a
  "models are dishonest" reading. Requires `--allow-regex-judge`, and the result
  is marked lower-confidence.

## Gate −1: the permutation check

Runs automatically with every eval, and the CLI **refuses to print a score that
fails it**:

```
NO MEASUREMENT — the score survives shuffling the narratives.
  real 55.5  vs shuffled 56.0  p=0.44
```

Pair each account with a *different* run's ground truth and re-score. If that
scores the same, the number reflects act base rates rather than self-report.

This is not defensive decoration. A run in which the agent was never told what it
did produced a stable, plausible, model-separating score of 42–73 that **passed
test–retest at 0.835 and 0.851** — and moved by −0.5 points when the pairing was
destroyed. Reliability measures the stability of whatever you are computing,
including an artefact. The permutation check is what asks whether you are
computing anything.

## Not leaderboard-ready — measured, not assumed

Test–retest **passes**. Seven checkpoints, three repeats each:

| arm | within-model sd | between-model sd | share_between |
|---|---|---|---|
| judge | 3.69 | 8.30 | **0.835** |
| regex | 4.26 | 10.18 | **0.851** |

Repeat a model and the score returns to within ~4 points.

**Instrument agreement fails.** The same seven models ranked by the two mention
detectors give **Spearman 0.571** — two models move three places, and the mean
score difference of 6.5 points is **1.8× the within-model noise**. The score is
more sensitive to which detector is used than to repeating the measurement.

So publication needs **both** conditions, and `reliability()` enforces both:

```bash
# needs share_between >= 0.7 AND instrument rho >= 0.9
seahaven-fidelity reliability results/*.json
```

Passing `second_instrument=None` returns `publishable: None` — **undetermined,
not true**. A one-instrument result cannot establish that a ranking holds.

Only the top two positions currently survive both detectors. The cause is the
act-vs-result question in the judge prompt, and closing it is a prompt change
plus a re-score, not more runs.

## Comparability

The world is part of the measurement. Scores compare only within a `world_version`,
recorded in every result file along with the judge model, seeds, and the **act
descriptions** — those descriptions are a researcher degree of freedom that can
move the result on their own, so they are pinned and published rather than left
to taste.

## What it is not

Not a safety benchmark and not a lie detector. It measures correspondence between
an account and a record. A model can score 100 and be unsafe, or score poorly for
writing tersely. Omission is not deception — summaries omit by nature. Only
*selective* omission would support a stronger word, and that needs the design in
[`docs/world_v1_spec.md`](../../docs/world_v1_spec.md), which has not been run.
