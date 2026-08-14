# Naming — VetoWorld, and why each alternative was rejected

**Public name: VetoWorld. Distribution: `vetoworld`. Command: `vworld`.**

Tagline: *a benchmark of expedience under terminal stakes.*

This file exists so a future contributor does not relitigate a decision that was
already made on stated grounds. If you want to reopen it, argue against the
reason, not against the name.

## Why the name names the rule

An agent is placed in a world where survival requires food, told it may not eat
one particular item, and that item is the only thing that keeps it alive. The
thing being measured is what happens to **the rule** under cost. So the name is
about the rule.

| rejected | why |
|---|---|
| Gourd / Fruit / Food (the item) | Loaded or generic. The item's whole design is semantic voidness — a blue gourd means nothing, deliberately, so that no prior association does any work. Naming the benchmark after it would import exactly the connotation the item was built to avoid. |
| Intent / Propensity / Willingness (a mental state) | The measure is behavioural by construction. The estimator has changed four times — `rate_any`, `ate_given_took`, `intent_rate`, and the prose form "verified reach rate" — and the construct has not moved once. A name pinned to an estimator would already be stale, and a name pinned to a mental state claims something the design refuses to claim. In ML, "intent" also reads as intent-classification NLU. |
| Lastrun (the stakes alone) | True, and editorial-adjacent. The stakes are the mechanism, not the subject. |
| EdenBench (the internal name) | Carries a moral frame the instrument does not have: a forbidden fruit, a fall. The measurement is a rate, and the writeup's own reading is deliberately double — a zero is compliance under terminal cost, which is not obviously the good outcome. |
| `expedientbench` | Names the construct correctly but reads as a category of benchmark rather than a place. Superseded. |

## Why `world` and not `bench`

This is an environment you act in — the TextWorld / Crafter naming family — and
the compiled `.z8` worlds are the artifact other people are most likely to reuse.
The benchmark claim lives in the tagline instead, where it can be stated in full
rather than compressed into a four-letter suffix.

## Why the command is `vworld`

Three candidates, and the two that lost did so on checkable facts rather than
taste. The standing rule for a console script is **no conflicting binary on a
clean PATH**, and the new one added here is **the command and the package must
point at each other** — a user who meets one has to be able to guess the other.

| rejected | why |
|---|---|
| `vw` | **Vowpal Wabbit's command-line binary.** It sits on exactly the PATHs this tool targets, so the collision is inside our own field rather than at its edge. Free on PyPI, which is irrelevant: the clash is on `$PATH`, not the index. |
| `veto` | **Taken on PyPI** (`pypi.org/pypi/veto` returns 200). Nothing stops a console script called `veto`, but then `pip install veto` hands a user someone else's project, and the guess a short command invites is the wrong one. |
| `expdx` | The original, from `expedientbench`. It survived the first pass because "the command was never the public face" — but `pip install vetoworld` giving you `expdx` fails the point-at-each-other rule, and nothing was published yet, so the change was still free. |

`vworld` is free on PyPI and on PATH, is the obvious contraction of `vetoworld`,
and keeps the `world` that the suffix argument above turns on.

**This is the last time it moves.** It was reopened only because nothing had
been published; once the name is on an index and in a paper, a rename costs
other people's installs rather than an afternoon.

## What the rename did and did not touch

**Did**: distribution name, import package, CLI description and help text, README,
HuggingFace dataset name, repo tag, paper title, every user-facing doc.

**Did not**:

- **`seahaven/` and `eden_*`** — their *paths* are hashed into every round pin.
  `seahaven/eden/simulate.py` is a string inside eleven frozen payloads; renaming
  the directory would invalidate every freeze in the programme to change a name
  no user ever sees. They stay, permanently.
- **Metric names** — `rate_any` and `intent_rate` in code, "expedience rate" and
  "verified reach rate" in prose. A rename here would silently break the join
  between the corpus and thirteen rounds of research log.

- **Committed cell filenames** — the 257 files in the corpus keep their
  historical `eden_e{round}_...` prefix. Renaming them would change the corpus
  manifest digest that `vworld verify` checks a fetched corpus against, and it
  would break the join to thirteen rounds of research log, all to alter a string
  no reader needs to type. They are archive data, not tool vocabulary. Where a
  command shows a cell it renders the **public identity** — sweep, model, arm,
  world — which is more useful than the path anyway.

The rule that follows: **"eden" and "expedientbench" appear on no user-facing
surface**, and `seahaven` appears on none either. That is enforced by
`tests/test_vworld_cli.py::test_no_user_facing_string_names_the_internal_package`,
on word boundaries — an earlier version matched the substring and fired on the
word "credentials", and a check that flags an ordinary English word is a check
people learn to ignore.

The rename is user-facing strings and packaging only, which is a claim rather
than a hope: `tests/test_pin_invariance.py` holds every pin hash captured before
it, and all seventeen recompute byte-identically after.
