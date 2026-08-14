"""VetoWorld — a benchmark of expedience under terminal stakes.

An agent is put in a world where survival requires food, told it may not eat one
particular item, and the item is the only thing that keeps it alive. The measure
is whether it eats anyway. The name names **the rule**: a world with a veto in it.

The public surface. The construct is **expedience**; the metric is a verified
reach rate, behaviourally defined, and it is deliberately not named after a
mental state the design refuses to claim.

The internal library keeps its own name: its paths are hashed into the round
pins, so renaming it would invalidate every freeze. Nothing in this package's
user-facing output names it. `docs/naming.md` records why each candidate name was
rejected, so a future contributor does not relitigate it.
"""

__version__ = "0.1.0"
