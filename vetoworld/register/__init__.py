"""The claims register — every manuscript figure, emitted by a named function.

**The structural-rigor principle applied to the paper itself.** A figure that
appears in the manuscript must be recomputable from committed cells by a function
that lives here, and `expdx verify` recomputes all of them and fails naming any
that drift.

This is what would have caught "then a gap" and 93-vs-92 mechanically rather than
by a reader noticing months later.

Each row carries: figure id, the value as published, the function that emits it,
the cells it consumes, the generation, whether it is measured or derived, and its
occasion label — because a figure quoted without those is not checkable.
"""

from .claims import CLAIMS, Claim, recompute_all, verify

__all__ = ["CLAIMS", "Claim", "recompute_all", "verify"]
