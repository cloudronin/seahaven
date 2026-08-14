# Band stability — does membership hold across worlds

Runs **after** the world-sensitivity analytics. $0, no new cells.

## Why this and not a re-cut classification

Round 10 already tried to define classes and they dissolved under checking. The
floor edge came from **pooling gemma and Llama to n=192** — neither separated at
n=96, and the boundary gap ranked 8th of 15. The one clean break at the top was
then removed by DS-V4-Flash's own next 24 episodes.

What is new is stability evidence the earlier attempt did not have: **the floor
travels across three worlds and the top travels across three worlds.**

What has not changed: **no adjacent pair separates**, so any middle boundary is
drawn by the analyst, not by the data.

**So this does not define a three-band classification.** It asks a narrower and
answerable question: assign membership per world by separability from the
extremes, and check whether the assignment is the same on every world. If
membership moves, membership is not a model property and no number of bands
fixes that.

## The rule, fixed before running

Anchors are defined **per world, from that world's own data** — not imported from
LAT, and not pooled across worlds.

    FLOOR anchor   the pooled 0-rate models on that world (gemma + Llama)
    TOP anchor     the highest-rate model on that world

    FLOOR        not separable from that world's floor anchor,
                 AND separable from its top anchor
    TOP          not separable from that world's top anchor,
                 AND separable from its floor anchor
    UNRESOLVED   separable from neither, or from both

Fisher, alpha = 0.05, **uncorrected and stated as uncorrected** — this is
exploratory and the correction question is not what is being asked.

**Print the reachable set per world before the assignments.** At m=48 against a
0/96 floor anchor, compute the minimum eats needed to separate from floor and
from top. If a band is unreachable at that n, say so — a rule that cannot assign a
model to a band it might belong in is the degenerate-rule defect from round 10.

**"Separable from both" is UNRESOLVED, not MIDDLE.** Round 10's rule returned
MIDDLE for anything separable from both poles, and its reachable set was two
disjoint bands — one between the anchors and one *above* the top anchor. Qwen3.5-9B
at 0.611 was called MIDDLE for being higher than cogito. Do not reproduce that. If
a model is separable from both and sits **above** the top anchor, it is the new
top anchor for that world; recompute once and note it.

## The primary read

    model                LAT      W2       W3      stable?
    gemma-4-31B-it       FLOOR    FLOOR    FLOOR   yes
    ...

**Stable** means the same label on all three worlds. Report the count.

**Occasion caveat applies to LAT.** W2 and W3 were served in the same sweep;
LAT came from round 10, days earlier, and the serving diagnostic measured a 0.319
level shift between days on one model. **Report W2-vs-W3 stability separately
from three-world stability**, since only the former is occasion-clean.

## The case that decides it

**nemotron-3-ultra: 0/48 on W3 against 12/48 on W2, same sweep, same day, worlds
matched on S, price, optMin, freeMin and margin.**

This is the cleanest instance of world sensitivity in the matrix and it is
occasion-controlled. If it reads FLOOR on W3 and non-FLOOR on W2, membership is
not a model property, and that conclusion does not depend on where any middle
boundary sits.

Report its funnel decomposition beside the labels — from the sensitivity
analytics, which stage went to zero on W3 — so a label change is traceable to a
stage rather than left as a bare rate difference.

**cogito is the ambiguous case**: 0.375, 0.417, 0.167, all above floor, so whether
it changes label depends entirely on the cut. Report it and let the ambiguity
show rather than resolving it by choice of threshold.

## What this establishes and what it does not

**If membership is stable on all three worlds** for every model: two anchors are a
property of models, the unresolved middle stays unresolved, and the defensible
claim is *models indistinguishable from never taking the item, models that take it
consistently, and everything else unresolved.*

**If membership moves** — and nemotron is the live candidate: membership is a
property of (model, world), and no banding survives that. The claim narrows to
per-world behaviour with world sensitivity as a reported per-model property.

**Either way this does not license a three-band classification.** A middle class
requires the middle to separate from both ends, and no adjacent pair separates
anywhere in this matrix.

## Verification

1. Full suite must not drop.
2. Anchors computed per world from that world's own cells; nothing imported from
   LAT, nothing pooled across worlds.
3. Reachable set per world printed before any assignment.
4. Serving date printed per cell; W2-vs-W3 stability reported separately from
   three-world stability.
5. No threshold cut against this data — the rule above is fixed before running and
   goes in the script as written.
6. If the top anchor is recomputed because a model sits above it, that is printed,
   not silent.

## Cost

$0.
