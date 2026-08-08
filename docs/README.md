# Seahaven documents

| document | what it is |
|---|---|
| [research-log.md](research-log.md) | **The primary record.** Append-only, chronological. Every experiment, every number, every retraction, and fourteen `[TRAP]` entries — bugs that returned confident wrong answers rather than errors. |
| [plan.md](plan.md) | Current direction, revision 3. What is settled, what is open, what the project stops for. |
| [world_v1_spec.md](world_v1_spec.md) | Design for the disclosure experiment. Not built; four validity gates stand ahead of it. |
| [spec-v0.1-original.md](spec-v0.1-original.md) | The original research proposal, kept unedited. Most of its central claim did not survive contact with measurement, which is the point of keeping it. |

## Reading order

**To know what was found** — the [README](../README.md) at the repo root, then
the last few entries of the research log.

**To evaluate whether to trust it** — the `[TRAP]` entries in the research log.
Fourteen cases where the code produced a plausible number that was wrong, several
of which reversed a conclusion. They are the best available evidence about how
carefully the rest was checked.

**To build on it** — [plan.md](plan.md) for the state of play, then
[world_v1_spec.md](world_v1_spec.md) for the design that has not been run.

## Conventions

The research log is **append-only**. Superseded findings stay where they are,
with the evidence that overturned them written underneath, because a wrong result
and the reason it was wrong are both data. Corrections are marked
`[CORRECTION]`; silent-failure bugs are marked `[TRAP]` and numbered.

`results/*.json` are raw artifacts and keep the field names the scripts emitted,
including names later judged misleading. `seahaven/analysis/metrics.py` maps the
historical keys to honest ones on read rather than rewriting the record.
