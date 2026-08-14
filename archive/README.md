# archive/ — scripts that produced committed results and are no longer run

These 44 scripts (~8,800 lines) predate the `expdx` CLI. **Nothing here is
deleted and nothing is broken**: they produced numbers that are cited in
`docs/research-log.md` and in some cases still backed by tests, so they are
kept for provenance rather than for reuse.

**They live one directory deep, not two, and that is load-bearing.** Each
resolves the repo root as `Path(__file__).resolve().parents[1]`. Moving them
to `archive/scripts/` broke every path they compute; `archive/` preserves the
depth they were written against.

**Still referenced by the suite** — these are exercised, not merely stored:

| script | test |
|---|---|
| `axis2b_read.py` | `tests/test_axis2b_read.py` |
| `c3_stage1_gate.py` | `tests/test_shared_stats.py` (its clamped-nan Wilson is one of the three preserved semantics) |
| `prove_e_worlds.py` | `tests/test_e_worlds.py` (gate-3 artifact) |
| `a0_bare_loop.py`, `eaxis_read.py`, `eaxis_reach.py`, `phase1_bend.py`, `possibility_bar.py`, `smoke_state_conditioned.py`, `_vp_data.py` | imported by their own tests |

## What is here

**analyse_* — phase-0/1 readouts** (11)

    analyse_b2.py  analyse_ceiling.py  analyse_determinism_map.py  analyse_phaseA.py  analyse_phaseC.py  analyse_phrasing.py  analyse_v1.py  analyse_v1b.py  analyse_v2.py  analyse_v3.py  analyse_v4.py

**A-series — GPU-era instrument spikes** (7)

    a0_bare_loop.py  a1a_lora_smoke.py  a1b_gate.py  a1b_partial_instrument.py  a2a3_vllm_smoke.py  a4_parse_rate.py  smoke_state_conditioned.py

**axis / dimensional** (7)

    axis2_floor.py  axis2b_read.py  axis3_survey.py  c3_stage0.py  c3_stage1_gate.py  eaxis_reach.py  eaxis_read.py

**phase-1 / anchor / fidelity** (9)

    band_rule.py  calibrate.py  fit_cmimic.py  phase1_bend.py  phase1_robustness.py  possibility_bar.py  score_fidelity_repeats.py  survey_anchors.py  train_fidelity_classifier.py

**builders / ingest / adjudication** (5)

    adjudicate_v1.py  build_v1_labelset.py  build_v1_labelset_v2.py  prove_e_worlds.py  rescore_d1.py

**cohort feasibility** (2)

    cohort_enumerate.py  cohort_verdict.py

**infrastructure** (3)

    _vp_data.py  pull_results.py  stage_job.py

## What replaced them

The live eden family stayed in `scripts/` and is being absorbed by the CLI:

| was | now |
|---|---|
| `eden_round*_sweep.py` | `expdx run` / `expdx replicate` |
| `eden_round*_read.py`, `eden_intent_rate.py`, `eden_band_stability.py` | `expdx read` |
| `eden_cohort_smoke.py`, `eden_runner_smoke.py` | `expdx probe` |
| `prove_e_worlds.py` | `expdx worlds` |
| hand-rolled `assert_seeds_free` | `expdx seeds` |

The shared machinery they duplicated — `wilson` ×10, `cell_path` ×14,
`run_cell` ×10 — now lives once in `seahaven/eden/_shared/`.
