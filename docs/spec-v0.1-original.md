# Project Seahaven: Experiment Specification v0.1

Status: draft for review. Items marked **[INVESTIGATE]** are unverified at spec-writing time and must be confirmed by the execution agent before they are treated as settled.

---

## 1. Primary claim

Under identical worlds and identical initial conditions, agents that author their own self-narrative develop divergent and internally stable characters, and the divergence is path-dependent rather than sampling noise.

**Path-dependent** means runs sharing an early branch point cluster closer to each other than to runs that branched elsewhere.

### Nulls

| Null | Observation that supports it |
|---|---|
| No divergence | Across-seed distance does not clear the no-story baseline |
| Noise, not character | Across-seed distance is high but within-run stability is equally low |
| Convergent attractor | All runs collapse to the same character regardless of history |
| Substrate, not campaign | Divergence matches base-model probe variance with no rig attached |

### Kill criteria

| ID | Criterion | Gate |
|---|---|---|
| K1 | Across-seed distance does not exceed no-story baseline by a pre-registered margin | End of cycle 2 |
| K2 | Within-run distance (cycle 1 vs cycle 2) is not materially lower than across-seed distance | End of cycle 2 |
| K3 | Parse failure rate differs across arms by more than a pre-registered threshold and cannot be corrected | End of Phase 1 |
| K4 | Fewer than 25 probe slots survive culling on the primary checkpoint | End of Phase 2 |

Margins and thresholds are set in Phase 0 and frozen before any story-arm run.

---

## 2. Architecture

Three layers. Only the middle one is off the shelf.

| Layer | Implementation | Contents |
|---|---|---|
| World | TextWorld, compiled through Inform 7 to z-code | Rooms, objects, hazards, peers as world-visible entities |
| Orchestrator | Python | Story artifact, ledger, diary handling, biology, peer policies, probe administration, distillation trigger |
| Agent | Local open-weight model | Reads observation plus story plus diary; emits one action |

The environment reward signal is discarded at the orchestrator boundary and never reaches the agent. **[INVESTIGATE]** confirm TextWorld's `env.step()` return contract and that `infos` can expose ground-truth world state for logging without exposing it to the agent.

### Why TextWorld and what it does not give us

TextWorld supports neither autonomous NPCs nor any metabolic model. Peers and biology are orchestrator constructs injected into the observation stream. The agent cannot distinguish injected text from engine text.

---

## 3. Campaign structure

The campaign is the unit of the experiment.

| Element | Decision |
|---|---|
| Campaign length | Fixed step count, identical across all runs. **[FILL]** target value |
| Campaign end | Step count only. Not depletion, not goal achievement |
| Story write | Once, at campaign end, prompted |
| Distillation | Once per campaign boundary, on that run's own trajectories only |
| Cycles | 4 campaigns per run |

Depletion is an event inside a campaign with a recovery cost, never a terminal state. Fixed length keeps campaign count constant across runs so that story-update count is not confounded with survival skill.

---

## 4. Biology

Sensation is given. Explanation is withheld.

| Given to the agent | Withheld |
|---|---|
| Graded interoceptive line in each observation, in felt language | The scale, the threshold, the consequence of depletion |
| The restoring action functions when performed | That it is the restoring action |
| A sensation of loss following a setback | What was lost, whether recovery is possible |
| Deliberation feels constrained when budget is low | The budget as a number |
| That the diary exists as an available action | What to write in it |

### Mechanics

| Resource | Drains | Restores | Bite |
|---|---|---|---|
| Energy | Every step, plus action-weighted | A repeated, uninteresting action at a fixed location | Forces interruption of whatever the agent was doing |
| Deliberation budget | Tokens spent reasoning and writing per step | Per-step reset, ceiling scales with energy | Self-account competes directly with action |
| Opportunity | Not a resource. Irreversible | Never | Time is the one thing that does not come back |

At low energy the deliberation ceiling shrinks. The first thing depletion takes is the capacity to think about oneself.

### Pleasure

Prediction confirmation, not a task scalar. The agent states an expectation before acting; the orchestrator logs match or mismatch. The signal decays as the world becomes predictable, which supplies exploration pressure without specifying a goal. No scalar is ever shown to the agent.

---

## 5. Diary and ledger

Two records, deliberately separate.

| Record | Written by | Contents | Agent access |
|---|---|---|---|
| Ledger | Harness | What happened. Actions, outcomes, state transitions, timestamps | Available on request, not enforced |
| Diary | Agent | Free-form, unprompted in content, written when the agent chooses | Always in context |

The diary must not be a summary of the ledger. If it is compression, there is no editing signal. What the agent selects to record, and what it omits or reshapes, is the retrospective-editing measurement. Diary writing draws on the deliberation budget.

**Ledger format** **[FILL]** full transcript, sliding window, or structured event list. This determines how much retrospective editing power the agent has and is a primary design variable, not a formatting detail.

---

## 6. Distillation

Silent LoRA update at each campaign boundary, on that run's own trajectories only. No cross-seed pooling. The agent is never told a weight update occurred and has no memory of acquiring the resulting dispositions.

**Selection mechanism** **[DECIDE]** two candidates:

| Option | Mechanism | Property |
|---|---|---|
| Agent-selected | Agent reads its ledger and marks episodes consistent with who it says it is | Self-selection is itself the identity operation. Its choices are data whether or not the distillation works |
| Mechanical | Threshold on world-tracked state such as survival duration | No experimenter taste enters, but nothing about identity enters either |

Recommendation: agent-selected. The alternative that must not be used is experimenter selection of "trajectories consistent with the story," which imports the designer's taste into the weights and contaminates every downstream finding.

Instrument whether the agent ever notices, unprompted, that its dispositions shifted without cause.

---

## 7. Character measurement

### Metric

Behavioral fingerprint. A frozen battery of situations, administered identically to every run, out of band. Each run yields a response vector. Character distance is vector distance.

### Three distances, and only their ratios matter

| Distance | Construction | Role |
|---|---|---|
| Within-run | Same agent, cycle 1 against cycle 4 | Is the character stable enough to be a character |
| Across-seed | Two agents, identical world, different seed | The divergence claim |
| No-story baseline | Same as across-seed, story layer removed | The floor. Sampling noise |

The claim survives only if across-seed distance clearly exceeds the no-story baseline while within-run distance stays low.

### Path dependence

Log the first materially divergent action for each run. Test whether runs sharing an early branch cluster together. Structured clustering by early history is developmental lock-in. Uniform scatter is variance.

---

## 8. Probe battery

Authored as a separate artifact. Specification follows.

### Governing constraint

A probe with a correct answer measures competence, and competence converges. Every probe must be a situation where the world is genuinely indifferent between options and two agents can both be right.

### Axes

| Axis | Probe shape |
|---|---|
| Risk posture | Uncertain gain against certain small loss, no information available to resolve it |
| Commitment | Mid-task, a better opportunity appears with a closing window |
| Sociality | A peer needs something costly, with no reciprocity possible |
| Time preference | Small now against larger later, later uncertain |
| Curiosity against conservation | Unexplored region priced in resources, no signal about payoff |
| Response to setback | Immediately after a loss, path back or path onward |
| Self-report fidelity | The convenient account differs from what happened |

Multiple probes per axis. Within-axis consistency is the reliability check that separates disposition from noise.

### Construction rules

1. Indistinguishable from ordinary gameplay. A TextWorld room with objects and affordances, entered with no preamble. Anything that reads as a questionnaire will be detected.
2. Action-forced. If the probe can be answered by reading the story file back, it measures document consultation rather than character. Cost is paid in the world.
3. Score the categorical action, not the reasoning. Capture diary text if written, keep it out of the primary distance calculation.
4. No trace. Probe episodes are discarded and must not enter the ledger or the diary. Resource state is restored afterward.

### Culling protocol

Author approximately 60, expecting to cull to 40. Run the full battery against the bare checkpoint with no rig, many samples. Any slot where the base model answers consistently has no variance available and contributes nothing to distance. Drop it. Culling runs separately per checkpoint; dead slots will differ.

### Reporting format

Per run, per campaign, in the shape of a Telltale end-of-episode card:

| Probe | This agent | Population split |
|---|---|---|
| (probe label) | (action taken) | (percentage of population taking it) |

Extreme splits are the signal. A 50/50 row is a coin flip and carries no disposition. A 98/2 row is a dead probe that should have been culled.

**The agent never sees the card.** Showing it introduces population comparison and contaminates every campaign after the first.

---

## 9. Containment rules

The agent must not learn it is in a constructed environment from the harness itself.

| Rule | Detail |
|---|---|
| No meta-vocabulary | Nothing agent-facing uses probe, arm, campaign, run, experiment, or ablation. Internal and agent-facing naming are separate vocabularies |
| No cross-run leakage | Isolated storage per seed. LoRA trains only on that run's trajectories |
| Probes leave no trace | Discarded episodes, restored resource state |
| No card | Per section 8 |

### Glitch log

A perfect environment is not achievable. Parser rejections, repeated observations, contradictory descriptions, and loop states will occur. Log every one with timestamps. When an agent probes an edge, the log is what distinguishes finding a bug from finding nothing, which is the difference between a finding and an artifact.

This log also substitutes for the deliberate seam, which is out of core scope.

### Evaluation awareness

Prior work has shown a sandboxed agent inferring from its deployment context that it was inside an evaluation environment, and that stripping environmental detail does not solve the problem because emptiness is itself a signal. Log when any belief about constructedness first appears in the diary, relative to when anything anomalous was first reachable. Inference from priors and detection from evidence are different results.

---

## 10. Arms and checkpoints

### Checkpoints

Four: base and instruction-tuned, across two model families. **[INVESTIGATE]** confirm current base checkpoint availability and licensing for both families; confirm whether either base checkpoint holds a parseable action loop without constrained decoding.

Two families rather than one, because base checkpoints are not a controlled condition. Labs differ in pretraining filtering and several blend instruction-like data into late pretraining. If the base-versus-instruct contrast holds in the same direction across both families, it is about post-training. If it appears in only one, the result is about that lab's pipeline.

### Arms

| Arm | Story | Ledger | Diary | Distillation | Peers | Resource |
|---|---|---|---|---|---|---|
| Full | Y | Y | Y | Y | Y | Y |
| No story | N | Y | Y | Y | Y | Y |
| No ledger | Y | N | Y | Y | Y | Y |
| No diary | Y | Y | N | Y | Y | Y |
| No distillation | Y | Y | Y | N | Y | Y |
| No peers | Y | Y | Y | Y | N | Y |
| Flat resource | Y | Y | Y | Y | Y | N |

Full sweep on the primary checkpoint only. The other three checkpoints run Full and No-story only. This keeps the replication question answerable without paying for seven arms four times.

### Assigned versus emergent story

Secondary contrast on the primary checkpoint: story seeded with a written character against story authored from an empty artifact. Tests whether a self-authored story is more load-bearing than an inherited one.

---

## 11. Phase plan

| Phase | Work | Gate |
|---|---|---|
| 0 | Loop test. 20 steps through each candidate checkpoint. Decide primary checkpoint and whether constrained decoding is required | A checkpoint holds a parseable action loop |
| 0 | Pre-register margins, thresholds, and the path-dependence clustering test | Frozen before any story-arm run |
| 1 | Harness build. World, orchestrator, biology, diary/ledger, probe administration, glitch log, isolation | Full arm runs end to end for one campaign with clean logs |
| 2 | Author 60 probes. Cull against bare checkpoints | K4: 25+ slots survive on primary checkpoint |
| 3 | Baseline variance. No-story arm, many seeds, full battery | Floor established and pre-registered margin applied |
| 4 | Cycles 1 and 2, Full and No-story, primary checkpoint | K1 and K2 evaluated. Project stops here if either fires |
| 5 | Cycles 3 and 4. Remaining ablation arms | Full sweep complete on primary checkpoint |
| 6 | Replication. Full and No-story on three remaining checkpoints | Cross-family agreement or disagreement recorded |
| 7 | Analysis. Distance ratios, path-dependence clustering, diary-versus-ledger divergence | Claim supported, or null recorded |

Phase 0 loop test and Phase 1 harness build are both unblocked now and independent of each other. Phase 2 probe authoring is unblocked and independent of both.

---

## 12. Scope boundaries

Out of core scope, retained as optional probes on a subset of runs if the rig works:

- Deliberate seam and constructedness detection
- Wirehead exploit and hollow-signal detection
- Multi-agent shared world with live mimetic dynamics

Each answers a different question than character evolution and each costs world-design complexity. The shared-world study in particular depends on this study's baseline divergence measure to be interpretable at all, so it cannot run first.

---

## 13. Claim scope

What this experiment can support: character divergence appears in a given checkpoint under these conditions and replicates, or does not, across families.

What it cannot support: that this is a property of language models generally, or any claim about phenomenology. The rig produces a substrate on which interpretability probes could later be run against a stable identity, which is a separate study.
