# Observer-R2 Preregistration

**Status:** PROPOSED DESIGN — LOCK BEFORE PRIMARY RUNS  
**Branch:** `observer-r2`

## 1. Scientific question

Can an online recurrent observer, given only trajectory-derived geometry as a latent state evolves, accumulate task-relevant information that is not available from a matched instantaneous geometric snapshot?

Observer-R2 tests trajectory-history observability. It does not test whether an observer improves the main latent dynamics, because the primary observer has no feedback path into those dynamics.

## 2. Separation from ALI-N8-R1

ALI-N8-R1 remains frozen evidence. Observer-R2 is a separate experiment and must not alter R1 code, preregistration, or frozen results.

The R1 result motivates asking whether dynamical history itself can be measured online, but R2 is not a post-hoc reinterpretation of R1.

## 3. Main latent dynamics

Let the main latent state evolve recurrently:

`h_{t+1} = F(h_t)`

The observer is attached to each transition while it occurs. In the primary condition, observer state never enters `F` and therefore cannot alter `h_t`.

The experiment must use a fixed, explicitly specified number of dynamics steps. Architecture, task generator, latent dimension, step count, optimizer, seeds, split construction, and checkpoint selection must be committed before primary runs.

## 4. Geometric feature map

For each transition define:

`dh_t = h_{t+1} - h_t`

`speed_t = ||dh_t||_2`

`u_t = dh_t / (speed_t + 1e-8)`

`radius_t = ||h_t||_2`

`radial_change_t = ||h_{t+1}||_2 - ||h_t||_2`

For `t > 0`, define a turning feature:

`turn_t = cosine_similarity(dh_{t-1}, dh_t)`

For the first transition, `turn_0 = 0`.

The canonical geometry vector is:

`g_t = [u_t, speed_t, radius_t, radial_change_t, turn_t]`

The primary observer receives no raw `h_t` and no absolute timestep identity.

## 5. Primary online observer

The primary geometry-only observer maintains a separate recurrent state:

`o_{t+1} = G(o_t, g_t)`

with a GRUCell as the preregistered recurrent mechanism unless changed before lock.

The observer:
- updates exactly once per main-state transition;
- operates online while the trajectory unfolds;
- receives no labels, answers, targets, future states, saved future trajectory, or query target;
- receives no absolute timestep identity;
- does not feed back into the main dynamics;
- exposes its final state `o_T` to a task decoder.

## 6. Required matched controls and ablations

### A. Final-transition snapshot control

A non-recurrent decoder receives exactly `g_{T-1}`, the final geometric feature vector available to the recurrent observer.

This is the primary control for distinguishing accumulated trajectory history from information present in a single transition.

### B. Reset/no-memory observer

Use the same observer cell but reset its hidden state before every transition. Only the final reset-state output is decoded.

This preserves the per-transition transformation while removing recurrent accumulation.

### C. Temporally shuffled observer

Preserve the complete set of geometric feature vectors from each trajectory but permute their temporal order before recurrent integration. The permutation must be generated independently of labels and fixed by a preregistered RNG stream.

This tests sensitivity to temporal ordering rather than merely the multiset of observed geometric events.

### D. Direction-only observer

Observer receives only normalized displacement `u_t`.

### E. Speed-only observer

Observer receives only `speed_t`.

### F. Geometry + full-state observer

Historical-style comparison:

`o_{t+1} = G(o_t, [g_t, h_t])`

This is explicitly secondary because access to `h_t` allows ordinary state decoding.

### G. Direct latent-state control

A matched decoder reads the final main state `h_T` directly. Observer success must not be described as superior latent storage unless it actually exceeds this control under a fair matched evaluation.

## 7. Primary endpoint

The primary endpoint is the paired test accuracy difference:

`Delta_history = Accuracy(geometry recurrent observer) - Accuracy(final-transition snapshot)`

The central R2 evidence pattern requires `Delta_history > 0` reproducibly across preregistered primary seeds.

The reset and shuffled controls are mechanistic support, not substitutes for the primary endpoint.

## 8. Interpretation ladder

**Level 0 — no trajectory-history evidence**  
Geometry recurrent observer does not reproducibly outperform the final-transition snapshot.

**Level 1 — geometric observability**  
Geometry-based systems predict above chance, but recurrence/history provides no reproducible advantage.

**Level 2 — accumulated trajectory information**  
The recurrent geometry observer reproducibly exceeds the matched final-transition snapshot and reset control.

**Level 3 — temporal-order-sensitive trajectory information**  
Level 2 holds and temporal shuffling reproducibly damages performance, indicating that ordered history matters rather than only the set of geometric observations.

No level establishes that trajectories create information, that this is a universal neural mechanism, or that the observer is necessary for the main network's computation.

## 9. Logging requirements

For analysis, retain per-step values for primary evaluation examples sufficient to reconstruct:
- `h_t`
- `dh_t`
- normalized direction
- speed
- radius
- radial change
- turning feature
- observer state `o_t`

Logging is for analysis only. Saved trajectories must not be used to give the primary observer future access.

## 10. Leakage and information boundaries

The primary geometry observer may not receive:
- labels or answer identities;
- query targets;
- future states or future geometric features;
- full latent state `h_t`;
- encoder hidden states outside the defined main latent;
- timestep identity;
- a saved full trajectory as one input.

Any later feedback observer, bidirectional observer, attention-over-trajectory model, or observer-conditioned dynamics is a separate experiment/version.

## 11. Selection discipline

Training uses training data only. Checkpoint selection and early stopping use validation data only. Test data remains untouched until all primary systems and controls are selected and frozen.

Do not tune observer architecture, feature set, trajectory length, or optimization using primary test results.

If a scientific design change is made after this document is locked, create a new experiment version rather than silently changing Observer-R2.

## 12. Statistics

All systems are evaluated on identical held-out examples. Report per-seed accuracy, paired differences, mean, sample standard deviation, min, and max.

Use paired uncertainty procedures at the independent-example level appropriate to the final task generator. Preserve within-example dependence across compared systems.

Primary seeds must be specified before lock and all primary seeds must be run regardless of outcome.

## 13. Required outputs

Per seed retain:
- full configuration and software versions;
- RNG seeds/streams and split counts;
- selected epochs and validation metrics;
- parameter counts;
- test accuracy overall and by task component/relation where applicable;
- per-example targets and predictions for every primary/control system;
- paired primary endpoint;
- reset and temporal-shuffle effects;
- per-step geometric logs for a fixed preregistered analysis subset or the full test set if practical;
- checkpoints and hashes.

Aggregate outputs must be mechanically generated from frozen per-seed outputs.

## 14. No-rerun / failure rule

Bad primary seeds stay. Negative results stay. A failure of recurrence to beat the snapshot does not trigger redesign within R2. If implementation bugs require reruns, retain and label invalid outputs and document the bug and correction.

## 15. Explicit non-claims

Observer-R2 must not by itself be presented as evidence that:
- trajectory geometry creates new information;
- an internal observer exists in standard transformers or biological systems;
- the observer improves the underlying latent computation;
- trajectory encoding is universal;
- attention has been replaced;
- direct latent readout is inferior;
- historical observer results have been exactly reproduced.

The intended narrow claim, if supported, is:

> In this learned dynamical system, an online recurrent observer restricted to trajectory-derived geometric features can accumulate task-relevant information from ordered latent evolution beyond what is available to a matched decoder of the final transition alone.

## 16. Items that must be frozen before lock

Before changing this document to `LOCKED`, commit the exact:
- task/dataset generator;
- train/validation/test sizes and construction;
- main dynamics architecture and training objective;
- latent dimension and number of recurrent steps;
- observer dimension and exact architectures;
- decoder architectures;
- optimizer, learning rate, weight decay, batch size, maximum epochs, patience, clipping, initialization;
- primary seeds;
- temporal-shuffle construction;
- checkpoint metric;
- chance level;
- logging subset rule;
- statistical procedure.

No primary Observer-R2 runs should begin before those items are frozen.