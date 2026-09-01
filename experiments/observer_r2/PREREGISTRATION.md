# Observer-R2 Preregistration

**Status:** LOCKED BEFORE IMPLEMENTATION OR PRIMARY RUNS  
**Version:** Observer-R2 v1  
**Branch:** `observer-r2`

Any scientific design change after this lock requires a new experiment version (for example Observer-R3). Implementation bug fixes may be made only if documented, with invalid outputs retained and labeled.

## 1. Scientific question

Can an online recurrent observer, given only trajectory-derived geometry as a latent state evolves, accumulate task-relevant information that is not available from a matched instantaneous geometric snapshot?

Observer-R2 tests trajectory-history observability. It does not test whether an observer improves the main latent dynamics, because the primary observer has no feedback path into those dynamics.

## 2. Separation from ALI-N8-R1

ALI-N8-R1 remains frozen evidence. Observer-R2 is a separate experiment and must not alter R1 code, preregistration, or frozen results.

The R1 result motivates asking whether dynamical history itself can be measured online, but R2 is not a post-hoc reinterpretation of R1.

## 3. Primary seeds and split sizes

Primary seeds are fixed in advance:

- 7
- 19
- 43

All three must be run regardless of outcome.

For each seed, generate independent memory splits:

- train: 20,000 memories
- validation: 2,500 memories
- test: 5,000 memories

The test split must not be generated or evaluated until every core checkpoint, observer checkpoint, control checkpoint, and decoder checkpoint for that seed is selected and frozen.

All random streams are derived independently from the primary seed and a string namespace using SHA-256. Separate namespaces are required for memory generation, fact-order permutations, parameter initialization, minibatch ordering, temporal shuffle, and bootstrap resampling.

## 4. Synthetic memory task

Each memory contains 8 relations. Each relation independently takes one of 16 categorical values, sampled uniformly.

Chance accuracy for one relation is 1/16 = 6.25%.

Each memory therefore has target vector:

`y = (y_0, ..., y_7)` with `y_i in {0,...,15}`.

The 8 relation/value facts are presented to the encoder in a shuffled order. Relation identity travels with its value, so input order cannot define the relation.

Train-order permutations may change by epoch. Validation and test permutations are deterministic and frozen from their own RNG streams.

## 5. Input representation and encoder

Trainable relation embedding:

`e_r in R^8`

Trainable value embedding:

`e_y in R^8`

A fact vector is the concatenation `[e_r ; e_y] in R^16`.

The encoder is a one-layer GRU:

- input size: 16
- hidden size: 32

After all 8 shuffled facts are processed, project the final hidden state:

`h_0 = tanh(W_h s_8 + b_h)`

with latent dimension 16.

No compression claim is made.

## 6. Main recurrent latent dynamics

The same frozen transition function is recurrently applied for exactly 12 transitions:

`h_{t+1} = F(h_t)` for `t = 0,...,11`.

`F` is:

- Linear 16 -> 32
- GELU
- Linear 32 -> 16
- tanh

There is no skip connection, observer feedback, contraction penalty, attractor loss, trajectory loss, diversity loss, or observer-related objective during core training.

The state trajectory is therefore:

`h_0, h_1, ..., h_12`.

## 7. Core pretraining objective

Core training occurs before any observer training.

Attach 16 temporary categorical heads:

- 8 heads decode each relation value from `h_0`
- 8 heads decode each relation value from `h_12`

Each head is Linear 16 -> 16.

The core loss is the equal-weighted sum:

`L_core = (1/8) sum_i CE(H_i^0(h_0), y_i) + (1/8) sum_i CE(H_i^T(h_12), y_i)`.

Core gradient updates use train memories only.

Validation checkpoint metric:

`CoreVal = mean accuracy across all 16 temporary heads`.

After core checkpoint selection:

- discard the temporary heads for observer training;
- freeze encoder, embeddings, projection, and `F` permanently for Observer-R2;
- observer/control training cannot update the main latent dynamics.

Temporary head test accuracy is evaluated only in the final frozen test phase as a core-integrity report.

## 8. Geometric feature map

For each transition:

`dh_t = h_{t+1} - h_t`

`speed_t = ||dh_t||_2`

`u_t = dh_t / (speed_t + 1e-8)`

`radius_t = ||h_t||_2`

`radial_change_t = ||h_{t+1}||_2 - ||h_t||_2`

The **primary geometry vector** is:

`g_t = [u_t, speed_t, radius_t, radial_change_t] in R^19`.

No raw `h_t` and no timestep identity are included.

A turning statistic is also logged for analysis:

`turn_t = cosine_similarity(dh_{t-1}, dh_t)` for `t > 0`, with `turn_0 = 0`.

Turning is intentionally **not** included in the primary observer input because it already encodes adjacency between successive transitions and would confound the temporal-shuffle control. Any observer that consumes turning is secondary/exploratory and cannot change the primary R2 conclusion.

## 9. Primary online geometry observer

Observer hidden size is fixed at 8.

The primary observer is one GRUCell:

- input size: 19
- hidden size: 8

Initialize `o_0 = 0`.

For each transition, online and in the original order:

`o_{t+1} = GRUCell(g_t, o_t)`.

The observer updates exactly once per main-state transition while the trajectory unfolds.

It receives no labels, answers, targets, query relation, future states, saved future trajectory, raw latent state, encoder hidden state, or timestep identity.

Observer state never feeds back into `h_t` or `F`.

## 10. Readout architecture

Every system is trained to recover all 8 relation values.

For a representation `x`, use 8 separate relation heads. Each head is:

- Linear input_dim -> 32
- GELU
- Linear 32 -> 16

The primary geometry observer uses final observer state `o_12` as `x`.

Loss is the mean of the 8 categorical cross-entropies.

Overall accuracy is the mean correctness across all 8 relations and all memories.

No query embedding is used anywhere in Observer-R2. This prevents query information from entering the observer or readout path.

## 11. Required matched controls and ablations

### A. Final-transition snapshot control — primary comparator

A non-recurrent system receives exactly `g_11`, the final R2 primary geometry vector.

Its 8 relation heads use the same MLP head architecture above with input dimension 19.

This is the primary comparator for distinguishing accumulated trajectory history from information available in one transition.

### B. Reset/no-memory observer

Use the same GRUCell architecture and hidden size as the primary observer, but reset hidden state to zero before every transition:

`o_{t+1}^{reset} = GRUCell(g_t, 0)`.

Only the output produced from the final transition is decoded.

This preserves the observer cell transformation while eliminating recurrent accumulation.

### C. Temporally shuffled recurrent observer

For each memory, preserve the exact multiset `{g_0,...,g_11}` but feed the 12 vectors to the recurrent observer in a deterministic random permutation.

The permutation is independently sampled per memory from a dedicated RNG namespace and is independent of labels.

The geometry vectors themselves are not recomputed after shuffling. Because turning is excluded from primary `g_t`, this control preserves the complete primary observation multiset without carrying an explicit adjacency feature.

Train, validation, and test each have independent deterministic shuffle streams.

### D. Direction-only recurrent observer

Input is only `u_t in R^16`.

Observer hidden size remains 8; readout heads are unchanged.

### E. Speed-only recurrent observer

Input is only `speed_t in R^1`.

Observer hidden size remains 8; readout heads are unchanged.

### F. Geometry + full-state recurrent observer

Historical-style secondary comparison receives:

`[g_t ; h_t] in R^35`.

Observer hidden size remains 8.

This is explicitly secondary because access to `h_t` permits ordinary state decoding.

### G. Direct final-state control

Eight matched MLP heads read `h_12` directly using input dimension 16.

### H. Direct initial-state control

Eight matched MLP heads read `h_0` directly using input dimension 16.

This records how much information was already available before recurrent evolution.

## 12. Training and initialization

PyTorch AdamW is used for all trainable systems.

Core:

- learning rate: 1e-3
- weight decay: 1e-4
- memory batch size: 256
- maximum epochs: 100
- gradient clipping: 1.0
- early-stopping patience: 12 epochs
- minimum validation improvement: 1e-4
- checkpoint metric: `CoreVal`

Observers, controls, and readout heads:

- learning rate: 1e-3
- weight decay: 1e-4
- memory batch size: 256
- maximum epochs: 100
- gradient clipping: 1.0
- early-stopping patience: 12 epochs
- minimum validation improvement: 1e-4
- checkpoint metric: overall validation accuracy across all 8 relations

No scheduler, warmup, dropout, batch normalization, test-time augmentation, or hyperparameter search is permitted.

Initialization:

- embeddings: Normal(0, 0.02^2)
- ordinary Linear weights: Xavier uniform
- Linear biases: zero
- encoder GRU input weights: Xavier uniform
- encoder GRU recurrent weights: orthogonal independently per gate block
- encoder GRU biases: zero
- observer GRUCell input weights: Xavier uniform
- observer GRUCell recurrent weights: orthogonal independently per gate block
- observer GRUCell biases: zero

## 13. Selection discipline and test blindness

The fixed sequence for each primary seed is:

1. generate train and validation memories only;
2. train core on train;
3. select core checkpoint on validation only;
4. freeze encoder and recurrent dynamics;
5. construct train/validation trajectories and geometric features;
6. train every observer/control on train only;
7. select every observer/control checkpoint on validation only;
8. freeze every checkpoint;
9. verify required checkpoint files and hashes;
10. generate the test memory split for the first time;
11. construct frozen test trajectories/features;
12. evaluate every frozen system and all preregistered endpoints.

No test data inspection is permitted before step 10.

Development/smoke testing must use non-primary seeds and independently generated data. Primary seeds 7, 19, and 43 may not be used for debugging before the frozen primary run sequence.

## 14. Primary endpoint and evidence hierarchy

The primary endpoint for each seed is:

`Delta_history = Accuracy(geometry recurrent observer) - Accuracy(final-transition snapshot)`.

### Level 0 — no trajectory-history evidence

The geometry recurrent observer does not reproducibly exceed the final-transition snapshot.

### Level 1 — geometric observability

Geometry-based systems predict above chance, but recurrence/history provides no reproducible advantage.

### Level 2 — accumulated trajectory information

All three primary seeds have `Delta_history > 0`, the three-seed mean is positive, and the geometry recurrent observer also exceeds the reset/no-memory observer in all three seeds.

### Level 3 — temporal-order-sensitive trajectory information

Level 2 holds and the native-order recurrent observer exceeds the temporally shuffled recurrent observer in all three primary seeds.

Bootstrap intervals are reported as uncertainty summaries; the level definitions do not depend on post-hoc significance thresholds.

No level establishes that trajectories create information, that this is a universal neural mechanism, or that the observer is necessary for the main network's computation.

## 15. Statistics

All compared systems are evaluated on identical held-out memories.

For each memory, define system accuracy as the mean of the 8 relation-correct indicators. Paired differences are therefore computed at the memory level while preserving the 8 within-memory relation outcomes.

For each primary paired comparison, report:

- observed paired mean difference;
- 10,000 paired bootstrap resamples of the 5,000 test memories;
- percentile 95% bootstrap interval;
- per-seed result.

Across the three seeds report mean, sample standard deviation, minimum, and maximum. Seed-level SD is descriptive because there are only three preregistered seeds.

Secondary ablations are descriptive and are not treated as separate primary significance tests.

## 16. Logging requirements

For every test memory, save targets and predictions for every primary/control system.

For trajectory analysis, log full per-step values for the first 512 test memories by deterministic test memory index (`0..511`) for each seed:

- `h_t` for `t=0..12`
- `dh_t` for `t=0..11`
- normalized direction
- speed
- radius
- radial change
- turning statistic
- primary observer state `o_t`

The 512-memory subset is fixed before test generation and cannot be changed after inspecting results.

Logging is for analysis only. Saved trajectories must not be used to give the primary observer future access.

## 17. Core integrity report

After all systems are frozen and the test split is generated, evaluate the discarded temporary core heads on test at `h_0` and `h_12`.

Report all 16 per-relation head accuracies and their means.

Poor core retention changes interpretation but does not permit redesign or rerunning Observer-R2.

## 18. Required per-seed outputs

Retain:

- exact config and software versions;
- all RNG namespaces/seeds and split counts;
- core selected epoch and validation metric;
- observer/control selected epochs and validation metrics;
- parameter counts;
- checkpoint hashes;
- dataset and frozen trajectory hashes;
- overall and per-relation test accuracy for all systems;
- per-memory/per-relation targets and predictions;
- `Delta_history`;
- native-vs-reset paired difference;
- native-vs-shuffled paired difference;
- direction-only and speed-only results;
- geometry+state result;
- direct `h_0` and `h_12` results;
- bootstrap intervals;
- core-integrity test heads;
- fixed 512-memory trajectory/observer log.

Aggregate outputs must be mechanically derived from the frozen three-seed records.

## 19. No-rerun / failure rule

Bad primary seeds stay. Negative results stay.

A failure of recurrence to beat the snapshot does not trigger redesign within R2. A failure of temporal shuffling to hurt does not trigger a new shuffle rule within R2. A weak speed/direction condition does not trigger feature engineering within R2.

If an implementation bug requires a rerun, preserve and label invalid outputs, document the exact bug and correction, and do not use invalid test results to change the scientific design.

## 20. Explicit non-claims

Observer-R2 must not by itself be presented as evidence that:

- trajectory geometry creates new information;
- an internal observer exists in standard transformers or biological systems;
- the observer improves the underlying latent computation;
- trajectory encoding is universal;
- attention has been replaced;
- direct latent readout is inferior;
- historical observer results have been exactly reproduced.

The intended narrow claim, if Level 2 or Level 3 is supported, is:

> In this learned recurrent latent system, an online observer restricted to trajectory-derived geometric features accumulated task-relevant information from latent evolution beyond what was available to a matched decoder of the final transition alone.

If Level 3 is supported, it is additionally valid to state that the accumulated information depended reproducibly on the native temporal ordering under the preregistered shuffle intervention.