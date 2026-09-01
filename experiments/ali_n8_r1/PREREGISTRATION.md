# ALI-N8-R1 Preregistration

**Status:** LOCKED BEFORE IMPLEMENTATION OR PRIMARY RUNS

**Version:** ALI-N8-R1 v2

This document fixes the scientific design of the first independently reproducible Active Latent Interrogation experiment. Any scientific design change after this point must receive a new experiment version (for example, ALI-N8-R2) rather than silently modifying R1.

## Provenance boundary

The historical autonomous N=8 ALI CSVs in `results/` are preserved recovered aggregate evidence. They record historical measurements, including seeds 5, 17, and 31, but the exact historical generator, implementation, training configuration, checkpoint-selection procedure, and per-seed/per-example outputs are unavailable. Those historical results are therefore not independently reproducible from this repository.

ALI-N8-R1 is a new experiment specified from first principles. It will not be reconstructed or tuned to reproduce the historical 64.67% result. Historical CSVs must remain unchanged. New outputs belong under `results/reproducible/ali_n8_r1/`.

## Scientific question

Given a fixed-width latent memory known to preserve eight stored facts, can controlled query-dependent perturbations of that frozen state selectively expose information through the response of a frozen nonlinear transformation?

The 16-dimensional continuous state used here is called a **fixed-width latent memory**, not an information-theoretically compressed memory. R1 makes no compression claim.

## Dataset

There are 8 relations and 16 possible categorical values. Each memory contains exactly one independently and uniformly sampled value for every relation:

`M = {(r_0,y_0), ..., (r_7,y_7)}`.

Fixed memory counts:

- train: 20,000
- validation: 2,500
- test: 5,000

Split by memory first. Every memory is then expanded into all eight query examples `(M,q_i,y_i)`, yielding 160,000 train, 20,000 validation, and 40,000 test query examples. The same underlying memory may never cross splits.

The eight relation/value pairs are shuffled during encoding. Relation identity travels with each value, so physical input position is not relation identity. Training permutations may vary deterministically by the designated RNG stream; validation and test permutations are generated deterministically and frozen.

Primary seeds are 5, 17, and 31. Separate deterministically derived RNG streams are used for memory generation, train/validation/test permutations, initialization, minibatch order, random-direction controls, and wrong-memory pairing.

## Input representation

Trainable relation embedding: `e_r(r) in R^8`.

Trainable value embedding: `e_y(y) in R^8`.

Each fact is `x_i = [e_r(r_i); e_y(y_i)] in R^16`.

Embeddings are trained only during core pretraining and frozen afterward.

## Encoder E

A one-layer GRU processes the shuffled eight-fact sequence:

`h_i = GRU(x_i,h_{i-1})`, hidden width 32.

The final hidden state is projected:

`m = tanh(W_m h_8 + b_m)`, with `32 -> 16`.

Thus `m in R^16` and each coordinate is bounded by tanh. This does not establish information-theoretic compression.

## Frozen nonlinear transformation F

`F` is a learned nonlinear memory-preserving transformation:

`u = GELU(W_1 m + b_1)`

`z = tanh(W_2 u + b_2)`

with widths `16 -> 32 -> 16`, no skip connection, no identity penalty, no contraction regularizer, no attractor objective, and no perturbation objective.

## Core pretraining

Temporary heads reconstruct all eight values from both `m` and `F(m)`. There are eight linear `16 -> 16` heads `H_i^m` and eight linear `16 -> 16` heads `H_i^z`.

Core loss:

`L_core = (1/8) sum_i CE(H_i^m(m),y_i) + (1/8) sum_i CE(H_i^z(F(m)),y_i)`.

Only training memories produce gradient updates. Validation selects the core checkpoint using the unweighted mean validation accuracy across all sixteen temporary heads. Test is untouched.

After selection, all temporary heads are discarded and the embeddings, encoder E, and transformation F are frozen permanently for R1. ALI never trains E or F.

## Perturbation scale

After the core checkpoint is selected, compute from training latent states only:

`c = median_train ||m||_2`

and freeze

`alpha = 0.1 c`.

Alpha is scale calibration, not performance tuning. It is never selected or modified using validation/test accuracy.

## Response

Every active probe uses one symmetric local finite difference:

`r(m,v) = [F(m + alpha v) - F(m - alpha v)] / (2 alpha)`.

Raw policy outputs are normalized:

`v = v_tilde / (||v_tilde||_2 + 1e-8)`.

There is one interrogation step only: no recurrent trajectory, horizon sweep, or multiple probes. For sufficiently small alpha the response approximates `J_F(m)v`, but R1 evaluates the actual finite difference.

## Query representation

Every separately trained query-aware system owns its own trainable query embedding `e_q(q) in R^8`. Query embeddings are not shared across experimental systems.

## Learned systems

### Content-adaptive ALI P(m,q)

Policy input `[m;e_q(q)]` with MLP `24 -> 32 -> 16`, GELU hidden activation, normalized output direction.

Reader receives exactly `[r;e_q(q)]`, with MLP `24 -> 32 -> 16`, GELU hidden activation, final 16 target logits.

The reader receives no `m`, perturbed state, `v`, policy hidden activation, original input, encoder state, target, or unreported memory-derived side channel.

### Query-only ALI P(q)

Policy MLP `8 -> 32 -> 16` from query embedding only, GELU hidden activation, normalized direction. Reader is the same `24 -> 32 -> 16` response/query architecture.

`P(q)` has no access to `m`, `F(m)`, input facts, encoder hidden state, or targets.

### Query-blind adaptive P(m)

Policy MLP `16 -> 32 -> 16`, GELU, normalized direction. Reader receives response plus true query using the standard reader architecture.

### Learned global fixed direction

One trainable vector in `R^16`, normalized during forward use and shared by every memory/query. The response reader trains normally.

### Random fixed direction

One seed-specific vector sampled from `N(0,I)`, normalized, fixed before reader training, independent of memory/query, and never optimized or performance-selected.

### Zero perturbation

Reader receives an all-zero 16-dimensional response plus the true query. This measures query/dataset leakage; chance is 1/16 = 6.25% under the specified generator.

## Direct controls

Two direct controls are mandatory.

1. `R_m(m,q)`: input `[m;e_q(q)]`, MLP `24 -> 32 -> 16`, GELU.
2. `R_F(F(m),q)`: input `[F(m);e_q(q)]`, same MLP.

ALI receives no credit merely because F creates a useful nonlinear feature representation.

Parameter counts for every model are reported. The fixed architectures above are not changed after test inspection.

## Direction-only leakage diagnostics

After training the adaptive policies, freeze them and train diagnostic readers that receive the direction rather than response:

- adaptive `P(m,q)`: `D(v,q) -> y`
- query-blind `P(m)`: `D(v,q) -> y`

Diagnostic architecture: `[v;e_q(q)]`, MLP `24 -> 32 -> 16`, GELU.

These diagnostics test whether a continuous adaptive direction itself carries target information rather than obtaining it through interrogation. `P(q)` cannot leak memory values through its direction by construction because its direction depends only on query identity.

## Wrong-memory adaptive intervention

For a true memory `m_i` and query `q_i`, native direction is `P(m_i,q_i)`. Construct a wrong-memory direction `P(m_k,q_i)` from a deterministic donor `k != i`, then perturb the true `m_i`. The reader still receives the true query.

Report native accuracy, wrong-memory accuracy, prediction-change rate, and paired difference. This tests whether the adaptive policy performs memory-specific computation before perturbation.

## Query-only native 8x8 swap matrix

For every query-only direction `v_j = P(q_j)` and every requested relation `q_i`, compute `r_j = r(m,v_j)` while the already-trained native ALI reader always receives the true `q_i`.

`A_native[i,j] = P(R(r_j,q_i) = y_i)`.

This is a causal intervention on the trained system, but off-diagonal responses may be out of distribution for its native reader. Therefore a diagonal in this matrix alone is not sufficient evidence of geometric selectivity.

For each row:

`D_native_i = A_native[i,i] - (1/7) sum_{j != i} A_native[i,j]`.

Overall `D_native` is the mean of the eight row advantages.

## Independent 8x8 decodability matrix

After query-only ALI, E, F, alpha, policy, and native reader are fully frozen, train a separate diagnostic decoder for each `(i,j)` pair:

`D_ij(r_j) -> y_i`.

There are 64 diagnostic readers, each `16 -> 32 -> 16` with GELU. Training memories train diagnostics, validation selects their checkpoints, and test memories evaluate them.

`A_decode[i,j] = P(D_ij(r_j) = y_i)`.

This does not claim absolute information-theoretic availability. It measures how accurately relation i can be decoded from the response to direction j using the same preregistered diagnostic architecture trained independently for that cell.

Compute row and global diagonal advantages identically to the native matrix. Save all 64 counts and accuracies for every seed, not only an averaged heatmap.

## Direction geometry

For query-only `P(q)`, save the complete 8x8 cosine-similarity matrix among the eight learned normalized directions. There is no diversity loss, orthogonality penalty, cosine regularizer, or geometry-selection criterion.

## Initialization

- relation/value/query embeddings: Normal(0, 0.02^2)
- ordinary linear weights: Xavier uniform
- ordinary linear biases: zero
- GRU input weights: Xavier uniform
- GRU recurrent weights: orthogonal initialization independently per gate block
- GRU biases: zero
- learned global fixed vector: Normal(0,1) before forward normalization
- random control vector: Normal(0,1) before permanent normalization

No pretrained components.

## Optimization

PyTorch AdamW for all trained systems.

Core:

- learning rate: 1e-3
- weight decay: 1e-4
- memory batch size: 256
- max epochs: 100
- gradient norm clip: 1.0
- early-stopping patience: 12
- checkpoint metric: CoreVal defined above
- minimum improvement: 1e-4

Main ALI/control models and diagnostic readers:

- learning rate: 1e-3
- weight decay: 1e-4
- query-example batch size: 256
- max epochs: 100
- gradient norm clip: 1.0
- early-stopping patience: 12
- checkpoint metric: overall validation target accuracy
- minimum improvement: 1e-4

No learning-rate scheduler, warmup, dropout, batch normalization, hyperparameter sweep, seed replacement, or test-time augmentation.

All target prediction uses categorical cross entropy. ALI has no auxiliary geometry/diversity/response-norm loss.

## Model-selection order

1. Generate memory-level train/validation/test splits.
2. Train core using train only.
3. Select core using validation only.
4. Freeze embeddings, E, and F.
5. Compute alpha using training latent states only and freeze it.
6. Train ALI/control systems using training query examples.
7. Select each using validation accuracy.
8. Freeze them.
9. Train post-hoc diagnostic readers using training responses.
10. Select diagnostics using validation responses.
11. Evaluate every frozen system on test for the primary R1 results.

Primary test results must not guide model selection or scientific design.

## Core integrity checks

After everything is frozen, report the discarded temporary heads' test reconstruction accuracies from both `m` and `F(m)`. These establish whether the substrate preserves the eight facts. Poor core reconstruction changes interpretation of a failed R1; it does not justify silently changing R1.

## Required per-seed artifacts

For seeds 5, 17, and 31 save machine-readable:

- complete configuration and software/library versions
- all derived RNG seeds
- split counts and deterministic dataset/split identifiers
- core checkpoint epoch and validation metric
- per-head core reconstruction accuracies
- median training latent norm and frozen alpha
- parameter counts
- selected epoch for every trained model
- overall and per-relation test accuracy
- per-example test targets and predictions
- all control results
- direction-only leakage results
- wrong-memory intervention results
- complete A_native counts/accuracies
- complete A_decode counts/accuracies
- complete query-only direction cosine matrix
- per-relation and global diagonal advantages
- checkpoint hashes

Retain checkpoints and training histories so interventions can be rerun.

## Aggregation and statistics

Aggregation is generated mechanically from per-seed artifacts. Report mean, sample standard deviation, minimum, and maximum for scalar metrics. For matrices report every seed separately plus cellwise mean and seed SD.

With only three model seeds, seed SD is descriptive. Where paired uncertainty is reported from test examples, resample at the **memory level**, preserving the eight queries belonging to each memory. The preregistered primary matrix endpoint is global diagonal advantage; individual matrix cells are descriptive and are not independently significance-tested as 64 separate primary hypotheses.

## Failure and no-rerun discipline

All primary seeds remain in the record, including poor runs. Negative results are retained. Test results cannot trigger architecture, alpha, seed, width, loss, or training changes inside R1. A genuine implementation bug may be corrected only with an explicit bug record and rerun provenance; a scientific design change creates a new experiment version.

If query-only ALI fails, R1 is not modified to rescue it. If adaptive directions leak targets, that is reported. If independent decodability removes a native swap diagonal, that is reported.

## Preregistered claim hierarchy

**Level 0 — no useful ALI effect.** Active response models do not demonstrate useful interrogation relative to controls.

**Level 1 — content-adaptive computation.** `P(m,q)` works but direction-only decoding also works strongly; this is not strong evidence of geometric interrogation.

**Level 2 — content-dependent active interrogation.** `P(m,q)` works, direction-only leakage is weak, and wrong-memory probe selection damages performance. This is stronger evidence that the policy uses the particular memory to determine how to interrogate it.

**Level 3 — query-selected functional direction.** `P(q)` works and native direction substitution degrades the trained system. Query identity selects functionally important directions, but native-reader OOD remains a possible contributor.

**Level 4 — query-specific geometric addressing.** The strongest R1 result requires reproducible positive diagonal structure in the independent decodability matrix, especially positive `D_decode` across seeds, together with native functional direction dependence. This supports query-specific geometric addressing **in this particular learned latent system**.

R1 does not establish universal latent addresses, information-theoretic compression, superiority to attention, a new universal computational primitive, or natural-language generalization.
