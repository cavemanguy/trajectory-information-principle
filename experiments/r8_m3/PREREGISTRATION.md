# R8-M3 Preregistration — Commitment Transition Mechanism

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent results:** R8-M1 = M3 (encoder–recurrence coadaptation supported); R8-M2 = S0 (neither tested fixed source tracks winner)  
**Primary study:** native dynamics only; no latent perturbation, active observer, controller, or trajectory steering

## 1. Scientific question

R8-M1 established that full relation-selective native survival under the joint training regime requires plasticity in both the encoder and recurrent map. R8-M2 then showed that winner identity did not reliably track either the tested relation-specific initialization bundle or the tested finite-sample data-column identity. Under R8-M2's frozen descriptor, stable winner commitment appeared at epoch 40.

R8-M3 asks:

> **What native optimization/geometric signal becomes predictive of the eventual specialist before the specialization is already stably committed?**

The primary goal is predictive mechanism narrowing, not a causal claim.

## 2. Guardrails

- No latent perturbations in training or primary analysis.
- All trajectory measurements use ordinary task examples and natural task contrasts.
- Component recombination is a passive diagnostic; it does not alter training trajectories.
- All seeds, checkpoints, metrics, gates, and decision rules are frozen before fresh outcomes.
- Prediction is not causation.
- Failure of simple predictors is not evidence of strong emergence by itself.
- R8-M1 and R8-M2 remain unchanged regardless of R8-M3 outcome.
- No failed gate may be repaired after outcome inspection.

## 3. Architecture and training lineage

Use the same recovered Observer-R2 / ND-R1 / R8-M1 / R8-M2 core lineage:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation embedding width 8;
- value embedding width 8;
- encoder GRU hidden width 32;
- latent width 16;
- recurrent map `F: 16 -> 32 -> 16` with GELU then tanh;
- 12 recurrent transitions;
- separate h0 and h12 linear heads per relation.

Objective:

`L = mean_r [CE(head0_r(h0), y_r) + CE(headT_r(h12), y_r)]`

Training:

- 100 epochs;
- AdamW, `lr=1e-3`, `weight_decay=1e-4`;
- batch size 256;
- gradient clipping 1.0;
- train/validation/test = 20,000 / 2,500 / 5,000 memories;
- natural-pair bank = 2,048 valid pairs per relation.

## 4. Fresh seeds

Twelve fresh family seeds are fixed:

`[14, 24, 34, 47, 58, 73, 89, 107, 116, 127, 139, 151]`

They do not overlap the historical Observer seeds 7/19/43, ALI-N8-R1 seeds 5/17/31, ND-R1 seeds 13/29/53, R8-M1 seeds 11/37/71, or R8-M2 seeds 3/9/21/28/44/62/86/101.

Each family contains two paired training paths:

- **B:** baseline deterministic minibatch order;
- **O:** identical initialization and identical data, alternate deterministic minibatch order.

B is used for the primary mechanism classification. O is a preregistered secondary optimization-path test.

## 5. Checkpoints

Save B and O at epochs:

`[0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]`

The denser 15–50 window is fixed because R8-M2 placed stable commitment onset at epoch 40.

B and O must be bit-identical at epoch 0.

## 6. Native survival endpoint

For each relation `r`, construct natural task pairs that differ only in relation `r`.

For recurrent transition `t`:

`S_r(t) = median_pairs ||h_t(x)-h_t(x')|| / (||h_0(x)-h_0(x')|| + eps)`.

At every checkpoint record:

- full `S_r(t)` for `t=0..12`;
- terminal vector `S_r(12)`;
- `log S_r(12)`;
- selectivity `G = SD_r(log(S_r(12)+eps))`;
- winner `argmax_r S_r(12)`.

Final winner is the epoch-100 B winner unless otherwise specified.

## 7. Training-validity gate

Every B and O condition in every family must reach at epoch 100:

- combined validation `(mean h0 + mean h12)/2 >= 0.38`; and
- mean h0 validation accuracy `>= 0.55`.

These thresholds are carried forward unchanged from successful R8-M1/R8-M2 lineage validation.

If any condition in any family fails either threshold, classify R8-M3 as **V — training validity failure**. Preserve all outputs but do not promote primary mechanism contrasts.

## 8. Primary precommitment timepoint

The primary predictor timepoint is fixed at **epoch 20**.

Reason: R8-M2 identified stable commitment at epoch 40 under its preregistered descriptor, and epoch 20 is an already-used checkpoint safely before that commitment point.

No other checkpoint may replace epoch 20 for the primary classification after outcomes are observed.

## 9. Primary candidate P1 — coadaptation-synergy predictor

At each checkpoint `e`, retain the epoch-0 encoder `E0` and recurrent map `F0`, plus current encoder `Ee` and recurrent map `Fe`.

Using the same frozen natural-pair bank, pass ordinary encoded states through the following passive component combinations:

1. matched current system: `(Ee, Fe)`;
2. current encoder + initial recurrence: `(Ee, F0)`;
3. initial encoder + current recurrence: `(E0, Fe)`.

For each relation define terminal log survival under each combination:

- `L_match_r(e)`;
- `L_encoder_r(e)`;
- `L_recurrence_r(e)`.

Define relation-specific coadaptation synergy:

`C_r(e) = L_match_r(e) - 0.5 * [L_encoder_r(e) + L_recurrence_r(e)]`.

Primary P1 prediction at epoch 20:

`w_C = argmax_r C_r(20)`.

Also compute Spearman correlation between the eight-element vector `C(20)` and the epoch-100 B terminal log-survival vector.

**P1 is supported only if both:**

1. `w_C` matches the final B winner in at least **5/12** fresh families; and
2. the deterministic 5,000-resample family-bootstrap 95% CI lower bound for mean Spearman correlation is `> 0`.

The 5/12 winner threshold has one-sided exact-binomial probability below 0.05 under unrelated 1/8 winner identity.

## 10. Primary candidate P2 — shared-gradient alignment predictor

At epoch 20 B, use a fixed diagnostic batch of 1,024 training memories and deterministic relation-presentation permutations.

For each relation separately, compute the joint h0+h12 loss for that relation and its gradient vector `g_r` with respect to **shared trainable parameters only**:

- encoder GRU;
- `to_h`;
- recurrent map `F`.

Exclude relation-specific embeddings and relation-specific heads from this primary gradient vector.

Define the total shared gradient:

`g_all = mean_r g_r`.

Define relation-wise gradient alignment:

`A_r = dot(g_r, g_all) / (||g_r|| ||g_all|| + eps)`.

Primary P2 prediction:

`w_A = argmax_r A_r(20)`.

Also compute Spearman correlation between `A(20)` and the epoch-100 B terminal log-survival vector.

**P2 is supported only if both:**

1. `w_A` matches the final B winner in at least **5/12** fresh families; and
2. the deterministic 5,000-resample family-bootstrap 95% CI lower bound for mean Spearman correlation is `> 0`.

Gradient norm, relation loss, pairwise relation-gradient cosine, and norm-weighted projections are recorded as secondary diagnostics only and cannot replace `A_r` in the primary test.

## 11. Primary outcome classification

After the validity gate:

- **T0 — neither precommitment predictor supported:** P1 false, P2 false.
- **T1 — coadaptation-synergy predictor supported:** P1 true, P2 false.
- **T2 — shared-gradient alignment predictor supported:** P1 false, P2 true.
- **T3 — both precommitment predictors supported:** P1 true, P2 true.

These outcomes are predictive mechanism categories only. T1/T2/T3 do not establish causation.

## 12. Secondary — commitment timing replication

Within B, compare the terminal-survival ranking at every checkpoint before 100 with the epoch-100 ranking.

Report per checkpoint:

- per-family Spearman rho;
- mean rho with 5,000-resample family-bootstrap CI;
- top-1 winner agreement count.

Define commitment onset as the earliest checkpoint where all are true:

1. at least **9/12** family rhos are positive;
2. the 95% family-bootstrap CI lower bound for mean rho is `>0`;
3. checkpoint winner agrees with epoch-100 winner in at least **6/12** families.

This is a descriptive replication endpoint. It may differ from R8-M2's epoch-40 result without invalidating the primary P1/P2 test.

## 13. Secondary — predictor lead/lag

For P1 and P2, compute the same winner agreement and rank-correlation summaries at every saved checkpoint.

For each predictor, report the earliest checkpoint at which its two primary-style support conditions would be met descriptively.

These earlier/later checkpoints are secondary temporal descriptors and cannot replace epoch 20 in the primary classification.

## 14. Secondary — matched versus component selectivity

At each checkpoint report relation selectivity for:

- `(Ee, Fe)`;
- `(Ee, F0)`;
- `(E0, Fe)`.

This tracks whether encoder placement, recurrent-map adaptation, and their matched synergy arise in a consistent temporal order.

No causal claim is attached to temporal precedence alone.

## 15. Secondary — optimization-path sensitivity

For B versus O, report:

- epoch-100 winner agreement;
- Spearman correlation of terminal log-survival vectors at every checkpoint;
- commitment onset separately in B and O;
- whether epoch-20 P1/P2 predictions track each path's own final winner;
- whether a predictor computed on B predicts O's winner or vice versa.

This quantifies path dependence under identical initialization/data while changing only minibatch order.

No binary primary threshold is attached to B/O agreement.

## 16. Secondary — recurrent local geometry

On the native B state distribution, record passive local recurrent diagnostics at saved checkpoints, including:

- natural first-transition gain `S_r(1)`;
- median norm of natural relation contrast at h0;
- optional Jacobian-vector gain along the **natural contrast direction** using automatic differentiation, with no injected random perturbation.

These measurements are secondary because R8-M3's primary mechanism test is P1/P2.

## 17. Claim boundaries

R8-M3 does not establish:

- strong emergence;
- a universal trajectory-information principle;
- new information creation;
- essential chronology;
- practical superiority;
- language-model or transformer generalization;
- causation from a merely predictive feature;
- perturbation necessity.

If P1 or P2 predicts winner identity before commitment, the correct conclusion is that the tested native quantity is a reproducible **precommitment marker/candidate mechanism** in this architecture. A later causal study would be required to establish necessity or sufficiency.

If both fail, the correct conclusion is that these preregistered simple precommitment markers did not account for winner selection; stronger emergence language remains unwarranted.
