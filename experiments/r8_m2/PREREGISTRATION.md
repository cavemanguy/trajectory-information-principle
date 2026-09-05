# R8-M2 Preregistration — Symmetry-Breaking Provenance and Early Commitment

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent result:** R8-M1 = M3, encoder–recurrence coadaptation supported  
**Primary study:** native trajectories only; no latent perturbation, active observer, or controller

## 1. Scientific question

R8-M1 established that the full magnitude of relation-selective native survival under the joint training regime depends on plasticity in both the encoder and recurrent map. It did not identify why a statistically symmetric eight-relation task develops a different strongly preserved relation in different seeds.

R8-M2 asks:

> **Which ordinary source of symmetry breaking best explains winner identity: relation-specific initialization, finite-sample data-column asymmetry, or stochastic optimization order; and when during training does the final survival ranking become predictable?**

The experiment is designed to test simple causes before using stronger language such as emergence.

## 2. Guardrails

- No latent perturbations in training or analysis.
- All trajectory measurements are passive.
- All seeds, mappings, checkpoints, gates, endpoints, and decision rules are fixed before fresh outcomes.
- No failed gate may be repaired after outcome inspection.
- Secondary predictors cannot rescue a failed primary source-tracking result.
- R8-M1 remains unchanged regardless of this outcome.
- Practical implications remain separate engineering hypotheses.

## 3. Architecture and objective

Use the same recovered Observer-R2 / ND-R1 / R8-M1 core lineage:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation embedding width 8;
- value embedding width 8;
- encoder GRU hidden width 32;
- latent width 16;
- recurrent map `F: 16 -> 32 -> 16` with GELU then tanh;
- 12 recurrent transitions;
- separate h0 and h12 linear heads per relation.

Every condition uses the original joint objective:

`L = mean_r [CE(head0_r(h0), y_r) + CE(headT_r(h12), y_r)]`

Training is fixed at 100 epochs with AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch 256, gradient clipping 1.0.

Data per family seed:

- train: 20,000 source memories;
- validation: 2,500 source memories;
- test: 5,000 reserved but not required for the primary source-tracking classification;
- natural-pair bank: 2,048 valid source-memory pairs per relation.

## 4. Fresh family seeds

Eight fresh family seeds are fixed:

`[3, 9, 21, 28, 44, 62, 86, 101]`

They do not overlap the historical Observer seeds 7/19/43, ALI-N8-R1 seeds 5/17/31, ND-R1 seeds 13/29/53, or R8-M1 seeds 11/37/71.

Each family contains four paired training conditions.

## 5. Fixed relation derangement

Use the fixed cyclic derangement

`pi(r) = (r + 3) mod 8`

or equivalently

`pi = [3,4,5,6,7,0,1,2]`.

`pi(r)` always means the baseline identity assigned to model relation slot `r` in a mapped condition.

## 6. Paired conditions

### B — baseline

Use the baseline initialized model, source data in identity relation order, and the baseline deterministic minibatch order.

### P — relation-specific parameter-bundle permutation

Start from the same complete baseline initialization, then apply `pi` only to relation-specific initialized parameter bundles:

- relation-embedding row;
- h0 head weight/bias;
- h12 head weight/bias.

Model relation slot `r` receives the baseline relation-specific bundle from `pi(r)`.

All shared parameters (`value embedding`, encoder GRU, `to_h`, recurrent map `F`) remain bit-identical to B at epoch 0. Data and minibatch order remain identical to B.

If final specialization is substantially seeded by the relation-specific initialized bundle, the P winner should follow the bundle identity after mapping back through `pi`.

### D — finite-sample data-column permutation

Use the exact same baseline initialization as B.

Generate one underlying source-memory matrix and map source columns to model relation slots by

`y_D[:, r] = y_source[:, pi(r)]`.

Apply the same mapping to train, validation, test, and natural-pair source memories. The model's relation-specific initialized parameters are not permuted. Minibatch order is identical to B.

If final specialization substantially follows finite-sample source-column idiosyncrasy, the D winner should follow source-column identity after mapping back through `pi`.

### O — alternate minibatch order

Use the exact same baseline initialization and exact same identity-mapped data as B.

Keep the same per-example within-memory relation presentation permutations as B, but use an independently derived minibatch order each epoch.

This changes only the SGD example-order path.

O is a preregistered secondary sensitivity condition rather than part of the primary S0–S3 source classification.

## 7. Checkpoints

Save all four conditions at epochs:

`[0, 1, 2, 5, 10, 20, 40, 60, 100]`

B, D, and O must have identical epoch-0 parameter hashes. P must match B on all shared parameter tensors, with the relation-specific bundle permutation exactly verified.

## 8. Native survival endpoint

For each model-relation slot `r`, construct valid natural input pairs differing only in that model relation.

For D, pairs are generated in underlying source coordinates and mapped through `pi`, so the source identity of model slot `r` is known exactly.

For recurrent time `t`:

`S_r(t) = median_pairs ||h_t(x)-h_t(x')|| / (||h_0(x)-h_0(x')|| + eps)`.

At each training checkpoint record:

- the full eight-relation terminal survival vector `S(12)`;
- log terminal survival vector;
- selectivity `G = SD_r(log(S_r(12)+eps))`;
- winner relation `argmax_r S_r(12)`;
- full `S_r(t)` for `t=0..12`.

## 9. Training-validity gate

Every condition in every family must reach at epoch 100:

- combined validation `(mean h0 + mean h12)/2 >= 0.38`; and
- mean h0 validation accuracy `>= 0.55`.

These thresholds are carried forward unchanged from the successful R8-M1 baseline gate.

If any condition in any family fails either threshold, classify R8-M2 as **V — training validity failure** and do not promote the primary source-tracking contrasts. All results remain preserved.

## 10. Primary H1 — relation-specific initialization-bundle tracking

For family `s`:

- baseline winner: `w_B`;
- P winner in model-label coordinates: `w_P`;
- P winner's baseline bundle identity: `pi(w_P)`.

A winner match occurs when

`pi(w_P) == w_B`.

Under unrelated winner identity across 8 symmetric relations, chance match probability is 1/8. With 8 families, at least 4 mapped winner matches has one-sided exact binomial tail probability <0.05.

Also compute terminal log-survival Spearman correlation between B and P in two coordinate systems:

1. raw model-label alignment;
2. P mapped back into baseline bundle-identity coordinates using `pi`.

For each family define

`Delta_rho_init = rho_bundle_aligned - rho_raw`.

Use a deterministic 5,000-resample family-level bootstrap over the eight family seeds for the mean `Delta_rho_init`.

**H1 is supported only if both:**

1. mapped winner matches are at least 4/8; and
2. the 95% bootstrap CI lower bound for mean `Delta_rho_init` is >0.

## 11. Primary H2 — finite-sample data-column tracking

For D, the source-column identity assigned to model slot `r` is `pi(r)`.

A source-column winner match occurs when

`pi(w_D) == w_B`.

Also compare B with D terminal log-survival vectors using:

1. raw model-label alignment;
2. D mapped back into underlying source-column coordinates.

Define

`Delta_rho_data = rho_source_aligned - rho_raw`.

**H2 is supported only if both:**

1. mapped source-column winner matches are at least 4/8; and
2. the 95% family-bootstrap CI lower bound for mean `Delta_rho_data` is >0.

## 12. Primary outcome classification

After the validity gate:

- **S0 — neither simple source tracks winner:** H1 false, H2 false.
- **S1 — relation-specific initialization tracking:** H1 true, H2 false.
- **S2 — finite-sample data-column tracking:** H1 false, H2 true.
- **S3 — both sources contribute:** H1 true, H2 true.

These outcomes concern source tracking only. They do not establish deterministic causation of every seed's winner.

## 13. Secondary — minibatch-order sensitivity

For B versus O, report across families:

- raw winner agreement count;
- per-family Spearman correlation of terminal log-survival vectors;
- mean Spearman with 5,000-resample family bootstrap CI;
- per-relation absolute log-survival differences.

No binary primary claim is attached to O because there is no preregistered equivalence null for 'same enough'. This result will quantify optimization-path sensitivity without post-hoc thresholding.

## 14. Secondary — early commitment of final ranking

Within B, for every checkpoint `e` in `[0,1,2,5,10,20,40,60]`, compare the eight-relation terminal-survival ranking at epoch `e` with the epoch-100 ranking.

Report:

- per-family Spearman rho;
- mean rho and family-bootstrap CI;
- top-1 winner agreement count with epoch 100.

Define the descriptive **commitment onset** as the earliest checkpoint where all are true:

1. at least 6/8 family rhos are positive;
2. the 95% family-bootstrap CI lower bound for mean rho is >0;
3. epoch-e winner agrees with epoch-100 winner in at least 4/8 families.

This is a preregistered descriptor, not a separate primary hypothesis.

## 15. Secondary — epoch-0 shared-gradient asymmetry

Before training B, use a fixed diagnostic batch of 2,048 training memories and deterministic relation-presentation permutations.

For each relation separately, compute the joint h0+h12 loss for that relation and the L2 norm of its gradient with respect to the **shared encoder/recurrent parameters only**:

- encoder GRU;
- `to_h`;
- recurrent map `F`.

Do not include relation-specific embeddings or heads in this primary gradient scalar.

Report per-family Spearman correlation between the eight initial shared-gradient norms and final B terminal survival. Also report the mean rho and family-bootstrap CI.

Initial loss, relation-embedding norm, head norms, and initial survival remain secondary descriptive predictors.

## 16. Claim boundaries

Even a strong S1, S2, or S3 would show only that an ordinary asymmetry source tracks the later specialized channel in this synthetic recurrent system.

S0 would not prove strong emergence; it would mean the tested simple sources did not account for winner identity under the preregistered tracking rules.

R8-M2 does not establish:

- a universal trajectory-information principle;
- new information creation;
- chronology as essential;
- strong emergence;
- practical advantage;
- language-model generalization;
- perturbation necessity.
