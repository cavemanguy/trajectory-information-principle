# R8-M4 Preregistration — Functional Necessity of Dynamical Specialization

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent results:** R8-M1 = M3; R8-M2 = S0; R8-M3 = T0  
**Primary study:** causal training intervention using natural task pairs; no latent-state perturbation, active observer, or trajectory steering

## 1. Scientific question

R8-M1 established that relation-selective native survival is produced by encoder–recurrence coadaptation. R8-M2 ruled out two simple fixed sources of winner identity. R8-M3 found that two epoch-20 predictors did not reliably identify the final winner, while changing only minibatch order materially changed commitment timing and sometimes winner identity.

R8-M4 asks:

> **Can the system learn the task comparably when relation-selective native survival is actively suppressed after the precommitment stage, or does suppressing that specialization impair terminal task performance?**

This is a functional-necessity test. It does not test whether Euclidean survival magnitude by itself is sufficient for reader usefulness; R11 already showed that it is not.

## 2. Guardrails

- All task trajectories remain native; no hidden-state perturbations are injected.
- The intervention is an auxiliary training objective computed from ordinary task examples and natural counterfactual pairs.
- Baseline and intervention paths are bit-identical through epoch 20, then fork.
- The manipulation must first pass a preregistered **specialization-suppression gate** before performance is interpreted.
- A performance drop after suppression supports a functional contribution under this training intervention, not a universal necessity theorem.
- Failure to suppress specialization is a manipulation failure, not evidence that specialization is necessary.
- Preserved performance despite successful suppression is evidence that the observed specialization is not necessary for task performance under this architecture/training regime.

## 3. Architecture and task lineage

Use the same recovered Observer-core architecture and task as R8-M1/M2/M3:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embeddings width 8;
- encoder GRU hidden width 32;
- latent width 16;
- recurrent map `F: 16 -> 32 -> 16`, GELU then tanh;
- 12 recurrent transitions;
- separate h0 and h12 linear heads per relation;
- joint task loss `mean_r[CE(h0_r)+CE(h12_r)]`;
- AdamW `lr=1e-3`, `weight_decay=1e-4`, batch 256, gradient clip 1.0;
- 20,000 train / 2,500 validation / 5,000 test memories.

## 4. Fresh seeds

Twelve fresh family seeds are fixed:

`[16, 26, 39, 52, 64, 78, 93, 109, 122, 136, 148, 163]`

These do not overlap the historical Observer, ALI-N8-R1, ND-R1, R8-M1, R8-M2, or R8-M3 seed sets.

## 5. Shared precommitment training and fork

For each seed, train one ordinary baseline model from epoch 0 through epoch 20.

At the end of epoch 20, clone the complete model and optimizer-independent parameter state into three continuation conditions. The three paths use the same deterministic task minibatch order and relation-presentation permutations from epochs 21–100.

### B — baseline continuation

Continue ordinary joint task training with no auxiliary geometric objective.

### E — selectivity-equalization continuation

Continue the same task loss plus an auxiliary natural-pair objective that penalizes **between-relation dispersion of terminal log survival**.

For a fixed auxiliary batch of ordinary source memories at each training step, construct one natural counterfactual variant for every relation. Let

`z_r = log(mean_i ||h12(x_i)-h12(x_i^r)|| / (mean_i ||h0(x_i)-h0(x_i^r)|| + eps) + eps)`.

Define

`R_eq = mean_r (z_r - mean_j z_j)^2`.

Use

`L_E = L_task + lambda * R_eq`, with fixed `lambda = 0.50`.

The auxiliary batch size is 64 source memories per task minibatch. All eight counterfactual relation variants are evaluated jointly.

### M — mean-survival control continuation

Use the same natural-pair construction and the same `lambda = 0.50`, but do not penalize relation dispersion.

At the epoch-20 fork, compute and freeze the family-specific scalar

`m20 = mean_r z_r`.

Then use

`R_mean = (mean_r z_r - stopgrad(m20))^2`

and

`L_M = L_task + lambda * R_mean`.

M controls for an auxiliary survival-based training objective and extra geometric gradient while not directly targeting relation selectivity.

## 6. Checkpoints and evaluation

Save B/E/M at epochs:

`[20, 25, 30, 35, 40, 50, 60, 80, 100]`.

Use a fixed held-out natural-pair bank of 2,048 valid pairs per relation for evaluation only.

At every saved checkpoint record:

- relation-wise terminal survival `S_r(12)`;
- `G = SD_r(log(S_r(12)+eps))`;
- mean log survival `C`;
- winner relation;
- validation h0 accuracy;
- validation h12 accuracy;
- combined validation accuracy.

At epoch 100 also report test h0/h12/combined accuracy and per-relation test accuracy.

## 7. Lineage/training-validity gate

Baseline B must satisfy at epoch 100 for every fresh seed:

- combined validation >= 0.38;
- h0 validation >= 0.55.

These are carried forward unchanged from R8-M1/M2/M3 lineage validation.

If any B seed fails, classify **V — baseline lineage validity failure** and stop primary interpretation.

E and M are interventions and are not required to satisfy the baseline competence gate; their task-performance differences are outcomes.

## 8. Manipulation gate — successful specialization suppression

The E intervention is considered successful only if **all** of the following hold across the 12 paired fresh seeds at epoch 100:

1. `G_E < G_B` in at least 9/12 seeds;
2. mean paired `DeltaG_EB = G_E - G_B < 0` with deterministic 5,000-resample family-bootstrap 95% CI upper bound `< 0`;
3. mean relative reduction `(G_B-G_E)/(G_B+eps) >= 0.30`.

If this gate fails, classify **F0 — manipulation failure**. Performance differences remain descriptive only.

## 9. Primary functional endpoint

Primary task endpoint is **epoch-100 test h12 accuracy**, because the specialization under study concerns survival through the recurrent rollout.

Define paired differences:

`Delta_EB = Acc_h12(E) - Acc_h12(B)`

`Delta_EM = Acc_h12(E) - Acc_h12(M)`.

Use deterministic 5,000-resample family bootstraps over the 12 paired seeds.

## 10. Frozen functional classification

After baseline validity and successful manipulation:

### F1 — specialization suppressed, task performance preserved

Classify F1 if:

- the 95% CI for mean `Delta_EB` includes 0 or has lower bound greater than `-0.02`; and
- mean `Delta_EB > -0.02`.

Interpretation: successful suppression did not cause a preregistered meaningful terminal-task loss. The observed specialization is not necessary for comparable task performance under this intervention/regime.

### F2 — suppression harms performance, but specificity control fails

Classify F2 if:

- mean `Delta_EB <= -0.02`;
- 95% CI upper bound for mean `Delta_EB < 0`;
- but E is not at least 0.015 worse than M with CI upper bound below 0.

Interpretation: suppression is associated with functional harm, but the result cannot isolate selectivity suppression from generic auxiliary geometric regularization.

### F3 — selective-specialization contribution supported

Classify F3 only if:

- mean `Delta_EB <= -0.02` and its 95% CI upper bound `< 0`;
- mean `Delta_EM <= -0.015` and its 95% CI upper bound `< 0`;
- M reduces G substantially less than E, operationalized as mean `G_M-G_E > 0` with 95% CI lower bound `> 0`.

Interpretation: under this controlled training intervention, specifically suppressing relation-selective native survival impairs terminal task performance beyond the matched mean-survival control. This supports a functional contribution of the specialization in the tested system.

## 11. Secondary analyses

Report without changing the primary classification:

- h0 and combined test accuracy;
- per-relation h12 accuracy changes;
- relation-wise survival/performance coupling;
- whether the baseline winner relation suffers disproportionately under E;
- epoch at which E first diverges from B in G and h12 performance;
- reader margins at h12;
- a standardized post-hoc linear probe trained separately on frozen B/E/M h12 states to distinguish native trained-reader effects from generic linear accessibility.

## 12. Claim boundaries

R8-M4 cannot establish a universal necessity of trajectory specialization, strong emergence, new information creation, essential chronology, or practical superiority.

F1 would show that the observed relation-selective survival pattern is **not necessary for comparable task performance under this tested suppression intervention**.

F2 would be ambiguous functional harm.

F3 would support a **functional contribution** of the specialization under the tested training intervention, not prove that Euclidean survival magnitude itself is the causal mediator.
