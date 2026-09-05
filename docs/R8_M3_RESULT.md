# R8-M3 Result — Commitment Transition Mechanism

**Frozen primary classification:** **T0 — neither precommitment predictor supported**

R8-M3 used twelve fresh paired families in the recovered Observer-core lineage. All families passed the preregistered training-validity gate.

At the fixed primary precommitment epoch 20:

- **P1 coadaptation-synergy predictor:** not supported. Exact final-winner prediction was 2/12. Mean Spearman correlation with the epoch-100 terminal-survival ranking was 0.268, bootstrap 95% CI [0.081, 0.452]. The positive rank relationship did not satisfy the frozen exact-winner criterion.
- **P2 shared-gradient-alignment predictor:** not supported. Exact final-winner prediction was 3/12. Mean Spearman correlation with the epoch-100 terminal-survival ranking was 0.306, bootstrap 95% CI [0.101, 0.516]. Again, the positive rank relationship did not satisfy the frozen exact-winner criterion.

Secondary timing/path results:

- Baseline commitment onset: epoch 35.
- Alternate-minibatch-order commitment onset: epoch 20.
- Baseline/alternate final winner agreement: 7/12.
- Mean baseline/alternate final terminal-ranking Spearman correlation: 0.442.

Because initialization and data were held fixed between the baseline and alternate-order paths, these secondary results show that minibatch order materially affects when specialization commits and often which relation becomes dominant. This is evidence of optimization-path sensitivity, not proof of strong emergence.

## Preserved interpretation

R8-M1 established that full specialization depends on encoder–recurrence coadaptation. R8-M2 ruled out two obvious fixed sources of winner identity under its preregistered tests. R8-M3 now shows that two plausible epoch-20 precommitment markers contain weak-to-moderate rank information but do not reliably identify the eventual winner, while changing only SGD example order materially changes the resulting dynamical organization.

The correct next question is functional necessity:

> Does the learned relation-selective dynamical specialization materially contribute to task performance, or can the system perform comparably when that specialization is suppressed?

## Claim boundary

R8-M3 does not establish strong emergence, a universal trajectory-information principle, new information creation, essential chronology, practical superiority, or causation from either preregistered predictor.
