# R8-M7R Preregistration — Competence-Triggered Reversible Demand Tracking

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent:** R8-M7 = **V — reversible-demand training validity failure**

## Question

> Once a recurrent lineage is demonstrably competent and its native A/B specialization identities are stable, does the same lineage reversibly reorganize relation-selective native dynamics under controlled terminal demand A → B → A?

R8-M7 failed because all 12 fresh families were forked at epoch 60 before satisfying its preregistered competence gate. R8-M7 remains V. R8-M7R changes only the baseline-fork rule and phase duration; it does not reclassify R8-M7.

## Architecture and data

Use the same established 16-D R8 lineage and data generator as R8-M7:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embedding width 8;
- encoder GRU hidden width 32;
- latent/recurrent state width 16;
- recurrent transition hidden width 32;
- 12 autonomous recurrent transitions after h0;
- separate h0 and h12 linear heads per relation;
- AdamW lr 1e-3, weight decay 1e-4;
- batch 256; gradient clip 1.0;
- 20,000 train / 2,500 validation / 5,000 test memories;
- natural-pair bank 2,048 per relation.

No external information enters after h0.

## Fresh seeds

Twelve new family seeds are fixed:

`[214, 230, 247, 263, 279, 296, 313, 329, 346, 362, 378, 397]`

They do not overlap the known Observer/ALI/ND-R1/R8-M1 through R8-M7/R8-M5R fresh-seed sets.

## Symmetric baseline training

Train each family under the ordinary symmetric lineage objective. Record evaluation checkpoints every 10 epochs beginning at epoch 40 and continuing through a hard maximum of epoch 400. A maturity trigger is not eligible before epoch 60; epochs 40 and 50 exist only so the frozen three-checkpoint competence rule can be evaluated at epoch 60 if appropriate.

A checkpoint is **competent** iff:

- combined validation >= 0.38; and
- h0 validation >= 0.55.

At every evaluation checkpoint compute terminal natural-pair survival and record the current survival winner and loser.

## Frozen maturity trigger

The maturity checkpoint **M** is the first eligible checkpoint at epoch 60 or later satisfying both:

1. the current checkpoint and the preceding two 10-epoch checkpoints are all competent; and
2. the terminal-survival winner identity A and loser identity B are unchanged between the current checkpoint and the immediately preceding checkpoint.

Thus the model must demonstrate competence across 20 epochs of continued training and locally stable A/B identities before the demand challenge begins.

No human chooses M, A, or B. The first checkpoint satisfying the rule is used automatically.

If no maturity checkpoint is reached by epoch 400, classify the family as a maturity failure. If any fresh family has no valid maturity checkpoint, the cross-family frozen outcome is:

**V0 — competence/stability maturity failure**

and D0/D1/D2 are not promoted.

At M, save the complete model and optimizer state. Define:

- **A** = terminal-survival winner at M;
- **B** = terminal-survival loser at M.

The exact model and optimizer state at M are copied into all post-maturity conditions.

## Post-maturity demand conditions

Each demand phase lasts **40 epochs**. This gives relative checkpoints M+40, M+80, M+120.

All conditions share identical training memories, presentation permutations, minibatch order, architecture, optimizer state at the fork, hyperparameters, and duration.

### SWITCH

Terminal h12 demand schedule:

- M+1 through M+40: A;
- M+41 through M+80: B;
- M+81 through M+120: A.

The current target receives terminal-loss weight 4.0; every other terminal relation receives weight 1.0. Weighted terminal loss is normalized by total weight. h0 loss remains symmetric.

### FIX

Terminal h12 demand remains on A for all three phases using the same normalized 4.0-versus-1.0 weighting.

### H0SWITCH

The A → B → A schedule is applied only to h0 loss. Terminal h12 loss remains symmetric.

## Native survival metric

Use the same R8-M7 metric.

For relation r:

`L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)`

and

`Q = L_B - L_A`.

Q < 0 favors A; Q > 0 favors B.

For SWITCH define:

- `Q_A1 = Q(M+40)`;
- `Q_B = Q(M+80)`;
- `Q_A2 = Q(M+120)`;
- `AB_shift = Q_B - Q_A1`;
- `BA_shift = Q_A2 - Q_B`.

## Post-fork validity

Every family must:

- reach a frozen maturity checkpoint M;
- copy the identical model state into SWITCH/FIX/H0SWITCH;
- complete all three conditions through M+120;
- contain no NaN/Inf in losses, validation, Q, or survival metrics.

Any violation after a valid M yields:

**V1 — post-maturity execution validity failure**

and D0/D1/D2 are not promoted.

## Primary reversible-tracking criteria

All cross-seed intervals use deterministic 5,000-resample paired family bootstraps.

### T — reversible relative tracking

T is supported only if all four hold:

1. mean `AB_shift >= +0.75` and 95% CI lower bound > 0;
2. mean `BA_shift <= -0.75` and 95% CI upper bound < 0;
3. mean `Q_B >= +0.20` and 95% CI lower bound > 0;
4. mean `Q_A2 <= -0.20` and 95% CI upper bound < 0.

This requires B to overtake A under B demand and A to retake B after A demand returns.

## Specificity controls

### C1 — SWITCH versus FIX

At M+80 require:

`Q_B_SWITCH - Q_B_FIX >= +0.50`

with 95% CI lower bound > 0.

### C2 — terminal-demand versus h0-weighting

Define:

`AMP = 0.5 * ((Q_B - Q_A1) - (Q_A2 - Q_B))`.

Require:

`AMP_SWITCH - AMP_H0SWITCH >= +0.25`

with 95% CI lower bound > 0.

Specificity S requires C1 and C2.

## Exact specialist reassignment

For every SWITCH family record whether:

- B is the exact terminal-survival winner at M+80; and
- A is the exact terminal-survival winner at M+120.

Exact reassignment E is supported if both events occur in at least 8 of 12 fresh families.

## Frozen classifications

After maturity and execution validity:

- **D0 — reversible demand tracking not supported:** T false.
- **D1 — reversible relative demand tracking supported:** T true, but S or E false.
- **D2 — demand-specific specialist reassignment supported:** T true, S true, E true.
- **V0 — competence/stability maturity failure.**
- **V1 — post-maturity execution validity failure.**

No post-hoc threshold, maximum-epoch, phase-length, A/B selection, or maturity-rule changes are permitted.

## Secondary analyses

Without changing D0/D1/D2, report:

- maturity epoch M per family;
- competence trajectory before M;
- A/B identity history around M;
- per-phase h12 and h0 accuracy for A and B;
- G and D at M, M+40, M+80, M+120;
- exact winner identities;
- relation-wise survival and h12 accuracy;
- demanded-relation accuracy changes;
- non-target preservation/sacrifice;
- association between Q tracking amplitude and demanded-relation functional gain.

## Strongest allowed conclusion

Under D2:

> Within this symmetric synthetic autonomous recurrent system, after each lineage independently reaches a preregistered competence-and-stability maturity criterion, the same trained lineage can reversibly reassign native relation-selective preservation in response to controlled terminal-demand changes A → B → A, beyond continued-training and h0-weighting controls.

R8-M7R cannot establish conscious choice, universal trajectory computation, strong emergence, essential chronology, language-model generalization, or that trajectory history carries information beyond the complete state.
