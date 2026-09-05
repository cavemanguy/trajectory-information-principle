# R8-M7 Preregistration — Reversible Demand Tracking

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION

## Question

> Does the same trained recurrent lineage reversibly reorganize native relation-selective dynamics when controlled terminal demand changes A → B → A?

This study is designed to distinguish mere seed-to-seed heterogeneity from demand-tracking reorganization in one fixed architecture.

## Lineage

Use the established 16-D R8 lineage:

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

Twelve fresh family seeds:

`[207, 223, 239, 254, 271, 288, 304, 321, 337, 354, 371, 389]`

These do not overlap known Observer/ALI/ND-R1/R8-M1 through R8-M6/R8-M5R fresh-seed sets.

## Baseline pretraining and target selection

Each seed is trained symmetrically for 60 epochs under the ordinary lineage objective.

At epoch 60, before any demand-phase outcome is generated:

- compute terminal natural-pair survival S_r(12) on the frozen pair bank;
- define **A** as the relation with maximum terminal survival;
- define **B** as the relation with minimum terminal survival.

This target-selection rule is deterministic and frozen. A and B are selected automatically from the baseline checkpoint; no human outcome selection is permitted.

The complete model state and optimizer state at epoch 60 are copied exactly into all post-baseline conditions.

## Post-baseline conditions

All conditions use identical training memories, presentation permutations, minibatch order, architecture, optimizer hyperparameters, and training duration. Only the preregistered loss-demand schedule differs.

Each demand phase lasts 20 epochs, giving checkpoints at epochs 60, 80, 100, and 120.

### SWITCH

Terminal h12 demand schedule:

- epochs 61–80: target A
- epochs 81–100: target B
- epochs 101–120: target A

The current target relation receives terminal loss weight 4.0; every other terminal relation receives weight 1.0. The weighted terminal loss is normalized by the sum of weights, so the terminal-loss scale remains comparable.

h0 loss remains symmetric.

### FIX

Terminal h12 demand remains on A in all three phases, with the same 4.0 versus 1.0 normalized weighting. This controls for continued training, extra terminal emphasis, and persistent relation-specific weighting.

### H0SWITCH

The target schedule is A → B → A, but the 4.0 versus 1.0 normalized weighting is applied only to the h0 loss. Terminal h12 loss remains symmetric. This controls for relation-specific objective weighting that does not directly place extra demand on the terminal recurrent readout.

## Native survival metric

For each relation r, define terminal survival S_r(12) using the same natural-pair procedure as R8-M5/M5R.

Define centered log-survival alignment

`L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)`.

Define the A/B differential

`Q = L_B - L_A`.

Thus:

- Q < 0 favors A;
- Q > 0 favors B.

Because A is the baseline survival winner and B the baseline loser, Q is expected to be negative at epoch 60 by construction. The hard test is whether SWITCH can drive Q positive under B demand and then negative again when A demand returns.

## Validity gate

For every fresh seed:

- epoch-60 baseline combined validation >= 0.38;
- epoch-60 baseline h0 validation >= 0.55;
- all three post-baseline conditions complete through epoch 120;
- no NaN/Inf in losses or survival metrics.

If any family violates the gate, classify:

**V — reversible-demand training validity failure**

and do not promote the primary result.

## Primary reversible-tracking criteria

All cross-seed intervals are deterministic 5,000-resample paired family bootstraps.

For SWITCH define:

- `Q_A1 = Q(epoch 80)`
- `Q_B = Q(epoch 100)`
- `Q_A2 = Q(epoch 120)`
- `AB_shift = Q_B - Q_A1`
- `BA_shift = Q_A2 - Q_B`

### T — reversible relative tracking

T is supported only if all four hold:

1. mean `AB_shift >= +0.75` and its 95% CI lower bound > 0;
2. mean `BA_shift <= -0.75` and its 95% CI upper bound < 0;
3. mean `Q_B >= +0.20` and its 95% CI lower bound > 0;
4. mean `Q_A2 <= -0.20` and its 95% CI upper bound < 0.

This requires the formerly least-preserved relation B to overtake A under B demand and for A to retake B when A demand returns.

## Specificity controls

### C1 — switch versus fixed-A control

At epoch 100:

`Q_B_SWITCH - Q_B_FIX >= +0.50`

with 95% CI lower bound > 0.

### C2 — terminal-demand tracking versus h0-weighting control

Define tracking amplitude

`AMP = 0.5 * ((Q_B - Q_A1) - (Q_A2 - Q_B))`.

Require:

`AMP_SWITCH - AMP_H0SWITCH >= +0.25`

with 95% CI lower bound > 0.

Specificity S is supported only if C1 and C2 both hold.

## Exact specialist reassignment criterion

For each SWITCH family record whether:

- B is the exact terminal-survival winner at epoch 100;
- A is the exact terminal-survival winner at epoch 120.

Exact reassignment E is supported if both events occur in at least 8 of 12 fresh families.

## Frozen classifications

After validity:

- **D0 — reversible demand tracking not supported:** T false.
- **D1 — reversible relative demand tracking supported:** T true, but S or E false.
- **D2 — demand-specific specialist reassignment supported:** T true, S true, E true.
- **V — reversible-demand training validity failure.**

No post-hoc threshold changes are permitted.

## Functional secondary analyses

Preregistered secondary analyses report, without changing D0/D1/D2:

- per-phase h12 accuracy for A and B;
- change in demanded-relation h12 accuracy after each switch;
- per-phase h0 accuracy for A and B;
- G and D at epochs 60/80/100/120;
- exact winner identities for all relations;
- relation-wise survival and h12 accuracy;
- whether stronger Q tracking is associated with larger target h12 gains;
- whether non-target relations are preserved or sacrificed during switching.

These analyses may motivate a later causal suppression test, but no suppression/hidden-state perturbation is part of the R8-M7 primary study.

## Strongest allowed conclusion

Under D2:

> Within this symmetric synthetic autonomous recurrent system, the same trained lineage can reversibly reassign native relation-selective preservation in response to controlled changes in terminal task demand, beyond continued-training and h0-weighting controls. This supports adaptive demand-sensitive dynamical organization in the tested architecture.

R8-M7 cannot establish conscious choice, universal trajectory computation, strong emergence, essential chronology, language-model generalization, or that the path carries information beyond the complete state.
