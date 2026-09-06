# R9-H1 Frozen Classifier Specification

This specification is frozen before scientific outcomes are inspected.

## Validity

For each architecture / interaction arm:

- `V+` iff mean held-out `DATA_ACC >= 0.20` and mean held-out `RETURN_EARLY_ACC >= 0.15` across all eight fresh seeds.
- otherwise `V-`.

Validity is an interpretation floor, not a performance ranking.

## Bootstrap

All directional cross-seed contrasts use a deterministic percentile bootstrap over the eight family seeds with 20,000 resamples and RNG seed 91091. Report the mean and 95% percentile CI.

## Generic anomaly classification

For a directional contrast where positive is the preregistered effect:

- `A+` iff the contrast is positive in at least 6/8 seeds and the bootstrap 95% CI lower bound is > 0.
- `A-` iff the contrast is negative in at least 6/8 seeds and the bootstrap 95% CI upper bound is < 0.
- otherwise `A0`.

For a non-directional serial-order contrast, classify:

- `A+` iff at least 6/8 seeds share the positive sign and the CI excludes zero on the positive side;
- `A-` iff at least 6/8 seeds share the negative sign and the CI excludes zero on the negative side;
- otherwise `A0`.

Here `A+` and `A-` indicate reproducible sign, not better/worse architecture.

## Answer-channel confound

A controller / perturbation mechanism is `AX` if either of the following holds at aggregate level:

1. mean standalone controller-signal target decoding accuracy is >= 0.50 and is within 0.10 absolute accuracy of the arm's own mean DATA accuracy; or
2. the controller-signal decoder exceeds the current-payload context-only decoder by >= 0.35 absolute accuracy.

This is intentionally conservative: such a signal can plausibly act as a direct answer channel rather than a dynamical teaching/interrogation mechanism.

## Frozen anomaly contrasts

### History accessibility

For every preregistered state stream:

`H1 = ACC(state + one-step displacement) - ACC(state)`

`H2 = ACC(state + multi-step displacement) - ACC(state)`

`HSHUF = ACC(state + one-step displacement) - ACC(state + shuffled displacement)`

Classify `H1` and `HSHUF` directionally. `H2` is reported descriptively and classified with the same positive criterion.

### Complementary module accessibility

For each preregistered module pair:

`COMP = ACC(joint) - max(ACC(module A), ACC(module B))`

`COMP_SHUF = ACC(joint) - ACC(module A + shuffled module B)`

A complementary-accessibility `A+` requires both `COMP` and `COMP_SHUF` independently classify `A+`. If either is `AX`, the family is `AX`; otherwise `A0` unless both pass.

### Cross-module transformation

`XFORM = final-module state-only hidden-key accuracy - first-module state-only hidden-key accuracy`

Classify by reproducible sign. The sign is descriptive reformatting, not information creation.

### Reactivation recovery

`REACT = hidden-key probe accuracy on fourth early-return DATA item - accuracy on first early-return DATA item`

Classify directionally positive. No chronology claim follows from this contrast alone.

### Perturbation / controller dependence

For each applicable arm:

`CTRL_SHUF = native RETURN_EARLY_ACC - shuffled-controller RETURN_EARLY_ACC`

`CTRL_ZERO = native RETURN_EARLY_ACC - controller-removed RETURN_EARLY_ACC`

`CTRL_RAND = native RETURN_EARLY_ACC - matched-random-controller RETURN_EARLY_ACC`

`CTRL_WRONG = native RETURN_EARLY_ACC - wrong-context-controller RETURN_EARLY_ACC`

A controller-dependence candidate requires at least three of the four contrasts to classify `A+`, including `CTRL_SHUF`. Answer-channel confounding overrides this classification to `AX`.

### Serial-order asymmetry

Two frozen non-directional contrasts:

`ORDER_H = H1(final state of R-T-R) - H1(final state of T-R-T)`

`ORDER_X = XFORM(R-T-R) - XFORM(T-R-T)`

Report sign and CI separately. Do not collapse them into a performance ranking.

### Parallel specialization

Use the `R || T` pair:

- complementary-accessibility result above;
- `PAR_H = H1(RNN branch) - H1(Transformer branch)` as a signed specialization contrast;
- branch-bypass task effects are reported descriptively.

### Teacher-induced persistence

For the Transformer-above-RNN perturbation teacher:

`PERSIST_TASK = teacher-off RETURN_EARLY_ACC - never-taught control-RNN RETURN_EARLY_ACC`

`PERSIST_H = H1(teacher-trained RNN with teacher off) - H1(never-taught control-RNN)`

A persistent-teaching candidate requires at least one of these to classify `A+` and neither to classify `A-`. Immediate teacher-on gains do not count.

### Learning-trajectory coupling

For the update-gated arm, at fixed training checkpoints fit within each seed:

`next_update_norm ~ intercept + current_loss + current_controller_mean`

Record the standardized coefficient on `current_controller_mean` as `LEARN_BETA`.

Across seeds, classify its reproducible sign using the non-directional rule. This remains exploratory regardless of sign.

## Low-competence labeling

If an anomaly classifies `A+` or `A-` in an arm whose task validity is `V-`, retain the anomaly but label it `LOW_COMPETENCE_A+` or `LOW_COMPETENCE_A-` in the final report.

## No post-hoc rescue

No threshold, contrast, seed set, controller audit, or classification rule may be changed after family outcomes are available. Any correction after outcome inspection requires a new experiment identifier and preserved original result.
