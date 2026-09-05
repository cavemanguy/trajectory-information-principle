# E1 — Persistent Calibration Adapter Benchmark

**Status: benchmark specification frozen before E1 benchmark-family outcome inspection.**

## Engineering question

> Can the recurrent-map substructure implicated by R8-M9/M11 be used as a compact online calibration adapter that retains a useful hidden sensor calibration through long neutral retraining better than ordinary full-model or matched-size alternatives?

E1 is an engineering benchmark, not a new claim about the trajectory-information hypothesis. Its purpose is to ask whether the previously observed history-sensitive recurrent mechanism buys anything practical under a deliberately simple hidden-calibration problem.

## Practical scenario

A sensor feeds a classifier. The underlying physical quantity and classification task stay the same, but the sensor can acquire one of two opposite calibration shears, `A` or `B`. The model is not given a mode bit.

The model receives a short supervised recalibration burst in the current drift. It then returns to a long period of neutral operation and continues learning on neutral data. After that neutral period, the prior drift can recur without another calibration burst.

Useful behavior means:

1. adapt to the drift during the short labeled calibration burst;
2. continue performing the neutral task during the long neutral period;
3. retain enough of the previous calibration to perform better when that drift returns;
4. do so with fewer trainable parameters than full-model online adaptation.

This is **parameter-state / online-learning memory**, not transient hidden-state memory.

## Synthetic sensor task

For each example draw latent clean state

`z ~ N(0, I_16)`.

A fixed four-class task matrix `W_task` is generated once from deterministic global seed `20260905`; the target is

`y = argmax(W_task z)`.

The neutral sensor returns

`x_0 = z + eta`,

with iid Gaussian observation noise `eta` of standard deviation `0.10`.

For each family, deterministic task-relevant unit directions `u` and `v` are generated and orthogonalized. Drift state `s in {-1,+1}` returns

`x_s = (I + s * gamma * u v^T) z + eta`,

with frozen `gamma = 0.75`.

Thus A and B are equal-magnitude opposite rank-one calibration shears. The input dimensionality and target mapping do not change, and no explicit drift identity is supplied to the model.

## Model

The base model intentionally mirrors the recurrent-map structure used in the R8 mechanism work while remaining a small standalone classifier:

- input width 16;
- encoder `Linear(16,16) -> Tanh`;
- autonomous recurrent map applied for 12 steps:
  - `F1 = Linear(16,32)`;
  - `GELU`;
  - `F2 = Linear(32,16)`;
  - `Tanh`;
- four-class linear reader from `h12`.

No external information enters after `h0`.

## Base training

For each family, train one base model on neutral sensor data only.

- train examples: 12,288;
- validation examples: 3,072;
- test examples: 6,144;
- batch size: 256;
- AdamW learning rate `1e-3`, weight decay `1e-4`;
- gradient clip 1.0;
- 80 neutral pretraining epochs;
- deterministic permutations and minibatch order under namespace `E1-CAL`.

Validity requires neutral test accuracy >= 0.80 before any calibration fork. Failure in any benchmark family yields `V0` and no engineering classification.

## Benchmark families

Twelve fresh engineering families are fixed:

`[1109, 1127, 1144, 1162, 1181, 1199, 1218, 1237, 1255, 1274, 1292, 1311]`

These are separate from the R8-M9/M11 research families and from the seed set in Luke's off-axis control PR.

## Online-adaptation conditions

Every condition starts from the exact same pretrained model state. Run both initial drift directions, A (`s=-1`) and B (`s=+1`), and average the two within family before cross-family aggregation.

### FULL

All model parameters are trainable during calibration and neutral hold.

### F1

Only `F1.weight` and `F1.bias` are trainable. Encoder, F2, and reader are frozen.

### F2

Only `F2.weight` and `F2.bias` are trainable. Encoder, F1, and reader are frozen.

F2 is the primary matched-size location control for F1.

### HEAD

Only the final four-class reader is trainable.

### NOADAPT

No parameters are updated. This is the zero-adaptation reference.

Parameter counts and optimizer-update element counts are recorded exactly for every condition.

## Calibration burst

For each initial drift A and B:

- fork the exact pretrained model;
- train the condition's allowed parameter set for 20 epochs on 4,096 drifted labeled examples;
- evaluate on a held-out 6,144-example test set from the same drift.

Define

`CAL_GAIN_c = ACC_cal,c - ACC_base_drift`,

where `ACC_base_drift` is the NOADAPT pretrained model accuracy on that drift before calibration.

## Neutral hold

Immediately after calibration, continue training the same allowed parameter set for **120 epochs** on neutral labeled data.

Record at +30, +60, +90, and +120 epochs:

- neutral test accuracy;
- prior-drift test accuracy;
- parameter distance from post-calibration state;
- parameter distance from pretrained base state.

The model receives no drift label and no prior-drift examples during the hold.

Define the primary retained gain at +120:

`RETAINED_GAIN_c = ACC_return,c(+120) - ACC_base_drift`.

This asks whether the previously calibrated model still has useful competence for the old drift after long neutral retraining, relative to simply never adapting.

Also report the descriptive retention fraction

`RET_FRAC_c = RETAINED_GAIN_c / CAL_GAIN_c`

when `CAL_GAIN_c > 0`.

## Switch challenge

After the +120 neutral hold, expose the model to the **opposite** drift and resume supervised adaptation with the same trainable parameter subset for 10 epochs.

Record opposite-drift test accuracy after epochs 1, 3, 5, and 10.

This is secondary and measures the cost of persistence: a useful persistent adapter should not become effectively impossible to retune.

## Primary engineering gates

All family-level statistics first average A-start and B-start results, then use a deterministic paired-family 5,000-resample bootstrap.

### Gate A — F1 can actually calibrate

F1 is calibration-viable iff:

1. mean `CAL_GAIN_F1 >= +0.05`; and
2. bootstrap 95% CI lower bound > 0.

If this fails, classification is **P0 — F1 calibration adapter not viable**.

### Gate N — neutral-operation noninferiority

At +120 neutral hold define

`D_neutral = ACC_neutral,F1 - ACC_neutral,FULL`.

F1 is neutral-operation viable iff:

1. mean `D_neutral >= -0.02`; and
2. bootstrap 95% CI lower bound > -0.05.

This prevents calling persistence useful if F1 merely refuses to relearn the neutral operating condition.

If Gate A passes but Gate N fails, classification is **P1 — calibration works but neutral-operation tradeoff is too large**.

### Gate Rfull — retention advantage over full-model online learning

Define

`D_full = RETAINED_GAIN_F1 - RETAINED_GAIN_FULL`.

Support requires:

1. mean `D_full >= +0.03`; and
2. paired bootstrap 95% CI lower bound > 0.

### Gate Rmatch — retention advantage over the matched-size F2 location control

Define

`D_F2 = RETAINED_GAIN_F1 - RETAINED_GAIN_F2`.

Support requires:

1. mean `D_F2 >= +0.02`; and
2. paired bootstrap 95% CI lower bound > 0.

### Efficiency gate

F1 must expose no more than 50% of the trainable parameters used by FULL online adaptation.

## Frozen E1 classifications

After validity and Gates A/N:

- **P2 — compact persistent adapter, no specific retention edge:** F1 calibrates and remains neutral-viable, but Rfull fails.
- **P3 — compact retention advantage over full adaptation:** Rfull passes but Rmatch fails. This supports a small-adapter benefit but not an F1-location-specific benefit.
- **P4 — F1-specific persistent-adapter advantage:** Rfull, Rmatch, and the efficiency gate all pass.

P4 is the strongest E1 engineering success state.

## Mandatory baselines and reporting

Report for FULL, F1, F2, HEAD, and NOADAPT:

- exact trainable parameter count;
- base drift accuracy;
- post-calibration accuracy and calibration gain;
- neutral accuracy at all hold checkpoints;
- prior-drift accuracy and retained gain at all hold checkpoints;
- retention fraction where defined;
- switch-challenge curve;
- per-family values, mean, median, min, max, and bootstrap CI;
- optimizer update count and trainable-parameter-element update count.

Also report an **explicit checkpoint oracle** descriptively: post-calibration accuracy is the performance obtainable if the exact calibrated parameter snapshot is stored and restored when the same drift returns. This oracle is not a fair no-mode baseline; it is included to show the ceiling provided by straightforward explicit storage.

## Interpretation rules

A successful E1 means only that a compact F1-focused online adapter provides a useful retention/efficiency tradeoff on this hidden sensor-calibration benchmark.

It does **not** establish:

- that F1 is universally the best adapter location;
- that implicit parameter memory is better than explicit mode storage when a reliable mode identifier exists;
- that the mechanism generalizes to real sensors;
- that the effect is runtime hidden-state memory;
- that trajectories are a universal encoding principle;
- that the R8 scientific claims become stronger because an engineering benchmark succeeds.

A failed E1 is also useful: it would show that the scientifically interesting persistent-history mechanism does not automatically provide an engineering advantage under this benchmark.
