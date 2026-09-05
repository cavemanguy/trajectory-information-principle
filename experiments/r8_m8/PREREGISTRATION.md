# R8-M8 Preregistration — Demand Hysteresis and Regime Persistence

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION

## Question

> After a recurrent lineage has matured into a stable native A/B specialization pattern, can opposite demand histories produce different native dynamical organization under the same current demand, and does that path dependence persist under prolonged identical demand?

R8-M8 follows R8-M7R and R8-M7I. R8-M7R showed demand-sensitive movement without reliable specialist takeover. R8-M7I remains formally V0 because of its preregistered baseline-Q reproduction gate; its post-primary mirror pattern suggested that prior reinforcement changed later resistance. R8-M8 tests history dependence directly rather than attempting another binary A→B→A switch.

## Architecture and data

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

## Fresh family seeds

Twelve fresh family seeds are fixed:

`[421, 438, 454, 471, 489, 506, 523, 541, 558, 574, 592, 611]`

Use a new deterministic seed namespace `R8-M8` for all generated data, initialization, presentation permutations, and minibatch ordering.

## Maturity trigger

Train each family under the ordinary symmetric lineage objective. Record checkpoints every 10 epochs beginning at epoch 40, with a hard maximum baseline epoch of 400.

A checkpoint is competent iff:

- combined validation >= 0.38; and
- h0 validation >= 0.55.

The maturity checkpoint M is the first checkpoint for which:

1. the current checkpoint and the preceding two 10-epoch checkpoints are all competent; and
2. the terminal-survival winner A and loser B are unchanged between the current and immediately preceding checkpoint.

No human chooses M, A, or B.

If any family fails to reach M by epoch 400, the cross-family frozen outcome is:

**V0 — maturity validity failure.**

At M, save the full model and optimizer state. Define:

- A = terminal-survival winner at M;
- B = terminal-survival loser at M.

A and B must be distinct.

## Continuous demand parameter

Define terminal-demand parameter λ in [0,1]. h0 loss remains symmetric throughout all post-maturity conditions.

For terminal h12 loss:

- `w_A = 1 + 3*(1-λ)`
- `w_B = 1 + 3*λ`
- every other relation has weight 1.

Normalize the weighted terminal loss by the total weight. Because `w_A + w_B = 5`, total terminal loss weight is constant across λ.

Thus:

- λ=0 favors A 4:1 over B;
- λ=0.5 weights A and B equally at 2.5 each;
- λ=1 favors B 4:1 over A.

## Opposite-history sweeps

Fork the exact M model and optimizer state into two conditions.

### A_SWEEP — ascending demand

- λ=0.00 for 60 epochs;
- λ=0.25 for 30 epochs;
- λ=0.50 for 30 epochs;
- λ=0.75 for 30 epochs;
- λ=1.00 for 30 epochs.

Record Q and validation metrics after each level.

### B_SWEEP — descending demand

- λ=1.00 for 60 epochs;
- λ=0.75 for 30 epochs;
- λ=0.50 for 30 epochs;
- λ=0.25 for 30 epochs;
- λ=0.00 for 30 epochs.

Record Q and validation metrics after each level.

The two branches therefore reach λ=0.50 after the same total post-maturity training time: 120 epochs. Their current architecture, data generator, optimizer hyperparameters, cumulative epoch count, and current λ are matched; only their preceding demand history differs.

## Native organization metric

Retain the R8-M7R/M7I metric. For relation r:

`L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)`

and

`Q = L_B - L_A`.

Positive Q favors B; negative Q favors A.

For matched demand level λ define:

`H(λ) = Q_B_SWEEP(λ) - Q_A_SWEEP(λ)`.

Under path dependence seeded by the opposite histories, the preregistered direction is H(λ)>0: the branch arriving from B should remain more B-favoring than the branch arriving from A at the same current demand.

## Primary matched-midpoint history test H

At λ=0.50, define:

`H_mid = Q_B_SWEEP(0.50) - Q_A_SWEEP(0.50)`.

Across the 12 fresh families, H is supported only if:

1. mean `H_mid >= +0.50`; and
2. the deterministic 5,000-resample paired-family bootstrap 95% CI lower bound is >0.

This is the primary history-dependence test because both branches have the same current demand and the same cumulative post-maturity training duration.

## Sweep-loop test L

For each family compute H at λ = [0, 0.25, 0.50, 0.75, 1]. Define signed trapezoidal loop area:

`AREA = integral_0^1 H(λ) dλ`

using the five recorded λ levels.

L is supported only if:

1. mean `AREA >= +0.25`; and
2. its paired-family bootstrap 95% CI lower bound is >0.

Report H(λ) at every level regardless of classification.

## Long-hold persistence test P

At the first λ=0.50 checkpoint in each sweep, clone the complete model and optimizer state into paired hold branches:

- `A_HOLD` starts from the A_SWEEP midpoint state;
- `B_HOLD` starts from the B_SWEEP midpoint state.

Train both for an additional 120 epochs at identical λ=0.50, recording at +30, +60, +90, and +120 hold epochs.

Define:

`H_hold120 = Q_B_HOLD(+120) - Q_A_HOLD(+120)`.

P is supported only if:

1. mean `H_hold120 >= +0.25`; and
2. its paired-family bootstrap 95% CI lower bound is >0.

Also report the fraction of the initial midpoint separation retained after +120 epochs:

`retention = mean(H_hold120) / mean(H_mid)`

when mean(H_mid) is nonzero. Retention is descriptive and does not change P.

This long-hold test distinguishes persistent path dependence from a transient lag that collapses under prolonged identical demand.

## Execution validity

Every family must:

- reach a valid M;
- fork identical model and optimizer states into A_SWEEP and B_SWEEP;
- complete both sweeps;
- fork exact midpoint states into A_HOLD and B_HOLD;
- complete both holds through +120;
- contain no NaN/Inf in loss, Q, survival, or validation metrics.

Any post-maturity violation yields:

**V1 — post-maturity execution validity failure.**

No floating-point reproduction gate against a previous run is used. Fresh lineages are validated internally by exact fork-state hashes plus the frozen maturity rule.

## Frozen classifications

After validity:

- **Y0 — matched history dependence not supported:** H false.
- **Y1 — matched midpoint history dependence supported:** H true, L false.
- **Y2 — hysteresis-like sweep separation supported:** H true, L true, P false.
- **Y3 — persistent history-dependent regime separation supported:** H true, L true, P true.

No threshold, seed, λ schedule, phase duration, maturity rule, or classification rule may be changed after fresh outcomes are exposed.

## Secondary analyses

Without changing Y0/Y1/Y2/Y3, report:

- maturity epoch M, A, and B per family;
- Q and exact winner at every λ on both sweeps;
- H(λ) pointwise with bootstrap intervals;
- relation-wise survival and h12 accuracy at each λ;
- A/B h12 accuracy differences between matched-history branches;
- G and D along both sweeps;
- midpoint state distance between A_SWEEP and B_SWEEP;
- hold-decay curve H at +30/+60/+90/+120;
- whether exact A/B winner identity remains different under matched λ=0.50;
- relationship between H_mid and functional A/B accuracy differences.

## Strongest allowed conclusion

Under Y3, the strongest allowed conclusion is:

> Within this symmetric synthetic autonomous recurrent system, opposite controlled demand histories can leave the same mature lineage in persistently different native dynamical organizations under the same current functional demand and matched training duration, with a direction-consistent sweep loop that survives a prolonged identical-demand hold.

Even Y3 does not establish mathematical bistability, formal thermodynamic hysteresis, conscious choice, universal trajectory computation, essential chronology, information beyond the complete state, or generalization to language models, biological systems, or physical systems.
