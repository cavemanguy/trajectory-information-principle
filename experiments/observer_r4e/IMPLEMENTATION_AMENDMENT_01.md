# Observer-R4E — Implementation Amendment 01

**Status: locked before any Phase-I test evaluation.**

The historical preregistration freezes the scientific model families and test gate but leaves several ordinary implementation degrees of freedom unspecified. This amendment fixes those choices before any R4E Phase-I test result is inspected.

## Candidate assignment

For each training memory, select four distinct antithetic pair IDs from the 32 `V_train` pairs using a deterministic label-free RNG namespace derived from:

`Observer-R4E|seed|candidate_assignment|train|memory_index`

Include both signs of each selected pair, producing exactly eight candidate perturbations per memory.

For validation, use the analogous namespace with `val` and select four distinct pairs from the 16 `V_val` pairs, again including both signs. No label, outcome, V2, Vret, future state, or test result enters assignment.

The frozen `V_test` bank is not rolled out during model selection.

## Optimization

All eight relation-specific predictors are parameter-independent; they may be optimized together in one program/optimizer for compute efficiency, but no parameter is shared across relation heads.

Common settings:

- optimizer: AdamW
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `4096`
- maximum epochs: `25`
- early-stopping patience: `4`
- minimum validation MSE improvement: `1e-5` relative to the predictor's validation objective
- gradient clipping: `1.0`
- no scheduler
- no dropout
- deterministic initialization namespace per seed/model family/width

## State-conditioned response operators

`J_hat(a1)` and `J_hat(a1,g0)` are independent per-relation matrix predictors. Each uses one GELU hidden layer with width in `{32,64}`, validation selected, and outputs 128 values reshaped to `16 x 8`; the predicted consequence is the resulting matrix multiplied by candidate `delta`.

## Nonlinear consequence model

`C_phi(a1,g0,delta)` uses two GELU hidden layers with equal width in `{32,64}`, validation selected, and a 16-D output, separately for each relation.

The declared input ablations use the same selected-capacity rule and may not exceed the primary nonlinear family.

## Selection

Width is selected independently within each source seed and model family by validation MSE. Ties within `1e-5` prefer width 32.

After width and checkpoint selection, all Phase-I models and controls are frozen. Only then may `S_test x V_test` be evaluated once.

Nothing in this amendment changes the original evidence gate, outcome definitions, direction banks, epsilon/locality rule, perturbation timing, target, controls, statistics, or Phase-II stopping rule.
