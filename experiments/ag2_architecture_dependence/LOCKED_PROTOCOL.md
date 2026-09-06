# AG2 Locked Protocol

**Status:** locked before AG2 primary analysis on 2026-09-03.

AG2 is a new comparative/descriptive branch following AG1. AR1–AR6 remain closed. AG2 contains no state, gate, weight, or operator intervention and does not create AR7.

## Reused systems

Architectures: attractor reference, vanilla RNN, GRU, leaky RNN.

Tasks: AG1 Task 1 and Task 2.

System seeds: 7, 19, 31, 43, 59.

AG1 dependent variables are frozen: locality `same-stage - far-stage` and the exact AG1 affine rescue fraction/denominator rule.

## Primary predictor set

A. adjacent-stage global affine prediction quality

B. local-transform dispersion

C. neighborhood survival

D. task-subspace turn rate

E. Jacobian dispersion

F. recurrent/input-drive ratio

G. effective-rank change

H. cross-stage individual correspondence retrieval

All representation predictors use train-only fitting, validation-only hyperparameter selection where applicable, and frozen test evaluation. The independently trained neural system is the replication unit.

Primary neighborhood k is 10; sensitivity values are 5, 10, and 20. Ridge grid is `[1e-4, 1e-3, 1e-2, 1e-1, 1]`. The standardized nonlinear control is a one-hidden-layer width-64 model and may not be escalated architecture by architecture.

Primary representations are recurrent state `h_t` and unit displacement `u_t`. Geometry is evaluated within coarse Y where required.

## Controls

Required controls include C-label permutation within Y, example-correspondence shuffling within Y, parameter count, coarse Y performance, same-stage C accuracy, and representation scale.

## Success levels

Level 0: AG1 contrast only.

Level 1: at least one preregistered property strongly separates the Task-1 attractor reference from non-attractor systems.

Level 2: the same discriminator/direction appears across both controlled tasks.

Level 3: multiple diagnostics converge on one transport regime.

Level 4: a preregistered property predicts affine rescue across independent systems/seeds/tasks.

Level 5: a compact preregistered measurement set explains a substantial reproducible portion of alignability differences across both tasks.

No causal success level exists in AG2.

## Claim boundary

Preferred language: stage-local representation, cross-stage alignability, coherent representational transport, population-consistent transformation, example-conditioned transformation, successive re-expression, representational reorganization.

Do not claim information moves or is destroyed, that the network reconstructs information, that Jacobians/gates cause locality, that canonical coordinates are true content, or that the attractor architecture preserves information better.

Two computational amendments were recorded during execution: a bounded standardized nonlinear-control fit and a globally reduced deterministic diagnostic sample after runtime limits were encountered. The latter was explicitly recorded after partial Task-1 intermediate values had been printed; all affected summaries were discarded and recomputed under the common reduced-sample rule.
