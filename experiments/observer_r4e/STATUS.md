# Observer-R4E — Learned Perturbation-Consequence Model

**Status: Phase-I model selection in progress; R4E test set remains sealed.**

Observer-R4E asks whether a non-oracle model can predict the downstream functional consequence of a candidate small perturbation to the frozen observer state before choosing whether to apply it.

The conceptual separation is:

`controllability != policy learnability != consequence predictability != objective sufficiency`

R4B established oracle controllability. R4C and R4D-1D did not establish direct non-oracle policy learnability. R4E therefore tests consequence predictability first. Phase II (label-free selection) is forbidden unless the preregistered Phase-I prediction gate passes.

## Frozen lineage

Primary source seeds: **7, 19, 43**.

The exact historical Observer-R2 checkpoint bytes were recovered on 2026-09-04. Their SHA-256 values match the hashes frozen in the historical R4D/R4E source manifest exactly:

| Seed | Observer SHA-256 |
|---:|---|
| 7 | `9664b6fa8a72e4c8fa663636a33237012dfb4b318f5ad231bb1621f12ff684e5` |
| 19 | `f774b6ec38a043826201e3a3e649d3123c1dc62e0ee9bd578e8db950531674d7` |
| 43 | `4053447456bffc44de11b6660e8f02df96b9379cdbfee478c20a4c554017783b` |

The R4E integrity gate passed 3/3 using those exact checkpoint bytes. No observer was retrained or substituted.

The temporary fresh-source reconstruction branch (seeds 11/23/47) is **not part of the scientific lineage**. It was abandoned before any fresh test evaluation once the exact historical observer bytes were recovered.

## Frozen R4E preregistration

Historical preregistration SHA-256:

`a390e96ce932204749ec088b0f90b25803549d4bac76e092015e85d93be2b073`

The frozen design uses one perturbation after the first geometry observation:

`a1' = a1 + delta`

and then continues the exact frozen observer through native `g1...g11`.

For each of the eight relation heads, the primary consequence target is the 16-D final-logit change:

`Delta_logits_r = logits_final_r(a1 + delta) - logits_final_r(a1)`.

The primary model input is exactly `[a1, g0, delta]` (8 + 19 + 8 = 35 dimensions). Correct labels, V2, Vret, future observations, future states, and relation-ID tensors are excluded from the consequence predictor.

## Direction banks

The preregistered banks are disjoint and antithetic:

- `V_train`: 64 signed directions / 32 +/- pairs
- `V_val`: 32 signed directions / 16 +/- pairs
- `V_test`: 32 signed directions / 16 +/- pairs
- `V_phase2`: 32 signed directions / 16 +/- pairs

RNG namespace: `Observer-R4E|seed|directions|split`.

Phase-I primary evaluation is `S_test x V_test`: unseen source memories and explicitly unseen perturbation directions. Test evaluation is not opened until all Phase-I predictors and controls are frozen.

## Validation-only locality audit

The preregistered primary scale `alpha = 0.02` passed for all three source seeds; no fallback was needed.

| Seed | epsilon | mean cos(a1+delta,a1) | immediate prediction-change rate |
|---:|---:|---:|---:|
| 7 | 0.01725 | 0.999796 | 3.43% |
| 19 | 0.01583 | 0.999795 | 3.12% |
| 43 | 0.01396 | 0.999785 | 2.99% |

Preregistered locality requirements are mean cosine >= 0.999 and immediate frozen-observer prediction-change rate <= 5%.

## Consequence data generated so far

Using training/validation trajectories only:

- 20,000 training memories per seed
- 8 deterministic candidate perturbations per training memory
- 160,000 training consequence examples per seed
- 20,000 validation consequence examples per seed
- eight 16-logit consequence vectors per perturbation

No Phase-I test endpoint has been inspected.

## First mechanistic baseline: global linear response

The simplest preregistered baseline fits a single constant `16 x 8` response matrix per relation:

`J_global^r delta`.

Validation-only aggregate results:

| Seed | cosine | explained variance | normalized MSE |
|---:|---:|---:|---:|
| 7 | **0.794** | **0.672** | 0.328 |
| 19 | **0.560** | **0.442** | 0.558 |
| 43 | **0.556** | **0.371** | 0.629 |

This already shows that candidate perturbation consequences are substantially predictable on validation data from a state-independent linear response field, especially in seed 7. It is **not** a Phase-I result because the frozen test set remains unopened.

The remaining preregistered hierarchy is:

`J_global delta -> J_hat(a1) delta -> J_hat(a1,g0) delta -> C_phi(a1,g0,delta)`.

The simplest model that explains held-out consequences will be preferred. If a linear response model matches the nonlinear predictor, the interpretation will remain linear rather than being upgraded to a nonlinear/self-modeling story.

## Scientific boundary

A positive R4E result would not establish agency, self-understanding, information creation, a world model, extra Shannon information, or general perturbation utility. At most it can establish local prediction of candidate perturbation consequences and—only if Phase II is earned—label-free consequence-based action selection.
