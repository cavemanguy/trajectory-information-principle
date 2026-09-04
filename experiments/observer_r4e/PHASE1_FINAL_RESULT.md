# Observer-R4E — Phase-I Final Result

**Status: CLOSED AT PHASE I. Phase II NOT RUN.**

The preregistered Phase-I gate failed, so the experiment stops before label-free perturbation selection.

## Primary gate result

The declared nonlinear consequence predictor `C_phi(a1,g0,delta)` was required to beat all six non-oracle controls in all three source seeds, with positive explained variance and a paired-bootstrap cosine CI above the strongest control.

| Seed | C_phi cosine | strongest control | control cosine | C_phi EV | bootstrap C_phi-control 95% CI | pass |
|---:|---:|---|---:|---:|---|---|
| 7 | 0.689 | delta-only | 0.722 | 0.852 | [-0.035, -0.031] | NO |
| 19 | 0.420 | delta-only | 0.363 | 0.704 | [0.055, 0.060] | YES |
| 43 | 0.296 | delta-only | 0.498 | 0.484 | [-0.211, -0.193] | NO |

Seed 19 passed individually. Seeds 7 and 43 failed because the nonlinear full model was worse than the delta-only control. Therefore the all-three-seed Phase-I gate fails. Per the frozen stopping rule, Phase II is not run.

## Concurrent mechanistic hierarchy

Despite the primary gate failure, the preregistered linear-response hierarchy generalized strongly to `S_test x V_test`:

| Seed | J_global cosine | J(a1) cosine | J(a1,g0) cosine | J(a1,g0) EV |
|---:|---:|---:|---:|---:|
| 7 | 0.752 | 0.833 | 0.855 | 0.882 |
| 19 | 0.647 | 0.711 | 0.798 | 0.823 |
| 43 | 0.572 | 0.658 | 0.757 | 0.680 |

The ordering `J_global < J(a1) < J(a1,g0)` holds in all three source seeds.

Memory-level paired bootstrap comparisons:

| Seed | J(a1)-global cosine | J(a1,g0)-J(a1) cosine | J(a1,g0)-global cosine |
|---:|---|---|---|
| 7 | 0.081 [0.075, 0.086] | 0.022 [0.020, 0.025] | 0.103 [0.098, 0.109] |
| 19 | 0.064 [0.059, 0.068] | 0.088 [0.082, 0.093] | 0.151 [0.144, 0.159] |
| 43 | 0.086 [0.078, 0.093] | 0.099 [0.093, 0.107] | 0.185 [0.178, 0.192] |

This is consistent with a state-conditioned local response field, with first-transition geometry `g0` adding predictive susceptibility information beyond observer state `a1` alone. Because the explicit operator model outperforms the generic nonlinear predictor, the result should not be framed as a nonlinear self-model.

## Antisymmetry diagnostic

For true paired `+delta,-delta` consequences:

| Seed | mean A | median A | mean kappa |
|---:|---:|---:|---:|
| 7 | 0.0170 | 0.0125 | 0.00831 |
| 19 | 0.0109 | 0.0073 | 0.00211 |
| 43 | 0.0107 | 0.0084 | 0.00231 |

`A` near zero indicates that the local perturbation response is close to antisymmetric, supporting a first-order local linear-response interpretation at the preregistered epsilon.

## Interpretation

Strict preregistered evidence level: **Level 0**, because the declared Phase-I gate failed.

However, the concurrent mechanistic diagnostics show that downstream perturbation consequences are not generally inaccessible: a structured state-and-geometry-conditioned linear operator predicts them well on unseen memories and unseen directions. This is preserved as a mechanistic finding, not promoted into Phase-II control evidence.

No oracle/correct-answer information was used by the consequence predictors. No Phase-II perturbation selection was performed. No rescue with larger epsilon, additional axes, repeated perturbations, V2/Vret supervision, or recurrence is permitted.
