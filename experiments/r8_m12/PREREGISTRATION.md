# R8-M12 Preregistration — Pre-History Susceptibility Predictor

**Status: FROZEN BEFORE ANY R8-M12 FAMILY OUTCOME IS INSPECTED.**

## 1. Question

R8-M10 showed strong axis specificity: the persistent A/B reorganization followed the historically demanded A/B axis, while an equally strong C/D history reorganized C/D and left A/B comparatively flat. The effect was reproducible but highly heterogeneous across families.

R8-M11 independently localized a reproducible causal contribution to the input-facing recurrent stage `F1 = Linear(16,32)`.

R8-M12 asks:

> Before divergent history begins, does the mature model already expose a measurable F1-localized susceptibility that predicts how large its later axis-specific persistent reorganization will become?

This is a **prediction/localization** experiment. Even a positive result is not causal proof of why heterogeneity exists.

## 2. Parent experiment

R8-M12 reuses the exact R8-M10 architecture, maturity rule, A/B and C/D pair selection, four forked histories, and terminal metrics. The frozen R8-M10 `m7r_base.py` is imported directly from `experiments/r8_m10/`.

Histories remain:

```
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
```

Four lineages are forked from one mature state:

- `TRUE_A`: A/B axis with A_HISTORY
- `TRUE_B`: A/B axis with B_HISTORY
- `NULL_C`: C/D axis with A_HISTORY
- `NULL_D`: C/D axis with B_HISTORY

Primary parent quantities at post-maturity epoch 120:

```
H_true_AB   = Q_AB(TRUE_B) - Q_AB(TRUE_A)
H_null_AB   = Q_AB(NULL_D) - Q_AB(NULL_C)
H_null_CD   = Q_CD(NULL_D) - Q_CD(NULL_C)
H_true_CD   = Q_CD(TRUE_B) - Q_CD(TRUE_A)
SPECIFICITY = H_true_AB - H_null_AB
```

## 3. Pre-history susceptibility diagnostic

The diagnostic is measured **after maturity is declared but before any history lineage is trained**.

A fresh deterministic 2,048-example diagnostic memory bank is generated under the R8-M12 namespace and is never used for training or for the primary terminal outcome.

For an axis `(P,Q)`, define two losses on the same diagnostic examples and permutations:

- `L_P`: demand endpoint `lambda=0.0`
- `L_Q`: demand endpoint `lambda=1.0`

For a parameter block `c`, compute gradient vectors

```
g_P,c = grad_theta_c L_P
g_Q,c = grad_theta_c L_Q
```

without updating any parameter.

The frozen susceptibility score is

```
S_c = ||g_Q,c - g_P,c|| /
      (0.5 * (||g_Q,c|| + ||g_P,c||) + 1e-12)
```

This measures how different the two opposing demand directions look to that parameter block relative to its ordinary gradient scale.

Blocks:

- `F1`: `F.0.weight`, `F.0.bias` — **primary predictor**
- `F2`: `F.2.weight`, `F.2.bias` — matched recurrent-stage control
- `E`: relation/value embeddings, GRU encoder, and `to_h` — upstream control
- `R`: `head0` and `headT` readers — descriptive control

For each block also record `||g_P||`, `||g_Q||`, raw contrast norm, and gradient cosine.

The mature model state is hash-checked before and after the diagnostic. Any mutation is a validity failure.

The primary predictor is `S_F1_AB`. The primary response is later `SPECIFICITY`.

`S_F1_CD` and all non-F1 scores are secondary/descriptive.

## 4. Why relative gradient contrast is primary

M10 showed that realized update magnitudes were similar across true and null arms, so simple "one family moved more" is not a sufficient explanation. M11 localized a causal contribution to F1. R8-M12 therefore asks whether the **directional distinction between opposing demands at F1**, measured before the fork, predicts later history sensitivity.

The score is scale-normalized so it is not merely a proxy for absolute gradient size.

## 5. Fresh families

Twelve fresh R8-M12 families:

```
1341, 1359, 1376, 1394, 1412, 1429,
1447, 1465, 1483, 1500, 1518, 1536
```

All derived randomness uses `R8-M12|{seed}|{name}`.

These seeds are disjoint from R8-M9, R8-M10, R8-M11, and E1.

## 6. Parent replication gates

Before interpreting any predictor, the fresh families must reproduce the M10 phenomenon.

Bootstrap: 5,000 paired family resamples.

```
R:
  mean(H_true_AB) >= 0.50
  AND CI95_lower(H_true_AB) > 0

MC:
  mean(H_null_CD) >= 0.50
  AND CI95_lower(H_null_CD) > 0

FLAT:
  entire CI95(H_null_AB) inside (-0.25, +0.25)

SEP:
  mean(SPECIFICITY) >= 0.50
  AND CI95_lower(SPECIFICITY) > 0
```

If any parent gate fails, classification is `U0` and no susceptibility predictor is promoted.

## 7. Primary predictor gate

Across the 12 valid fresh families, compute

```
rho_F1 = Spearman(S_F1_AB, SPECIFICITY)
```

Support requires all three:

```
rho_F1 >= +0.60
bootstrap CI95_lower(rho_F1) > 0
two-sided permutation p < 0.05
```

The permutation test uses 20,000 deterministic response permutations.

If parent gates pass but this predictor gate fails:

`U1 — no preregistered F1 susceptibility predictor`

If it passes:

`U2 — pre-history F1 susceptibility predicts later specificity`

## 8. Localization descriptor

Compare the F1 correlation with the same frozen score from F2 and E:

```
D_F2 = rho_F1 - rho_F2
D_E  = rho_F1 - rho_E
```

A stronger localized descriptor requires:

```
D_F2 >= +0.20 and bootstrap CI95_lower(D_F2) > 0
D_E  >= +0.20 and bootstrap CI95_lower(D_E) > 0
```

If the primary predictor and both localization comparisons pass:

`U3 — F1-localized pre-history susceptibility predictor supported`

`U3` does not mean F1 is the unique cause or storage substrate.

## 9. Mandatory secondary reporting

Report without promotion:

- `S_F2_AB`, `S_E_AB`, `S_R_AB`
- `S_F1_CD`
- gradient cosines and raw contrast norms
- maturity epoch
- baseline validation combined and h0 accuracy
- baseline `Q_AB`
- baseline survival `G` and `C`
- all per-family primary effects
- all parent M10 gates

Secondary correlations may motivate later work but cannot replace a failed primary F1 gate.

## 10. Validity

A family is valid only if:

- M10 maturity is reached within the frozen cap
- A/B and C/D are disjoint
- the susceptibility diagnostic leaves the model state byte-identical
- all four history lineages begin from byte-identical model and optimizer states
- all lineages complete exactly 120 post-maturity epochs
- every primary value is finite

Any invalid family yields a validity classification, not a scientific result.

## 11. Interpretation boundary

A positive R8-M12 would support:

> Mature recurrent systems that present a larger F1-localized directional distinction between opposing A/B training demands tend to develop larger later axis-specific persistent reorganization.

It would **not** establish:

- that the predictor causes the later history effect
- that F1 uniquely stores history
- formal bistability or hysteresis
- information beyond the complete current state
- essential chronology
- a universal trajectory-information principle
- generalization beyond this synthetic recurrent system
