# R8-AD4 Final Result — Motion-Component Decomposition

**Primary classification:** D1 — direction-preserving decomposition

## Frozen gates

- G_FULL_REPLICATION: **True**
- G_DIRECTION_PRESERVES: **True**
- G_MAGNITUDE_PRESERVES: **False**
- G_CURVATURE_ADDS: **True**

## Primary summaries

- DELTA_FULL: mean +0.057167; 95% CI [+0.053931, +0.060419]; positive 12/12
- DELTA_SHUFFLE: mean +0.058620; 95% CI [+0.055523, +0.061692]; positive 12/12
- DELTA_DIRECTION: mean +0.054076; 95% CI [+0.050327, +0.058122]; positive 12/12
- DELTA_MAGNITUDE: mean +0.002781; 95% CI [+0.002189, +0.003366]; positive 12/12
- RET_DIRECTION: mean +0.943793; 95% CI [+0.922582, +0.964749]; positive 12/12
- RET_MAGNITUDE: mean +0.048058; 95% CI [+0.039210, +0.056738]; positive 12/12
- DELTA_CURVATURE: mean +0.044998; 95% CI [+0.043357, +0.046569]; positive 12/12
- DELTA_TURN: mean +0.000755; 95% CI [+0.000562, +0.000962]; positive 12/12

## Per-lineage values

| seed | M | Δfull | Δdir | Δmag | ret dir | ret mag | Δcurv |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1782 | 100 | +0.0494 | +0.0463 | +0.0017 | 0.938 | 0.034 | +0.0423 |
| 1801 | 120 | +0.0635 | +0.0621 | +0.0043 | 0.978 | 0.068 | +0.0460 |
| 1819 | 100 | +0.0562 | +0.0506 | +0.0028 | 0.900 | 0.049 | +0.0432 |
| 1838 | 110 | +0.0534 | +0.0482 | +0.0021 | 0.902 | 0.040 | +0.0446 |
| 1856 | 100 | +0.0560 | +0.0538 | +0.0014 | 0.959 | 0.024 | +0.0443 |
| 1875 | 110 | +0.0650 | +0.0653 | +0.0043 | 1.005 | 0.066 | +0.0485 |
| 1893 | 100 | +0.0608 | +0.0583 | +0.0039 | 0.958 | 0.064 | +0.0473 |
| 1912 | 100 | +0.0583 | +0.0511 | +0.0014 | 0.876 | 0.024 | +0.0485 |
| 1930 | 100 | +0.0663 | +0.0647 | +0.0034 | 0.976 | 0.051 | +0.0484 |
| 1949 | 120 | +0.0533 | +0.0513 | +0.0035 | 0.961 | 0.066 | +0.0452 |
| 1967 | 100 | +0.0560 | +0.0541 | +0.0020 | 0.966 | 0.037 | +0.0423 |
| 1986 | 100 | +0.0477 | +0.0432 | +0.0025 | 0.906 | 0.053 | +0.0391 |

## Claim boundary

R8-AD4 decomposes the previously established one-step linear-accessibility gain into normalized displacement direction and scalar magnitude, with prespecified curvature and basis-dependent coordinate analyses. It does not establish information beyond the complete Markov state-plus-map, causal necessity, essential chronology, or a universal trajectory code.
