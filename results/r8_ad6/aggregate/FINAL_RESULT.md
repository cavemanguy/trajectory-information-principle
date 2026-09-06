# R8-AD6 Final Result — Tangent vs Transverse Error Dynamics

**Primary classification:** P1 — tangent persistence with functional asymmetry supported

## Frozen gates

- G_RECOVERY: **True**
- G_FUNCTION: **True**

## Primary summaries

- D_REC: mean +0.505190; 95% CI [+0.409338, +0.596889]; positive 12/12
- tangent k=3 retention: mean 1.230823
- transverse k=3 retention: mean 0.725633
- PHASE_RET k=3: mean +0.535540; 95% CI [+0.427770, +0.640651]
- D_FUNC: mean +0.018986; 95% CI [+0.012918, +0.024908]; positive 11/12
- tangent terminal accuracy drop: mean +0.040240
- transverse terminal accuracy drop: mean +0.021254
- D_FUNC_CE: mean +0.053357
- maximum frozen geometry mismatch: 4.059e-07

## Per-lineage values

| seed | M | AD2 regime | ret tan | ret trans | D_REC | phase ret | drop tan | drop trans | D_FUNC |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1782 | 100 | QUASIPERIODIC_LIKE | 1.1082 | 0.8013 | +0.3068 | +0.8632 | +0.0042 | +0.0049 | -0.0008 |
| 1801 | 120 | QUASIPERIODIC_LIKE | 1.3076 | 0.7063 | +0.6013 | +0.5666 | +0.0394 | +0.0199 | +0.0196 |
| 1819 | 100 | PERIODIC | 1.2338 | 0.8052 | +0.4286 | +0.5297 | +0.0428 | +0.0226 | +0.0201 |
| 1838 | 110 | MIXED | 1.2096 | 0.7497 | +0.4599 | +0.5489 | +0.0388 | +0.0209 | +0.0179 |
| 1856 | 100 | PERIODIC | 1.2891 | 0.7151 | +0.5740 | +0.5101 | +0.0486 | +0.0253 | +0.0233 |
| 1875 | 110 | SENSITIVE | 1.3682 | 0.6751 | +0.6931 | +0.2623 | +0.0689 | +0.0327 | +0.0361 |
| 1893 | 100 | QUASIPERIODIC_LIKE | 1.3580 | 0.7040 | +0.6540 | +0.3840 | +0.0628 | +0.0319 | +0.0309 |
| 1912 | 100 | QUASIPERIODIC_LIKE | 1.0981 | 0.7493 | +0.3488 | +0.7504 | +0.0205 | +0.0141 | +0.0064 |
| 1930 | 100 | MIXED | 1.2710 | 0.6576 | +0.6134 | +0.7494 | +0.0259 | +0.0109 | +0.0150 |
| 1949 | 120 | REGULAR_NONPERIODIC | 1.1609 | 0.7433 | +0.4176 | +0.5143 | +0.0381 | +0.0206 | +0.0175 |
| 1967 | 100 | QUASIPERIODIC_LIKE | 1.5378 | 0.7721 | +0.7657 | +0.5636 | +0.0582 | +0.0253 | +0.0329 |
| 1986 | 100 | MIXED | 0.8277 | 0.6285 | +0.1992 | +0.1838 | +0.0347 | +0.0258 | +0.0089 |

## Claim boundary

R8-AD6 tests the preregistered tangent/phase explanation generated after AD5. It measures norm-matched tangent versus transverse recovery from the same native state and their downstream task consequences. Even a positive result does not establish information beyond the complete Markov state-plus-map, an independent hidden phase variable, essential chronology, a universal trajectory code, or generalization beyond the tested synthetic recurrent system.
