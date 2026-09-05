# R8-M10 Final Result — Off-Axis History Specificity Control

**Primary classification:** S2 — strong axis specificity supported

## Frozen gates

- R (A/B replication, mean >= 0.5): **True**
- MC (C/D manipulation check, mean >= 0.5): **True**
- SEP (specificity, mean >= 0.5): **True**
- FLAT (entire H_null_AB CI within +/-0.25): **True**

## Frozen primary statistics

- `H_true_AB`: mean 1.339117; median 0.878892; 95% CI [0.6494483474164895, 2.048794496669607]; range [0.055066, 3.063253]; positive 12/12
- `H_null_AB`: mean 0.056966; median 0.027882; 95% CI [0.023119944966105248, 0.09635003396124724]; range [0.000836, 0.184794]; positive 12/12
- `H_null_CD`: mean 2.676555; median 2.754856; 95% CI [1.356608510619366, 4.020671728860852]; range [0.016572, 6.127819]; positive 12/12
- `H_true_CD`: mean 0.130641; median -0.004630; 95% CI [-0.054981330266372204, 0.44390036384377807]; range [-0.199846, 1.744231]; positive 6/12
- `SPECIFICITY`: mean 1.282150; median 0.705339; 95% CI [0.597382239057866, 1.966141082606261]; range [0.046955, 3.024180]; positive 12/12

## Per-family distribution

Reported because R8-M8's mean described no observed family.

| seed | A | B | C | D | `H_true_AB` | `H_null_AB` | `H_null_CD` | `H_true_CD` | `SPECIFICITY` |
|---|---|---|---|---|---|---|---|---|---|
| 1061 | 4 | 7 | 1 | 3 | +3.0633 | +0.0391 | +5.4101 | +0.1347 | +3.0242 |
| 1078 | 1 | 7 | 0 | 6 | +0.0551 | +0.0008 | +0.0417 | -0.0200 | +0.0542 |
| 1094 | 1 | 5 | 3 | 4 | +0.1138 | +0.0668 | +0.5606 | -0.0421 | +0.0470 |
| 1113 | 7 | 1 | 3 | 5 | +0.1084 | +0.0049 | +0.0688 | +0.0445 | +0.1035 |
| 1129 | 0 | 4 | 6 | 7 | +0.9509 | +0.1623 | +4.7539 | -0.1998 | +0.7885 |
| 1146 | 4 | 5 | 0 | 6 | +2.5992 | +0.0047 | +4.2209 | +1.7442 | +2.5945 |
| 1164 | 3 | 2 | 1 | 7 | +0.8069 | +0.1848 | +1.2888 | -0.1272 | +0.6221 |
| 1183 | 4 | 1 | 0 | 5 | +2.8164 | +0.0277 | +6.1278 | +0.1202 | +2.7887 |
| 1201 | 5 | 4 | 6 | 7 | +0.1334 | +0.0023 | +0.0166 | +0.0224 | +0.1311 |
| 1218 | 6 | 2 | 0 | 3 | +0.1441 | +0.0281 | +0.0729 | +0.0108 | +0.1160 |
| 1236 | 3 | 4 | 1 | 5 | +2.3088 | +0.0168 | +5.0252 | -0.0318 | +2.2920 |
| 1254 | 7 | 4 | 1 | 5 | +2.9692 | +0.1453 | +4.5313 | -0.0880 | +2.8239 |

## Secondary optimization diagnostics (not gates)

- TRUE_A: update_norm_mean 0.022209; grad_norm_mean 1.411531; fork_distance 84.373714
- TRUE_B: update_norm_mean 0.022235; grad_norm_mean 0.961890; fork_distance 87.893955
- NULL_C: update_norm_mean 0.022233; grad_norm_mean 0.805567; fork_distance 87.779841
- NULL_D: update_norm_mean 0.022410; grad_norm_mean 0.775342; fork_distance 88.373400

Identical loss weights on different relation pairs do not guarantee identical optimization pressure. These diagnostics let a reader judge whether an observed null-arm flatness could instead reflect the null arm being pushed less far. They are not gates.

## Claim boundary

Within this synthetic recurrent system, persistent A/B reorganization under matched present demand is specific to the historically demanded axis: an equally weighted off-axis demand history reorganized its own axis without producing a comparable A/B effect. This does not establish bistability, hysteresis, or generalization beyond this system.
