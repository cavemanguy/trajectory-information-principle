# R8-M10 Final Result — Off-Axis History Specificity Control

**Primary classification:** S2 — strong axis specificity supported

## Frozen gates

- R (A/B replication, mean >= 0.5): **True**
- MC (C/D manipulation check, mean >= 0.5): **True**
- SEP (specificity, mean >= 0.5): **True**
- FLAT (entire H_null_AB CI within +/-0.25): **True**

## Frozen primary statistics

- `H_true_AB`: mean 1.338824; median 0.871536; 95% CI [0.6488688748199405, 2.0466595123940774]; range [0.055066, 3.063253]; positive 12/12
- `H_null_AB`: mean 0.054429; median 0.028466; 95% CI [0.021055809634479686, 0.09195827342511044]; range [-0.018434, 0.178802]; positive 11/12
- `H_null_CD`: mean 2.680668; median 2.754541; 95% CI [1.3585185521897714, 4.0243270723872255]; range [0.040513, 6.127819]; positive 12/12
- `H_true_CD`: mean 0.131210; median -0.004630; 95% CI [-0.05500280198504993, 0.4440234879107199]; range [-0.203519, 1.744231]; positive 6/12
- `SPECIFICITY`: mean 1.284395; median 0.708861; 95% CI [0.6033822307309643, 1.9664704211708672]; range [0.046955, 3.024180]; positive 12/12

## Per-family distribution

Reported because R8-M8's mean described no observed family.

| seed | A | B | C | D | `H_true_AB` | `H_null_AB` | `H_null_CD` | `H_true_CD` | `SPECIFICITY` |
|---|---|---|---|---|---|---|---|---|---|
| 1061 | 4 | 7 | 1 | 3 | +3.0633 | +0.0391 | +5.4101 | +0.1347 | +3.0242 |
| 1078 | 1 | 7 | 0 | 6 | +0.0551 | +0.0008 | +0.0417 | -0.0200 | +0.0542 |
| 1094 | 1 | 5 | 3 | 4 | +0.1138 | +0.0668 | +0.5606 | -0.0421 | +0.0470 |
| 1113 | 7 | 1 | 3 | 5 | +0.1084 | +0.0049 | +0.0688 | +0.0445 | +0.1035 |
| 1129 | 0 | 4 | 6 | 7 | +0.9569 | +0.1465 | +4.7439 | -0.2035 | +0.8104 |
| 1146 | 4 | 5 | 0 | 6 | +2.5992 | +0.0047 | +4.2209 | +1.7442 | +2.5945 |
| 1164 | 3 | 2 | 1 | 7 | +0.7862 | +0.1788 | +1.2882 | -0.1179 | +0.6074 |
| 1183 | 4 | 1 | 0 | 5 | +2.8164 | +0.0277 | +6.1278 | +0.1202 | +2.7887 |
| 1201 | 5 | 4 | 6 | 7 | +0.1407 | -0.0184 | +0.0405 | +0.0279 | +0.1591 |
| 1218 | 6 | 2 | 0 | 3 | +0.1441 | +0.0281 | +0.0729 | +0.0108 | +0.1160 |
| 1236 | 3 | 4 | 1 | 5 | +2.3127 | +0.0289 | +5.0612 | -0.0362 | +2.2839 |
| 1254 | 7 | 4 | 1 | 5 | +2.9692 | +0.1453 | +4.5313 | -0.0880 | +2.8239 |

## Secondary optimization diagnostics (not gates)

- TRUE_A: update_norm_mean 0.022209; grad_norm_mean 1.406684; fork_distance 84.393220
- TRUE_B: update_norm_mean 0.022234; grad_norm_mean 0.963007; fork_distance 87.871450
- NULL_C: update_norm_mean 0.022240; grad_norm_mean 0.805139; fork_distance 87.809341
- NULL_D: update_norm_mean 0.022413; grad_norm_mean 0.775018; fork_distance 88.381379

Identical loss weights on different relation pairs do not guarantee identical optimization pressure. These diagnostics let a reader judge whether an observed null-arm flatness could instead reflect the null arm being pushed less far. They are not gates.

## Claim boundary

Within this synthetic recurrent system, persistent A/B reorganization under matched present demand is specific to the historically demanded axis: an equally weighted off-axis demand history reorganized its own axis without producing a comparable A/B effect. This does not establish bistability, hysteresis, or generalization beyond this system.
