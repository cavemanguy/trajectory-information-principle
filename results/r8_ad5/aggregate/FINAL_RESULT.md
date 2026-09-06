# R8-AD5 Final Result — Causal Motion Intervention

**Primary classification:** K0 — neither preregistered causal contrast supported

## Frozen gates

- G_DIRECTION_CAUSAL: **False**
- G_CURVATURE_ORIENTATION: **False**

## Primary summaries

- K_DIR: mean -0.016768; 95% CI [-0.022089, -0.011453]; positive 1/12
- K_DIR_CE: mean -0.044629; positive 1/12
- direction accuracy drop: mean +0.019194
- magnitude-only accuracy drop: mean +0.035962
- generic random accuracy drop: mean +0.020108
- K_CURV: mean -0.030544; 95% CI [-0.040007, -0.020060]; positive 1/12
- K_CURV_CE: mean -0.237157; positive 1/12
- azimuth/turn-orientation accuracy drop: mean +0.047965
- polar/scalar-turn accuracy drop: mean +0.078509
- maximum frozen geometry mismatch: 2.375e-06

## Per-lineage values

| seed | M | AD2 regime | drop dir | drop mag | K_DIR | drop az | drop pol | K_CURV |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1782 | 100 | QUASIPERIODIC_LIKE | +0.0036 | +0.0027 | +0.0008 | +0.0161 | +0.0266 | -0.0105 |
| 1801 | 120 | QUASIPERIODIC_LIKE | +0.0196 | +0.0355 | -0.0159 | +0.0717 | +0.1262 | -0.0545 |
| 1819 | 100 | PERIODIC | +0.0212 | +0.0374 | -0.0162 | +0.0308 | +0.0478 | -0.0170 |
| 1838 | 110 | MIXED | +0.0188 | +0.0345 | -0.0157 | +0.0261 | +0.0378 | -0.0117 |
| 1856 | 100 | PERIODIC | +0.0212 | +0.0420 | -0.0208 | +0.0759 | +0.1252 | -0.0493 |
| 1875 | 110 | SENSITIVE | +0.0292 | +0.0641 | -0.0348 | +0.0391 | +0.0695 | -0.0304 |
| 1893 | 100 | QUASIPERIODIC_LIKE | +0.0318 | +0.0589 | -0.0271 | +0.0763 | +0.1182 | -0.0419 |
| 1912 | 100 | QUASIPERIODIC_LIKE | +0.0114 | +0.0183 | -0.0070 | +0.0513 | +0.0973 | -0.0459 |
| 1930 | 100 | MIXED | +0.0077 | +0.0220 | -0.0143 | +0.0414 | +0.0804 | -0.0390 |
| 1949 | 120 | REGULAR_NONPERIODIC | +0.0204 | +0.0359 | -0.0155 | +0.0763 | +0.1165 | -0.0402 |
| 1967 | 100 | QUASIPERIODIC_LIKE | +0.0248 | +0.0519 | -0.0271 | +0.0557 | +0.0878 | -0.0321 |
| 1986 | 100 | MIXED | +0.0208 | +0.0283 | -0.0075 | +0.0149 | +0.0089 | +0.0060 |

## Claim boundary

R8-AD5 tests explicit transition splices in frozen mature recurrent systems. A positive contrast supports anisotropic downstream functional sensitivity aligned to native trajectory geometry under matched Euclidean splice size. It does not establish information beyond the complete Markov state-plus-map, state-independent motion information, essential chronology, a universal trajectory code, or novelty relative to all prior literature.
