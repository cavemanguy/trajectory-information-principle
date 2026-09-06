# R8-AD1 Final Result — Learned Long-Run Attractor Diagnostic

**Primary classification:** A3 — fixed-point attractor behavior not supported

## Family classifications

- FIXED: **1/12**
- SHORT_CYCLE: **1/12**
- UNRESOLVED: **10/12**

## Cross-family diagnostics

- fixed fraction: mean 0.088216; median 0.000000; range [0.000000, 0.996094]
- family median `||h12-h512||`: mean 4.41717; median 4.651; range [2.0394, 5.63282]
- family median `||F(h512)-h512||`: mean 3.79634; median 4.08101; range [1.19209e-07, 5.4753]
- family median final-state spread from centroid: mean 2.60192; median 2.64089; range [1.82065, 3.31517]

## Per-family values

| seed | M | class | fixed fraction | median d12→512 | median end residual | median final spread |
|---:|---:|---|---:|---:|---:|---:|
| 1561 | 110 | UNRESOLVED | 0.0000 | 4.89313 | 3.69499 | 2.66442 |
| 1579 | 110 | UNRESOLVED | 0.0000 | 5.06066 | 5.4753 | 2.94194 |
| 1597 | 100 | SHORT_CYCLE | 0.0000 | 5.63282 | 2.56366 | 2.53575 |
| 1616 | 90 | UNRESOLVED | 0.0615 | 5.09017 | 4.16689 | 2.26738 |
| 1634 | 120 | UNRESOLVED | 0.0000 | 3.55015 | 3.74349 | 2.53347 |
| 1652 | 130 | UNRESOLVED | 0.0000 | 2.0394 | 4.06792 | 2.73671 |
| 1671 | 100 | UNRESOLVED | 0.0000 | 4.16277 | 4.63504 | 3.31517 |
| 1689 | 110 | UNRESOLVED | 0.0000 | 3.79599 | 3.45559 | 2.03082 |
| 1708 | 100 | FIXED | 0.9961 | 4.46229 | 1.19209e-07 | 1.82065 |
| 1726 | 100 | UNRESOLVED | 0.0000 | 5.0167 | 5.25975 | 2.85591 |
| 1745 | 90 | UNRESOLVED | 0.0000 | 4.64506 | 4.39934 | 2.90344 |
| 1763 | 100 | UNRESOLVED | 0.0010 | 4.65693 | 4.0941 | 2.61735 |

## Claim boundary

This diagnostic addresses fixed-point/short-cycle behavior for native encoded states of the synthetic learned R8 recurrent map under a 512-step horizon and 1e-5 tolerance. It does not establish a global attractor over all latent states, formal basin topology, hysteresis, or generalization beyond this system.
