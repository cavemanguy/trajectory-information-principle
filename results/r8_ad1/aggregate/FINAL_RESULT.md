# R8-AD1 Final Result — Learned Long-Run Attractor Diagnostic

**Primary classification:** A3 — fixed-point attractor behavior not supported

## Family classifications

- FIXED: **0/12**
- SHORT_CYCLE: **1/12**
- UNRESOLVED: **11/12**

## Cross-family diagnostics

- fixed fraction: mean 0.077555; median 0.000000; range [0.000000, 0.867676]
- family median `||h12-h512||`: mean 4.42007; median 4.65164; range [2.0394, 5.63282]
- family median `||F(h512)-h512||`: mean 3.79976; median 4.08108; range [1.07454e-07, 5.4753]
- family median final-state spread from centroid: mean 2.49656; median 2.6346; range [0.546866, 3.31517]

## Per-family values

| seed | M | class | fixed fraction | median d12→512 | median end residual | median final spread |
|---:|---:|---|---:|---:|---:|---:|
| 1561 | 110 | UNRESOLVED | 0.0000 | 4.83602 | 3.72773 | 2.65318 |
| 1579 | 110 | UNRESOLVED | 0.0000 | 5.06066 | 5.4753 | 2.94194 |
| 1597 | 100 | SHORT_CYCLE | 0.0000 | 5.63282 | 2.56366 | 2.53575 |
| 1616 | 90 | UNRESOLVED | 0.0625 | 5.15272 | 4.16599 | 2.29977 |
| 1634 | 120 | UNRESOLVED | 0.0000 | 3.53255 | 3.74218 | 2.52466 |
| 1652 | 130 | UNRESOLVED | 0.0000 | 2.0394 | 4.06792 | 2.73671 |
| 1671 | 100 | UNRESOLVED | 0.0000 | 4.16256 | 4.63512 | 3.31517 |
| 1689 | 110 | UNRESOLVED | 0.0000 | 3.80043 | 3.46584 | 2.02926 |
| 1708 | 90 | UNRESOLVED | 0.8677 | 4.50366 | 1.07454e-07 | 0.546866 |
| 1726 | 100 | UNRESOLVED | 0.0000 | 5.0167 | 5.25975 | 2.85591 |
| 1745 | 90 | UNRESOLVED | 0.0000 | 4.64506 | 4.39934 | 2.90344 |
| 1763 | 100 | UNRESOLVED | 0.0005 | 4.65822 | 4.09424 | 2.61601 |

## Claim boundary

This diagnostic addresses fixed-point/short-cycle behavior for native encoded states of the synthetic learned R8 recurrent map under a 512-step horizon and 1e-5 tolerance. It does not establish a global attractor over all latent states, formal basin topology, hysteresis, or generalization beyond this system.
