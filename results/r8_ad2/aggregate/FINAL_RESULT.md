# R8-AD2 Final Result — Native Dynamical Regime Classification

**Primary classification:** D4 — heterogeneous learned regimes

## Family classifications

- FIXED: **0/12**
- PERIODIC: **2/12**
- SENSITIVE: **1/12**
- QUASIPERIODIC_LIKE: **5/12**
- REGULAR_NONPERIODIC: **1/12**
- MIXED: **3/12**

## Cross-family diagnostics

- family median FTLE: mean -0.010666; median -0.001550; range [-0.068664, +0.022718]
- family median top-8 spectral power: mean 0.916651; median 0.917747
- family median spectral entropy: mean 0.235393; median 0.250412

## Per-family values

| seed | M | class | dominant frac | median FTLE | median top8 | median entropy |
|---:|---:|---|---:|---:|---:|---:|
| 1782 | 100 | QUASIPERIODIC_LIKE | 1.0000 | -0.001132 | 0.930577 | 0.223406 |
| 1801 | 120 | QUASIPERIODIC_LIKE | 0.9844 | -0.001694 | 0.916976 | 0.242622 |
| 1819 | 100 | PERIODIC | 0.9648 | -0.068664 | 0.972349 | 0.117217 |
| 1838 | 110 | MIXED | 0.6953 | -0.003255 | 0.970362 | 0.175253 |
| 1856 | 100 | PERIODIC | 1.0000 | -0.053532 | 0.982808 | 0.132265 |
| 1875 | 110 | SENSITIVE | 0.9805 | +0.022718 | 0.758434 | 0.420508 |
| 1893 | 100 | QUASIPERIODIC_LIKE | 1.0000 | -0.001641 | 0.908261 | 0.309976 |
| 1912 | 100 | QUASIPERIODIC_LIKE | 0.9961 | -0.000798 | 0.918517 | 0.258202 |
| 1930 | 100 | MIXED | 0.6055 | +0.007541 | 0.857295 | 0.306868 |
| 1949 | 120 | REGULAR_NONPERIODIC | 0.9922 | -0.000498 | 0.890588 | 0.274495 |
| 1967 | 100 | QUASIPERIODIC_LIKE | 0.9883 | -0.001460 | 0.907763 | 0.278251 |
| 1986 | 100 | MIXED | 0.3047 | -0.025572 | 0.985879 | 0.085648 |

## Claim boundary

R8-AD2 classifies finite-horizon native encoded-state dynamics. SENSITIVE denotes a positive finite-time Lyapunov-like estimate under the frozen procedure, not formal proof of chaos; QUASIPERIODIC_LIKE denotes low-sensitivity spectrally concentrated nonperiodic motion, not a formal quasiperiodic invariant set; recurrence above period 512 is not excluded.
