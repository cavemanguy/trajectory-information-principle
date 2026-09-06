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

- family median FTLE: mean -0.010646; median -0.001547; range [-0.068664, +0.022718]
- family median top-8 spectral power: mean 0.916644; median 0.917746
- family median spectral entropy: mean 0.235403; median 0.250422

## Per-family values

| seed | M | class | dominant frac | median FTLE | median top8 | median entropy |
|---:|---:|---|---:|---:|---:|---:|
| 1782 | 100 | QUASIPERIODIC_LIKE | 1.0000 | -0.001132 | 0.930577 | 0.223406 |
| 1801 | 120 | QUASIPERIODIC_LIKE | 0.9922 | -0.001635 | 0.916975 | 0.242623 |
| 1819 | 100 | PERIODIC | 0.9648 | -0.068664 | 0.972349 | 0.117217 |
| 1838 | 110 | MIXED | 0.6953 | -0.003255 | 0.970362 | 0.175253 |
| 1856 | 100 | PERIODIC | 1.0000 | -0.053203 | 0.982808 | 0.132265 |
| 1875 | 110 | SENSITIVE | 0.9805 | +0.022718 | 0.758434 | 0.420508 |
| 1893 | 100 | QUASIPERIODIC_LIKE | 1.0000 | -0.001641 | 0.908261 | 0.309976 |
| 1912 | 100 | QUASIPERIODIC_LIKE | 1.0000 | -0.000922 | 0.918516 | 0.258221 |
| 1930 | 100 | MIXED | 0.6055 | +0.007541 | 0.857295 | 0.306868 |
| 1949 | 120 | REGULAR_NONPERIODIC | 0.9922 | -0.000561 | 0.890520 | 0.274603 |
| 1967 | 100 | QUASIPERIODIC_LIKE | 0.9883 | -0.001460 | 0.907763 | 0.278251 |
| 1986 | 100 | MIXED | 0.3594 | -0.025535 | 0.985863 | 0.085641 |

## Claim boundary

R8-AD2 classifies finite-horizon native encoded-state dynamics. SENSITIVE denotes a positive finite-time Lyapunov-like estimate under the frozen procedure, not formal proof of chaos; QUASIPERIODIC_LIKE denotes low-sensitivity spectrally concentrated nonperiodic motion, not a formal quasiperiodic invariant set; recurrence above period 512 is not excluded.
