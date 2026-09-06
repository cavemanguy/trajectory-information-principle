# R8-AD3 Final Result — One-Step Path-History Accessibility

**Primary classification:** H1 — one-step path-history accessibility supported

## Frozen primary gates

- G_STATE: **True** — mean joint−state +0.057197, 95% CI [+0.053912, +0.060489], positive 12/12
- G_SHUFFLE: **True** — mean joint−shuffled +0.058760, 95% CI [+0.055487, +0.062062], positive 12/12

## Secondary controls

- joint−quadratic-static mean -0.014221, 95% CI [-0.018610, -0.009804]
- t=12 joint−state mean +0.013811, 95% CI [+0.011268, +0.016431]

## Per-lineage primary internal-window values

| seed | M | AD2 regime | state | velocity | joint | shuffled | quadratic | Δstate | Δshuffle | Δquadratic |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1782 | 100 | QUASIPERIODIC_LIKE | 0.1694 | 0.1816 | 0.2192 | 0.1680 | 0.2195 | +0.0499 | +0.0513 | -0.0003 |
| 1801 | 120 | QUASIPERIODIC_LIKE | 0.1653 | 0.1824 | 0.2295 | 0.1637 | 0.2449 | +0.0641 | +0.0658 | -0.0154 |
| 1819 | 100 | PERIODIC | 0.1890 | 0.2003 | 0.2460 | 0.1871 | 0.2599 | +0.0571 | +0.0590 | -0.0138 |
| 1838 | 110 | MIXED | 0.1676 | 0.1673 | 0.2206 | 0.1660 | 0.2339 | +0.0530 | +0.0546 | -0.0132 |
| 1856 | 100 | PERIODIC | 0.1842 | 0.2013 | 0.2401 | 0.1829 | 0.2514 | +0.0559 | +0.0572 | -0.0113 |
| 1875 | 110 | SENSITIVE | 0.1589 | 0.1822 | 0.2236 | 0.1575 | 0.2463 | +0.0647 | +0.0661 | -0.0227 |
| 1893 | 100 | QUASIPERIODIC_LIKE | 0.1664 | 0.1887 | 0.2267 | 0.1648 | 0.2449 | +0.0604 | +0.0619 | -0.0182 |
| 1912 | 100 | QUASIPERIODIC_LIKE | 0.1620 | 0.1643 | 0.2207 | 0.1604 | 0.2447 | +0.0587 | +0.0602 | -0.0240 |
| 1930 | 100 | MIXED | 0.1612 | 0.1804 | 0.2278 | 0.1596 | 0.2407 | +0.0667 | +0.0682 | -0.0129 |
| 1949 | 120 | REGULAR_NONPERIODIC | 0.1652 | 0.1708 | 0.2176 | 0.1644 | 0.2453 | +0.0524 | +0.0532 | -0.0276 |
| 1967 | 100 | QUASIPERIODIC_LIKE | 0.1652 | 0.1876 | 0.2221 | 0.1632 | 0.2269 | +0.0569 | +0.0589 | -0.0048 |
| 1986 | 100 | MIXED | 0.1925 | 0.1966 | 0.2390 | 0.1903 | 0.2455 | +0.0466 | +0.0487 | -0.0064 |

## Claim boundary

R8-AD3 tests whether correctly paired one-step native trajectory history improves simple linear readout accessibility relative to the instantaneous state and a dimension-matched shuffled-history control. It does not establish information beyond the complete Markov state-plus-map, causal necessity, essential chronology, or a universal trajectory code.
