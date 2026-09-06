# R8-AD1 Preregistration — Learned Long-Run Attractor Diagnostic

## Question

Does the mature learned R8 recurrent map exhibit stable fixed-point attractor behavior on native encoded states when autonomous recurrence is extended far beyond the ordinary 12-step window?

This is a diagnostic of the current learned R8 dynamics. It does not alter, reinterpret, or gate any prior R8 result.

## Engine

Use the exact `experiments/r8_m10/m7r_base.py` engine from the branch under test. Train each fresh family to the same frozen maturity criterion used by R8-M10. No history fork is applied: the diagnostic is performed on the mature baseline model.

## Fresh families

`[1561, 1579, 1597, 1616, 1634, 1652, 1671, 1689, 1708, 1726, 1745, 1763]`

## Native-state sample

For each family, encode 2,048 fresh validation memories/permutations into native `h0` states. These states are not used for training.

## Long rollout

For every native `h0`, repeatedly apply the learned autonomous map `F` for 512 steps. Record states at steps 0, 12, 64, 128, 256, 384, 448, and 512, plus the final 64 one-step residuals.

One-step residual:

`r_t = ||h_{t+1} - h_t||_2`.

## Fixed-point criterion

A sampled state is classified as fixed-point converged when both:

1. `max(r_t)` over the final 32 transitions is <= `1e-5`; and
2. `||F(h_512) - h_512||_2 <= 1e-5`.

A family is `FIXED` when at least 95% of its 2,048 native states satisfy the per-state fixed-point criterion.

## Short-cycle diagnostic

For non-fixed states only, test periods `p = 2..16` using the final 32 steps. A state is called period-p recurrent when:

- median `||h_t - h_{t-p}||_2 <= 1e-5` across the comparable tail; and
- its median one-step residual over the same tail is > `1e-5`.

A family is `SHORT_CYCLE` if it is not FIXED and at least 95% of native states satisfy the same period `p` for some `p in 2..16`.

Otherwise the family is `UNRESOLVED`.

## Distance-to-asymptote diagnostic

For each state record:

`d12_512 = ||h_12 - h_512||_2`.

Report family median and cross-family distribution. This is descriptive, not a gate.

## Cross-family classification

- `A1 — stable fixed-point attractor behavior supported`: at least 10/12 valid families are FIXED and 0 families are SHORT_CYCLE.
- `A2 — mixed long-run regimes`: neither A1 nor A3, with at least one FIXED or SHORT_CYCLE family.
- `A3 — fixed-point attractor behavior not supported`: 3 or fewer valid families are FIXED.
- `A4 — invalid/incomplete`: fewer than 12 valid mature families.

These labels apply only to native encoded states of this synthetic learned R8 system under the stated tolerance and 512-step horizon. They do not establish global attractors over all of R^16.
