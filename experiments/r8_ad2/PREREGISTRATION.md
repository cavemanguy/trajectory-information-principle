# R8-AD2 Preregistration — Native Dynamical Regime Classification

## Question

What kind of long-run autonomous motion is produced by the mature learned R8 recurrent map when no dynamical form is imposed during training?

R8-AD1 ruled out dominant fixed-point behavior under a 512-step horizon. R8-AD2 is a fresh-family descriptive classification study. It does not alter training, reward a preferred geometry, or gate any prior R8 result.

## Engine

Use the exact `experiments/r8_m10/m7r_base.py` engine inherited by this branch. Each family is trained only to the frozen R8 maturity criterion. No attractor, cycle, spectral, Lyapunov, or trajectory-shape loss is introduced.

## Fresh families

`[1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986]`

These seeds are frozen before outcome inspection and are disjoint from R8-AD1.

## Native-state sample

For each family, encode 256 fresh memories/permutations into native `h0` states. These states are not used for training.

## Long autonomous rollout

For each native state, repeatedly apply the learned map `F` for 4,096 steps. Keep the final 1,024 states for recurrence and spectral diagnostics.

## Fixed-point diagnostic

A state is `FIXED` when its maximum one-step residual over the final 64 transitions is <= `1e-5` and `||F(h_4096)-h_4096||_2 <= 1e-5`.

## Periodic diagnostic

For every non-fixed state, search candidate periods `p=2..512` by the distance from the final state to `h_{4096-p}`. Take the smallest-distance candidate period and validate it over the final 64 comparable pairs.

A state is `PERIODIC` when the median validated recurrence distance is <= `1e-4` while its median one-step residual is > `1e-5`.

This detects longer cycles than AD1 but does not claim periods above 512 are absent.

## Finite-time sensitivity diagnostic

Estimate a maximal finite-time Lyapunov-like exponent using a deterministic Benettin-style finite-difference procedure after a 512-step burn-in:

- perturbation norm `epsilon = 1e-4`;
- deterministic unit perturbation direction per state;
- 512 renormalized evolution steps;
- per-step log stretch `log(||delta_{t+1}||/epsilon)`;
- state estimate is the mean log stretch over the 512 steps.

This is a finite-time sensitivity estimate, not a proof of mathematical chaos.

A non-fixed, non-periodic state is `SENSITIVE` when its estimate is > `+0.01`.

## Spectral-complexity diagnostic

For each state, center the final 1,024-step trajectory, compute an FFT along time for all 16 latent coordinates, sum power across coordinates, remove the DC bin, and normalize the spectrum.

Record:

- normalized spectral entropy;
- fraction of non-DC spectral power contained in the top 8 frequency bins.

Because summed coordinate power is invariant to orthogonal rotations of latent coordinates, this is used only as a descriptive regularity measure.

A non-fixed, non-periodic, non-sensitive state is `QUASIPERIODIC_LIKE` when:

- `abs(FTLE) <= 0.01`; and
- top-8 spectral power fraction >= `0.90`.

This label means low-sensitivity, spectrally concentrated, nonperiodic motion under the stated finite horizon. It is not a formal proof of quasiperiodicity.

All remaining valid states are `REGULAR_NONPERIODIC`.

## Family classification

For each family compute fractions of native states in:

`FIXED`, `PERIODIC`, `SENSITIVE`, `QUASIPERIODIC_LIKE`, `REGULAR_NONPERIODIC`.

If one state class occupies >=80% of native states, the family receives that class. Otherwise the family is `MIXED`.

## Frozen cross-family classification

- `D1 — sensitive nonconvergent dynamics supported`: >=8/12 valid families are `SENSITIVE`.
- `D2 — regular nonperiodic dynamics supported`: >=8/12 valid families are either `QUASIPERIODIC_LIKE` or `REGULAR_NONPERIODIC`, and <=2/12 are `SENSITIVE`.
- `D3 — periodic/fixed dynamics supported`: >=8/12 valid families are either `FIXED` or `PERIODIC`.
- `D4 — heterogeneous learned regimes`: no D1-D3 gate passes and at least three distinct family classes are represented among valid families.
- `D5 — unresolved/mixed`: no D1-D4 gate passes.
- `D6 — invalid/incomplete`: fewer than 12 valid mature families.

## Claim boundary

R8-AD2 classifies native encoded-state dynamics of this synthetic learned R8 system under finite horizons and frozen thresholds. `SENSITIVE` is not synonymous with proven chaos; `QUASIPERIODIC_LIKE` is not a formal quasiperiodic invariant set; failure to detect recurrence through period 512 does not rule out longer cycles; and no result establishes global behavior over all of latent space.