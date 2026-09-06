# R8-AD5 — Causal Motion Intervention

**Status:** preregistered before implementation outcomes.

## Scientific question

R8-AD3/AD4 showed that correctly paired one-step history improves simple linear accessibility, that normalized displacement direction preserves most of that gain, and that a vector-valued second difference adds further accessibility. R8-AD5 asks the next causal question:

> If we intervene on a native transition while tightly matching the size of the state displacement, does changing the transition's direction have a different downstream functional consequence from changing only its magnitude? And, conditional on a previous native step, does changing the orientation of the turn have a different consequence from changing only the scalar turn angle?

This experiment does **not** attempt to create different futures from the exact same complete Markov state under the unchanged map. Instead it performs an explicit transition splice: the native state before the splice is held identical, the next state is replaced by a controlled counterfactual, and the frozen recurrent map then resumes normally.

## Frozen parent system

Use the exact R8 engine in `experiments/r8_m10/m7r_base.py` and reconstruct the same mature lineage set used in AD2–AD4:

`[1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986]`

No model parameters are changed by AD5. Each lineage is trained only to its original maturity criterion and then frozen.

AD5 uses **fresh intervention memories/permutations**, generated from a new deterministic `ad5_*` seed namespace. These intervention examples must not reuse AD3/AD4 probe examples.

Primary intervention set per lineage: 4,096 fresh examples. Smoke may use fewer.

## Native trajectory notation

For a fresh native trajectory:

`h0 -> h1 -> ... -> h12`

let

`d_t = h_(t+1) - h_t`, `m_t = ||d_t||`, `u_t = d_t / m_t`.

Primary splice times are frozen to:

`T = [2, 4, 6, 8, 10]`.

All interventions replace `h_(t+1)` and then apply the unchanged learned map `F` until `h12`, where the original frozen `headT` readers are evaluated.

The pre-splice state `h_t` is therefore exactly identical across native and intervention arms.

## Phase A — direction versus magnitude

For each native transition and relative endpoint perturbation scale

`rho in [0.05, 0.10, 0.20]`,

set the matched endpoint displacement budget

`r = rho * m_t`.

### A0. Native

`h_(t+1)^N = h_t + m_t u_t`.

### A1. Direction rotation

Construct a deterministic unit vector `q` orthogonal to `u_t` and rotate the native transition direction by

`theta = 2 asin(rho / 2)`.

Then

`u'_t = cos(theta) u_t + sin(theta) q`

and

`h_(t+1)^DIR = h_t + m_t u'_t`.

This preserves transition magnitude exactly while satisfying

`||h_(t+1)^DIR - h_(t+1)^N|| = r`.

### A2. Magnitude-only controls

Keep direction fixed and change only transition length:

`h_(t+1)^(MAG+) = h_t + (1 + rho) m_t u_t`

`h_(t+1)^(MAG-) = h_t + (1 - rho) m_t u_t`.

Each satisfies the same endpoint displacement budget `r`. Their downstream effects are averaged so that the magnitude control does not privilege expansion or contraction.

### A3. Generic norm-matched random control

Add a deterministic random unit-vector perturbation of norm `r` to the native `h_(t+1)`. This arm is secondary context, not a primary gate.

## Phase A primary functional metric

For each arm, compute terminal mean relation accuracy using the frozen native `headT` readers after recurrence resumes to step 12.

Define accuracy drop relative to the native trajectory:

`DROP_X = ACC_NATIVE - ACC_X`.

For each lineage, average over examples, relations, frozen times, and frozen scales.

Primary direction contrast:

`K_DIR = DROP_DIR - 0.5*(DROP_MAG+ + DROP_MAG-)`.

Positive `K_DIR` means a matched state perturbation that rotates the native transition is more functionally disruptive than changing only its length.

Also compute the analogous terminal cross-entropy contrast `K_DIR_CE`; positive means the direction arm raises loss more.

### Frozen direction gate — G_DIRECTION_CAUSAL

Pass iff all conditions hold across the 12 lineages:

1. mean `K_DIR >= +0.005` absolute accuracy (0.5 percentage points);
2. deterministic lineage-bootstrap 95% CI lower bound for `K_DIR` is `> 0`;
3. `K_DIR > 0` in at least 10/12 lineages;
4. mean `K_DIR_CE > 0` and `K_DIR_CE > 0` in at least 10/12 lineages.

## Phase B — curvature orientation versus scalar turn

This phase uses the same frozen splice times and fresh examples. Define the previous native displacement

`d_prev = h_t - h_(t-1)`

and the native next displacement

`d_next = h_(t+1) - h_t`.

Write `d_next` relative to the previous unit direction `v = d_prev/||d_prev||` as

`d_next = a v + p`, with `p` orthogonal to `v`.

Let `e = p/||p||` when the perpendicular component is non-negligible. A deterministic unit vector `q` is constructed orthogonal to both `v` and `e`.

Frozen azimuth rotation angles:

`alpha in [15 degrees, 30 degrees]`.

### B1. Turn-orientation / azimuth intervention

Rotate only the perpendicular orientation:

`e' = cos(alpha)e + sin(alpha)q`

`d_next^AZ = a v + ||p|| e'`.

This preserves, up to floating-point tolerance:

- the pre-splice state `h_t`;
- `||d_next||` (step speed);
- the parallel component `a`;
- `||p||`;
- the scalar cosine/angle between `d_prev` and `d_next`.

It changes the orientation of the turn in latent space.

### B2. Polar / scalar-turn control

Construct a rotation of `d_next` **within its original `(v,e)` plane** that:

- preserves `||d_next||`;
- has the same endpoint perturbation norm `||d_next^POL - d_next||` as the paired azimuth intervention;
- changes the scalar turn angle while preserving the original turn plane.

The sign of the polar rotation is chosen deterministically to remain inside `[0, pi]`.

Thus AZ and POL are pairwise matched in step norm and Euclidean endpoint displacement, but differ in whether the intervention changes turn orientation or scalar turn angle.

Samples with numerically degenerate step norms are left native in both paired arms and counted; the valid/nondegenerate fraction is reported.

## Phase B primary metric

Define terminal accuracy drops `DROP_AZ` and `DROP_POL` relative to native, averaged over examples, relations, times, and the two frozen azimuth angles.

`K_CURV = DROP_AZ - DROP_POL`.

Also compute cross-entropy contrast `K_CURV_CE`.

### Frozen curvature gate — G_CURVATURE_ORIENTATION

Pass iff:

1. mean `K_CURV >= +0.005` absolute accuracy;
2. deterministic lineage-bootstrap 95% CI lower bound is `> 0`;
3. `K_CURV > 0` in at least 10/12 lineages;
4. mean `K_CURV_CE > 0` and positive in at least 10/12 lineages.

## Frozen classification

Provided all 12 lineages are valid and all geometry checks pass:

- **K3 — direction and turn-orientation causal sensitivity supported:** both gates pass.
- **K2 — direction-specific causal sensitivity supported:** direction passes, curvature-orientation fails.
- **K1 — turn-orientation causal sensitivity supported:** curvature-orientation passes, direction fails.
- **K0 — neither preregistered causal contrast supported:** both fail.
- **KX — invalid/protocol failure:** lineage reconstruction, numerical finiteness, paired-norm geometry, or required artifact completeness fails.

No failed gate may be rescued by a secondary metric.

## Frozen geometry validity checks

For every tested nondegenerate example (within numerical tolerance):

- DIR preserves native step magnitude;
- DIR, MAG+, and MAG- have matched endpoint displacement `r`;
- MAG arms preserve native direction;
- AZ preserves next-step magnitude and native scalar turn cosine;
- AZ and POL have matched endpoint displacement;
- POL preserves next-step magnitude.

Tolerance: maximum absolute geometric mismatch `<= 1e-5` after excluding explicitly counted degenerate zero-norm cases.

Any systematic violation above tolerance makes the family invalid.

## Secondary analyses

Report, without promotion to the primary classification:

- each frozen time separately;
- each direction perturbation scale separately;
- each azimuth angle separately;
- per-relation effects;
- generic random norm-matched Phase-A control;
- effects grouped by the already-frozen AD2 regime labels;
- immediate one-step changes in the frozen `headT` logits as descriptive context;
- native versus intervention final-state Euclidean displacement;
- degenerate-step fractions.

## Claim boundary

A positive AD5 result would support a narrow causal statement about **anisotropic functional sensitivity aligned to native trajectory geometry under explicit transition interventions**. It would not establish:

- information beyond the complete Markov state-plus-map;
- that motion is independent of state representation;
- that a trajectory carries a separate metaphysical information channel;
- essential chronology in general;
- universal trajectory coding;
- practical superiority over transformers or other architectures;
- novelty relative to all prior dynamical-systems/RNN literature.

Even K3 would mean that, in this tested learned recurrent system, matched transition splices that alter direction/turn orientation have systematically different downstream consequences from matched splices that alter only transition magnitude/scalar turn. It would not by itself prove that the trajectory is the only or irreducible representation.
