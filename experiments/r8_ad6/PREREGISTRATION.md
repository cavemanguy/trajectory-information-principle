# R8-AD6 Preregistration — Tangent vs Transverse Error Dynamics

**Status:** frozen before implementation/outcome inspection

## Motivation

R8-AD4 established a diagnostic accessibility result: recent path history improves linear task readout, normalized displacement direction preserves most of the one-step gain, and a vector second difference adds further accessibility.

R8-AD5 then produced the frozen negative result **K0 — neither preregistered causal contrast supported**. Under matched transition-splice size, magnitude/tangent-like changes were more damaging than direction rotations, while generic random perturbations behaved similarly to direction rotations. AD5 explicitly labeled one possible explanation as exploratory: **tangent/phase sensitivity with transverse robustness**.

R8-AD6 tests that explanation directly. It does not alter, rescue, or reinterpret the frozen AD5 K0 result.

## Scientific question

For a frozen mature recurrent system, do norm-matched errors applied along the local native flow persist more strongly than errors applied transverse to that flow, and does that recovery asymmetry accompany greater downstream functional damage?

The experiment is diagnostic/causal and does not retrain the mature model after intervention.

## Parent system and lineages

Use the exact mature-lineage reconstruction used by R8-AD3 through R8-AD5.

Expected lineage seeds:

`[1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986]`

All model parameters are frozen during AD6 analysis.

## Fresh intervention data

For each lineage, generate a fresh deterministic AD6 intervention bank that does not reuse AD3/AD4/AD5 probe-example namespaces.

Full run: `N = 4096` fresh examples per lineage.

Smoke data are outcome-free and mechanically separate.

## Native trajectory

For each example compute the ordinary trajectory `h_0,...,h_12` under frozen recurrent map `F`.

Frozen intervention times:

`t in {2, 4, 6, 8}`

At native state `h_t`, define

`d_t = F(h_t) - h_t = h_(t+1) - h_t`

and unit tangent

`u_t = d_t / ||d_t||`.

Examples with `||d_t|| < 1e-7` are excluded for that condition and valid fraction is recorded.

## Perturbation scales

Frozen relative scales:

`rho in {0.05, 0.10, 0.20}`

with perturbation norm

`r = rho * ||d_t||`.

All arms for a given example/time/rho have the same Euclidean displacement `r` from the same native state `h_t`.

## Intervention arms

### Tangent

`z_T+ = h_t + r u_t`

`z_T- = h_t - r u_t`.

Primary tangent quantities average the + and - arms.

### Transverse

Generate four deterministic unit vectors `q_j` orthogonal to `u_t` by projecting independent Gaussian vectors, with deterministic fallback for degeneracy.

`z_Pj = h_t + r q_j`, `j=0..3`, with `q_j · u_t = 0`.

Primary transverse quantities average across the four transverse directions.

Geometry validity on valid examples requires:

- every perturbation norm matches `r` within `1e-5` absolute error;
- every transverse direction satisfies `|q_j · u_t| <= 1e-5`;
- all reported quantities are finite.

## Recovery rollout

From each intervened state, apply the same frozen recurrent map for four steps.

At horizon `k=0..4`, compare intervened `z_(t+k)` with native same-time `h_(t+k)`:

`e_k = z_(t+k) - h_(t+k)`.

At each native same-time state with a next step, define

`u_(t+k) = normalize(F(h_(t+k)) - h_(t+k))`.

Decompose:

`e_parallel = (e_k · u_(t+k)) u_(t+k)`

`e_perp = e_k - e_parallel`.

Record total retention `||e_k||/r`, parallel retention `|e_k·u|/r`, transverse retention `||e_perp||/r`, parallel fraction, and signed normalized progress offset `(e_k·u)/r` for T+ and T- separately.

The **primary recovery horizon is k=3**. Other horizons are secondary.

## Functional consequence

After the four-step recovery diagnostic, continue each intervened state through the same frozen map until step 12 and evaluate the existing frozen terminal heads.

For each condition compute terminal accuracy and cross-entropy changes relative to unperturbed native trajectory.

Tangent functional damage averages T+ and T-. Transverse functional damage averages P0..P3.

## Primary family-level summaries

Average over frozen times and scales after computing condition-level metrics.

### Recovery contrast

At `k=3`:

`RET_TAN = mean total retention of T+, T-`

`RET_TRANS = mean total retention over P0..P3`

`D_REC = RET_TAN - RET_TRANS`.

Positive D_REC means equal-size tangent errors remain farther from the same-time native trajectory after three updates.

### Functional contrast

`DROP_TAN = native terminal accuracy - tangent terminal accuracy`

`DROP_TRANS = native terminal accuracy - transverse terminal accuracy`

`D_FUNC = DROP_TAN - DROP_TRANS`.

Positive D_FUNC means local-tangent perturbations are more damaging.

Cross-entropy analogues are secondary and cannot rescue an accuracy failure.

### Phase-alignment diagnostic

At `k=3`:

`PHASE_RET = 0.5 * [median((e_T+·u)/r) + median(-(e_T-·u)/r)]`.

This is mechanistic context and cannot promote a failed recovery gate.

## Frozen cross-lineage gates

All 12 expected lineages must be valid.

Bootstrap family-level means with 20,000 deterministic resamples, seed `86061`.

### G_RECOVERY

Pass iff:

1. mean `D_REC >= 0.05` retention-ratio units;
2. bootstrap 95% CI lower bound for D_REC is `> 0`;
3. D_REC > 0 in at least 10/12 lineages.

### G_FUNCTION

Pass iff:

1. mean `D_FUNC >= 0.005` absolute accuracy (0.5 pp);
2. bootstrap 95% CI lower bound for D_FUNC is `> 0`;
3. D_FUNC > 0 in at least 10/12 lineages.

## Frozen classification

- **P1 — tangent persistence with functional asymmetry supported:** both gates pass.
- **P2 — tangent persistence only:** recovery passes, function fails.
- **P3 — functional tangent sensitivity only:** recovery fails, function passes.
- **P0 — tangent/phase account not supported by primary gates:** neither passes.
- **V0 — invalid:** seed set, maturity, geometry, completeness, or finite-value validity fails.

## Secondary analyses

Report without changing classification:

- recovery curves k=0..4;
- parallel/transverse component curves;
- PHASE_RET at k=1..4;
- by-rho and by-time results;
- R8-AD2 regime-stratified results;
- terminal cross-entropy contrasts;
- whether transverse total retention falls below 1.0;
- whether tangent error becomes predominantly parallel;
- descriptive correlations between retained tangent error and terminal damage.

## Interpretation boundaries

P1 would support, in this tested synthetic system, a specific mechanism for the AD5 reversal: local transverse errors are preferentially corrected while tangent errors persist and are functionally more consequential under matched perturbation size.

Even P1 would not establish information beyond complete Markov state-plus-map, an independent hidden phase variable, essential chronology, a universal trajectory code, formal phase reduction for all lineages, generalization to other architectures/domains, or practical superiority.

P2 would support recovery geometry without terminal functional linkage. P3 would support functional tangent sensitivity without preferential transverse recovery. P0 would reject this preregistered mechanism while preserving AD3/AD4 and AD5 K0 unchanged.
