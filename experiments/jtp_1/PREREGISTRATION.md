# JTP-1 — Joint Trajectory–Perturbation Mapping

**Status:** FROZEN BEFORE OUTCOME INSPECTION  
**Frozen date:** 2026-09-04  
**Parent program:** Trajectory Information / Observer / ALI  
**Source dynamics:** exact frozen Observer-R2 primary cores, seeds 7, 19, 43

## 1. Primary scientific question

Does the local perturbational response geometry of the frozen dynamical map systematically change along its native evolving trajectory in a way that cannot be reduced to response magnitude, contraction, gross state radius, or a destroyed state–time relationship?

JTP-1 tests the narrow claim

> local response geometry contains reproducible trajectory-dependent structure.

It does **not** assume that such structure exists.

## 2. Frozen source systems

No model will be retrained for JTP-1. The experiment reuses the exact Observer-R2 frozen core checkpoints produced by the successful primary workflow runs:

| Seed | Observer-R2 run | Core artifact | Artifact id | Artifact digest |
|---:|---:|---|---:|---|
| 7 | 33560910331 | `observer-r2-seed7-core` | 9821298916 | `sha256:676daf2ed7b0c37e0f8c54536b9d57d8c0e67da4d97e7a398d100685b70bf3a6` |
| 19 | 33562734327 | `observer-r2-seed19-core` | 9821996148 | `sha256:97bbe0d09f7620c39aa8bf65cdc4137b7badf6cf7e3bbfce1d5ba34e99bdf209` |
| 43 | 33566188625 | `observer-r2-seed43-core` | 9823283251 | `sha256:9ffabbc421acdeb9776137f07d05081c1dff3930d2fec5d89aeeeb8115c6c6c3` |

Each downloaded `core_best.pt` is verified against the `checkpoint_sha256` recorded in its accompanying `core_summary.json` before any JTP-1 computation is permitted.

The frozen core has latent dimension 16 and iterative map

\[
F(h)=\tanh\!\left(W_2\,\mathrm{GELU}(W_1 h+b_1)+b_2\right).
\]

The native trajectory is

\[
h_0,h_1,\ldots,h_{12}, \qquad h_{t+1}=F(h_t).
\]

## 3. Fixed analysis bank

JTP-1 uses **1,024 fresh analysis memories**. They are not the Observer-R2 train, validation, or final-test arrays. The memory values and the memory-encoding order are generated once from a fixed NumPy generator seed and are identical across the three frozen cores.

- analysis-bank seed: `20260904`
- number of memories: `1024`
- values per memory: `8`
- value alphabet: `0..15`
- fixed-memory SHA-256: `aea7617fbf9817cecdd29e28d55f4ecdd88678ca56a3312629c42d8f60a3c2a7`
- encoding-permutation SHA-256: `8cae4c20d0efc961714bcb37ad286903d9f1abef970d1add4de5328756d0801d`

The implementation asserts these hashes before execution.

## 4. Trajectory-time grid

Every available trajectory state is analyzed:

\[
t\in\{0,1,2,\ldots,12\}.
\]

No time window may be selected after observing JTP-1 results.

## 5. Perturbation-scale grid

For each seed, define

\[
s=\operatorname{median}_{m}\|h_0^{(m)}\|_2
\]

over the fixed 1,024-memory analysis bank.

The six primary perturbation scales are

\[
\epsilon/s\in\{0.001,0.003,0.01,0.03,0.1,0.3\}.
\]

Thus the primary grid contains `13 × 6 = 78` cells per seed.

Every primary cell is additionally recomputed at \(\epsilon/2\) and \(2\epsilon\) for numerical-convergence checks. These neighboring scales are diagnostics and may not replace the primary grid.

## 6. Direction bank and empirical response operator

The latent dimension is 16, so JTP-1 uses a complete deterministic 16-direction orthonormal bank: the rows of the normalized 16×16 Sylvester-Hadamard matrix.

Direction-bank SHA-256:

`d64f9a1707174557fec37dc9535cfd7f806c86b48314c9d092a4e6685eee6769`

For state \(h\), direction \(v_k\), and perturbation scale \(\epsilon\), measure the symmetric response

\[
r_k(h,\epsilon)=\frac{F(h+\epsilon v_k)-F(h-\epsilon v_k)}{2\epsilon}.
\]

Because the direction bank is a complete orthonormal basis, these responses reconstruct the finite-scale empirical local response operator \(J(h,\epsilon)\) without fitting a learned model.

## 7. Required controls

### 7.1 Native

Use the actual ordered states \(h_0,\ldots,h_{12}\).

### 7.2 Temporal shuffle

For each analysis memory, use a deterministic fixed-point-free permutation of the 13 trajectory indices. This preserves that memory's exact state multiset while destroying the native state–time assignment.

- temporal-control seed: `20260906`
- time-derangement SHA-256: `143c2843cd97da4a874964717464c603c595839e8d31d634afbc18a9cec2a468`

### 7.3 State-radius-matched control

For every native `(memory,time)` state, choose the nearest latent-norm state subject to both:

1. different memory, and
2. different trajectory time.

This preserves gross radial location as closely as possible while breaking the native trajectory identity and time assignment. Matching is deterministic, and the resulting match-index hashes are recorded separately for every seed.

### 7.4 Direction-association permutation

Destroy the association between perturbation directions and responses while preserving response magnitudes by applying a deterministic cyclic shift of five response columns in the Hadamard direction basis. The shift has no fixed direction.

## 8. Primary separation measures

For native operator \(A\) and a control operator \(B\):

\[
D(A,B)=\|A-B\|_F
\]

and

\[
R(A,B)=\frac{\|A-B\|_F}{\tfrac12(\|A\|_F+\|B\|_F)+\eta},
\qquad \eta=10^{-12}.
\]

Operator cosine is also recorded:

\[
C(A,B)=\frac{\langle A,B\rangle_F}{\|A\|_F\|B\|_F+\eta}.
\]

The primary geometric advantage for a grid cell is the paired state-level difference

\[
\Delta_R(t,\epsilon)
=R_{\rm temporal}(t,\epsilon)-R_{\rm state\ matched}(t,\epsilon).
\]

A positive \(\Delta_R\) means destroying the native time assignment separates local response geometry more than merely replacing the state by a gross-radius-matched non-native state.

The corresponding cosine diagnostic is

\[
\Delta_C(t,\epsilon)
=C_{\rm state\ matched}-C_{\rm temporal}.
\]

## 9. Uncertainty

For each seed and grid cell, bootstrap the 1,024 paired per-memory values of

\[
R_{\rm temporal}-R_{\rm state\ matched}
\]

with `2,000` deterministic bootstrap resamples. Record the percentile 95% interval.

The bootstrap generator is derived deterministically from seed, time, and epsilon index in the frozen implementation.

## 10. Numerical validity

For every state and primary grid cell, compare the empirical operator at \(\epsilon\) with those at \(\epsilon/2\) and \(2\epsilon\).

A seed/cell passes the preregistered numerical-stability screen only when both seed-level means satisfy

\[
\frac{\|J_{\epsilon}-J_{\epsilon/2}\|_F}{\|J_{\epsilon}\|_F+\eta}\le 0.10
\]

and

\[
\frac{\|J_{\epsilon}-J_{2\epsilon}\|_F}{\|J_{\epsilon}\|_F+\eta}\le 0.10.
\]

A consensus grid cell is considered numerically valid only if **all three seeds** pass.

## 11. Secondary mechanistic diagnostics

For every primary cell, record:

- native and control response/operator magnitudes,
- pairwise and population-mean Frobenius separation,
- normalized separation,
- operator cosine,
- full singular-value spectrum of the population-mean native operator,
- effective rank,
- dominant singular anisotropy `s1 / mean(s)`,
- symmetric-component fraction,
- antisymmetric-component fraction,
- top-4 left-singular-subspace alignment between native and temporal-control mean operators,
- native state norm.

These diagnostics explain a primary result; they do not replace the preregistered decision rule.

## 12. Frozen cell-level geometric criterion

A `(t, epsilon)` cell is a **consensus geometric cell** only if all three seeds satisfy all of the following:

1. numerical-stability screen passes,
2. \(\Delta_R\ge 0.05\),
3. the 95% bootstrap interval for \(\Delta_R\) has lower bound `> 0`, and
4. \(\Delta_C>0\).

No one-seed or two-seed cell qualifies.

## 13. Frozen outcome classification

### Outcome A — no preregistered trajectory-dependent response structure

Outcome C does not qualify and the remaining stable-grid behavior does not meet the preregistered sensitivity/contraction classification below.

If no grid cell is numerically valid across all three seeds, classify A with the explicit qualifier that the chosen finite-difference grid failed the validity screen.

### Outcome B — sensitivity/contraction explanation

If Outcome C fails, classify B when at least one stable consensus cell exists and either:

- the absolute rank correlation across stable cells between seed-mean absolute temporal separation \(D\) and native response magnitude has magnitude `>= 0.80`, **or**
- the largest seed-mean \(\Delta_R\) over stable cells is `< 0.05`.

This classification means the experiment does not establish distinctive trajectory-dependent response geometry beyond generic sensitivity/state effects.

### Outcome C — geometric trajectory structure

Classify C when at least **4 consensus geometric cells** exist and they span at least:

- 2 distinct trajectory times, and
- 2 distinct perturbation scales.

### Outcome D — bounded trajectory–perturbation regime

Classify D only if Outcome C qualifies and, additionally:

1. the largest 4-neighbor connected component of consensus geometric cells has size `>= 4`,
2. the maximum seed-mean \(\Delta_R\) consensus cell is interior in both time and epsilon,
3. that peak is at least `1.5×` the larger epsilon-boundary value at the same time, and
4. that peak is at least `1.5×` the larger time-boundary value at the same epsilon.

This is the preregistered operational definition of the proposed bounded “Goldilocks” regime.

## 14. Claim boundary

Even Outcome D would **not** establish that:

- a downstream network reads this structure,
- the structure improves task performance,
- chronological history itself is encoded,
- the system explicitly represents its past,
- the effect generalizes to transformers, biological systems, or unrelated dynamical models.

The strongest allowed JTP-1 claim is narrower:

> Along these frozen Observer-R2 dynamics, local perturbational response geometry shows reproducible scale- and trajectory-position-dependent structure that survives the preregistered controls and numerical checks.

## 15. Execution integrity

The experiment implementation writes, for each seed:

- source checkpoint hash,
- source core summary,
- fixed-bank hashes,
- trajectory tensor hash,
- state-match index hashes,
- package/runtime versions,
- all 78 primary-cell metrics,
- all numerical-convergence diagnostics.

The aggregate classification is produced mechanically by the frozen decision rule in `run_jtp1.py`. The result is not manually chosen after inspection.
