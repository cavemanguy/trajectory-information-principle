# R8 Current Synthesis — Updated through R8-AD6

**Status: September 6, 2026**

This file is the current concise synthesis for the late R8 mechanistic sequence. It supplements older `CURRENT_CLAIMS.md` / `EVIDENCE_LEDGER.md` entries that were written before M9–M12 and AD1–AD6 completed.

## Current narrow claim

Within the tested synthetic autonomous recurrent system, ordinary training organizes a learned recurrent flow whose structure is sensitive to optimization history and task demand. That organization can persist after present demand is matched, is substantially carried by the learned recurrent map, and gives rise to strong anisotropy around native trajectories.

The strongest current trajectory result is no longer only an accessibility result. The combined AD3–AD6 evidence supports this narrower mechanistic statement:

> **Recent native motion makes task information easier for simple readers to access, while norm-matched causal perturbations reveal that the learned flow preferentially corrects transverse deviations and preserves/amplifies along-flow deviations, which also cause larger downstream functional damage.**

This is still a statement about organization inside a deterministic Markov state-plus-map system. It does not establish an independent hidden phase variable or information beyond the complete state.

## M8–M12 mechanism chain

### R8-M8 — persistent history dependence

**Y3 — persistent history-dependent regime separation supported.** Opposite controlled demand histories left the same mature lineages in persistently different native dynamical organizations under matched present demand and matched training duration. The effect survived an additional 120 epochs of identical midpoint demand with about 90.7% retention.

Boundary: operational persistent path dependence is supported; formal bistability / strict hysteresis is not established.

### R8-M9 — component carriage

**C2 — recurrent-map-carried contribution supported.** Most of the persistent history-specific contribution followed the learned recurrent map under component swaps. Optimizer state was not necessary for the effect under the tested controls.

### R8-M10 — axis specificity

**S2 — strong axis specificity supported.** History targeted at one task axis reorganized that axis strongly while matched off-axis history reorganized its own axis without comparable effect on the first axis.

### R8-M11 — recurrent-map localization

**L1 — input-stage contribution supported.** The input-facing linear layer of the recurrent map (`F1`, 16→32) made a reproducible causal contribution to the persistent history effect.

Boundary: do not paraphrase this as “history is stored only in F1” or “F2 has no effect.”

### R8-M12 — failed simple susceptibility explanation

**U1 — no preregistered F1 susceptibility predictor.** The persistent specificity phenomenon replicated, but the preregistered simple pre-history F1 relative-gradient susceptibility predictor did not pass its frozen significance gate.

Boundary: a failed predictor is not a failed phenomenon.

## AD1–AD6 native-dynamics sequence

### R8-AD1 — long-run attractor diagnostic

**A3 — fixed-point attractor behavior not supported.** Across 12 mature lineages, 0/12 were classified FIXED under the frozen 512-step native-state criterion; 1/12 was a short cycle and 11/12 remained unresolved at that horizon.

### R8-AD2 — long-run regime classification

**D4 — heterogeneous learned regimes.** The 12 mature lineages separated into:

- FIXED: 0/12
- PERIODIC: 2/12
- QUASIPERIODIC_LIKE: 5/12
- SENSITIVE: 1/12
- REGULAR_NONPERIODIC: 1/12
- MIXED: 3/12

Boundary: `SENSITIVE` is a finite-time sensitivity label, not proof of chaos; `QUASIPERIODIC_LIKE` is descriptive rather than a theorem about an invariant set.

### R8-AD3 — one-step path-history accessibility

**H1 — one-step path-history accessibility supported.** A closed-form linear ridge reader given `[h_t, h_t-h_(t-1)]` outperformed a reader given `h_t` alone by **+5.72 pp**, 95% CI **[+5.39,+6.05] pp**, positive in **12/12 lineages**.

Against a dimension-matched shuffled-history control, the gain was **+5.88 pp**, CI **[+5.55,+6.21]**, also positive in **12/12**.

A larger quadratic static reader still beat the one-step representation by about 1.42 pp, so AD3 supports improved simple-readout accessibility rather than information unavailable from the instantaneous state under nonlinear readout.

### R8-AD4 — direction carries nearly all of the one-step accessibility gain

**D1 — direction-preserving decomposition.** The AD3 effect replicated on fresh probe data.

- full displacement gain over state: **+5.72 pp**
- normalized displacement direction gain: **+5.41 pp**
- retained fraction of full gain by direction: **~94.4%**
- scalar displacement magnitude gain: **+0.28 pp**
- retained fraction by magnitude: **~4.8%**

Adding the vector second difference `a_t=(h_t-h_(t-1))-(h_(t-1)-h_(t-2))` added another **+4.50 pp** of linear accessibility, CI **[+4.34,+4.66] pp**, positive in **12/12 lineages**. A scalar turning-angle cosine added only about **+0.08 pp**.

### R8-AD5 — causal transition splices reverse the accessibility intuition

**K0 — neither preregistered causal contrast supported.** AD5 intervened on native transitions while keeping the pre-splice state identical and matching Euclidean splice size.

Direction rotation versus magnitude-only change:

- direction-rotation terminal accuracy drop: **+1.92 pp**
- magnitude-only terminal accuracy drop: **+3.60 pp**
- generic norm-matched random drop: **+2.01 pp**
- `K_DIR = DROP_DIR - DROP_MAG`: **-1.68 pp**
- 95% CI: **[-2.21,-1.15] pp**
- positive in **1/12** lineages

Turn-orientation versus scalar-turn change:

- azimuth/turn-orientation drop: **+4.80 pp**
- polar/scalar-turn drop: **+7.85 pp**
- `K_CURV`: **-3.05 pp**
- 95% CI: **[-4.00,-2.00] pp**
- positive in **1/12** lineages

Both preregistered gates failed. This established a readability–causality dissociation: direction is highly informative to a simple reader, yet moving forward/backward along the native transition is more damaging than rotating the transition direction under matched splices.

### R8-AD6 — tangent versus transverse recovery

**P1 — tangent persistence with functional asymmetry supported.** AD6 directly tested the tangent/phase explanation generated after AD5 using norm-matched perturbations from the same native state.

After three recurrent updates:

- tangent total-error retention: **1.2308×**
- transverse total-error retention: **0.7256×**
- `D_REC = RET_TAN - RET_TRANS`: **+0.5052**
- 95% CI: **[+0.4093,+0.5969]**
- positive in **12/12 lineages**

The recovery curves were strongly anisotropic:

- step 1: tangent **1.069×**, transverse **0.646×**
- step 2: tangent **1.187×**, transverse **0.686×**
- step 3: tangent **1.231×**, transverse **0.726×**
- step 4: tangent **1.280×**, transverse **0.751×**

The signed along-flow offset also persisted: `PHASE_RET` at step 3 was **0.5355**, 95% CI **[0.4278,0.6407]**, and remained about **0.5244** at step 4.

Functional asymmetry also passed its frozen gate:

- tangent terminal accuracy drop: **+4.02 pp**
- transverse terminal accuracy drop: **+2.13 pp**
- `D_FUNC = DROP_TAN - DROP_TRANS`: **+1.90 pp**
- 95% CI: **[+1.29,+2.49] pp**
- positive in **11/12 lineages**
- cross-entropy contrast mean: **+0.0534**

The effect was present across all preregistered splice times and perturbation scales.

Boundary: AD6 supports strong tangent–transverse anisotropy around native learned trajectories. It does not establish an independent hidden phase variable, information beyond the complete Markov state-plus-map, essential chronology, or a universal trajectory code.

## Current mechanistic picture

The strongest current sequence is:

`training history`

→ `persistent task-axis-specific reorganization`

→ `substantial carriage by the learned recurrent map`

→ `reproducible contribution from the input-facing recurrent transformation`

→ `ongoing heterogeneous native dynamics rather than generic fixed-point settling`

→ `recent native motion improves simple task readout`

→ `one-step accessibility is overwhelmingly directional, not speed-based`

→ `a second vector difference exposes additional accessible structure`

→ `causal splices reveal greater vulnerability along the native flow than under matched direction rotation`

→ `direct tangent/transverse tests show transverse errors are preferentially corrected while tangent errors persist/amplify and cause larger functional damage`.

A compact description is:

> **The learned recurrent flow behaves locally like a computational path with transverse robustness and along-flow sensitivity.**

This is stronger than saying the trajectory is merely readable, but weaker than saying “the trajectory is the code.”

## What is still open

The project has **not** established that:

- trajectory history contains information beyond the complete Markov state-plus-map;
- temporal order is an independent information source;
- displacement direction, second difference, or phase is a universal neural code;
- an independent latent phase variable exists outside the state;
- tangent persistence is more than the dominant local Jacobian / finite-time sensitivity direction;
- heterogeneous AD2 regimes themselves are functionally necessary;
- the findings generalize to natural systems, transformers, LLMs, biological networks, or physical systems;
- a practical architecture advantage has been demonstrated.

## Next justified experiment

The next clean mechanistic question is:

> **Is the AD6 tangent–transverse asymmetry specifically aligned with native trajectory progress, or is it simply the dominant local Jacobian/singular-vector direction of the recurrent map?**

That comparison should test native tangent perturbations against equal-norm perturbations along leading and non-leading local Jacobian singular directions, then compare persistence, rejoining, and functional damage over the same finite horizon.

A positive native-tangent-specific result would support a trajectory-organized progress coordinate beyond generic local sensitivity. A null result would reduce the mechanism to ordinary local anisotropy of the learned map without invalidating AD3–AD6.

A separate future engineering branch is preserved for **query-conditioned active interrogation of a learned recurrent vector field**. It remains downstream of the native-system mechanistic tests.

## Authoritative result records

- `results/r8_ad1/aggregate/FINAL_RESULT.md`
- `results/r8_ad2/aggregate/FINAL_RESULT.md`
- `results/r8_ad3/aggregate/FINAL_RESULT.md`
- `results/r8_ad4/aggregate/FINAL_RESULT.md`
- `results/r8_ad5/aggregate/FINAL_RESULT.md`
- `results/r8_ad6/aggregate/FINAL_RESULT.md`

Interpretive notes:

- `docs/R8_AD5_RESULT.md`
- `docs/R8_AD6_RESULT.md`

## Permanent boundary

> **A failed explanation is not the same as a failed phenomenon, a positive accessibility result is not automatically causal, and a causal anisotropy result is not automatically an ontological claim about where information “really lives.”**
