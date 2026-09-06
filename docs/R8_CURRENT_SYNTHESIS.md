# R8 Current Synthesis — Updated through R8-AD4

**Status: September 6, 2026**

This file is the current concise synthesis for the late R8 mechanistic sequence. It supplements older `CURRENT_CLAIMS.md` / `EVIDENCE_LEDGER.md` entries that were written before M9–M12 and AD1–AD4 completed.

## Current narrow claim

Within the tested synthetic autonomous recurrent system, ordinary training organizes a learned recurrent flow whose structure is sensitive to optimization history and task demand. That organization can persist after present demand is matched, is substantially carried by the learned recurrent map, and makes task distinctions differentially accessible along native trajectories.

The strongest current trajectory-accessibility result is not that the trajectory creates information beyond the Markov state. It is:

> **Correctly paired recent trajectory history makes task information substantially more accessible to a simple linear reader than the instantaneous latent state alone, and most of that one-step gain is carried by displacement direction rather than displacement magnitude. A vector-valued second difference adds further accessibility beyond the full one-step displacement.**

## M8–M12 mechanism chain

### R8-M8 — persistent history dependence

**Y3 — persistent history-dependent regime separation supported.** Opposite controlled demand histories left the same mature lineages in persistently different native dynamical organizations under matched present demand and matched training duration. The effect survived an additional 120 epochs of identical midpoint demand with about 90.7% retention.

Boundary: operational persistent path dependence is supported; formal bistability / strict hysteresis is not established.

### R8-M9 — component carriage

**C2 — recurrent-map-carried contribution supported.** Most of the persistent history-specific contribution followed the learned recurrent map under component swaps. Optimizer state was not necessary for the effect under the tested controls.

Boundary: this does not imply the encoder contributes nothing or that all history is stored in one parameter block.

### R8-M10 — axis specificity

**S2 — strong axis specificity supported.** History targeted at one task axis reorganized that axis strongly while matched off-axis history reorganized its own axis without comparable effect on the first axis.

Boundary: this is task-axis specificity in the tested synthetic system, not a universal memory law or formal hysteresis theorem.

### R8-M11 — recurrent-map localization

**L1 — input-stage contribution supported.** The input-facing linear layer of the recurrent map (`F1`, 16→32) made a reproducible causal contribution to the persistent history effect. The downstream linear layer did not independently pass its frozen gate, and the interaction gate also failed.

Boundary: do not paraphrase this as “history is stored only in F1” or “F2 has no effect.”

### R8-M12 — failed simple susceptibility explanation

**U1 — no preregistered F1 susceptibility predictor.** The persistent specificity phenomenon replicated, but the preregistered simple pre-history F1 relative-gradient susceptibility predictor did not pass its frozen significance gate.

Boundary: a failed predictor is not a failed phenomenon.

## AD1–AD4 native-dynamics / accessibility sequence

### R8-AD1 — long-run attractor diagnostic

**A3 — fixed-point attractor behavior not supported.** Across 12 mature lineages, 0/12 were classified FIXED under the frozen 512-step native-state criterion; 1/12 was a short cycle and 11/12 remained unresolved at that horizon. The ordinary `h12` state was generally far from the long-run state.

Boundary: this does not prove that mathematical fixed points do not exist, that all dynamics are periodic, or that unresolved dynamics are chaotic.

### R8-AD2 — long-run regime classification

**D4 — heterogeneous learned regimes.** The 12 mature lineages separated into:

- FIXED: 0/12
- PERIODIC: 2/12
- QUASIPERIODIC_LIKE: 5/12
- SENSITIVE: 1/12
- REGULAR_NONPERIODIC: 1/12
- MIXED: 3/12

Boundary: `SENSITIVE` is a finite-time sensitivity label, not formal proof of chaos; `QUASIPERIODIC_LIKE` is descriptive rather than a theorem about an invariant set.

### R8-AD3 — one-step path-history accessibility

**H1 — one-step path-history accessibility supported.** Using fresh probe splits and frozen mature lineages, a closed-form linear ridge reader given

`[h_t, h_t - h_{t-1}]`

outperformed a reader given `h_t` alone across the internal `t=1..11` window by:

- mean gain: **+0.057197** absolute accuracy (+5.72 pp)
- 95% CI: **[+0.053912, +0.060489]**
- positive: **12/12 lineages**

Against the dimension-matched shuffled-history control, the gain was:

- mean: **+0.058760**
- 95% CI: **[+0.055487, +0.062062]**
- positive: **12/12**

A generous 152-feature quadratic static reader beat the 32-feature one-step representation by about 1.42 pp on average. Therefore AD3 supports improved **linear accessibility from correctly paired path history**, not information unavailable from the instantaneous state under nonlinear readout.

The effect was strongest early but remained positive later, including at `t=12`.

### R8-AD4 — motion-component decomposition

**D1 — direction-preserving decomposition.** The AD3 one-step effect replicated on fresh AD4 probe data:

- full displacement gain over state: **+0.057168**, CI **[+0.053932, +0.060424]**, positive 12/12
- full displacement gain over shuffled displacement: **+0.058621**, CI **[+0.055524, +0.061697]**, positive 12/12

The decomposition was strongly directional:

- normalized displacement direction gain: **+0.054076**, CI **[+0.050327, +0.058123]**, positive 12/12
- mean fraction of the full gain retained by direction: **0.943787** (~94.4%), CI **[0.922595, 0.964752]**
- scalar displacement magnitude gain: **+0.002780**, CI **[+0.002189, +0.003365]**, positive 12/12
- mean fraction retained by magnitude: **0.048052** (~4.8%), CI **[0.039207, 0.056724]**

Thus the frozen direction-preservation gate passed and the magnitude-preservation gate failed.

AD4 also preregistered a higher-order test. Adding the vector second difference

`a_t = (h_t - h_{t-1}) - (h_{t-1} - h_{t-2})`

to the full one-step representation improved linear accessibility by:

- **+0.044996** (+4.50 pp)
- 95% CI **[+0.043355, +0.046567]**
- positive **12/12**

so `G_CURVATURE_ADDS` passed.

By contrast, adding only the scalar cosine turning measure produced a much smaller **+0.000754** mean gain. This supports a useful higher-order **vector-valued second-difference** feature; it does not justify saying that a single turning angle is the code.

Coordinate-level results are preserved as secondary because latent axes are arbitrary/basis-dependent. The preregistered best-single-coordinate and top-four-coordinate probes were positive across lineages, but they are not promoted over the rotation-sensitive claim boundary.

## Current mechanistic picture

The current evidence is most naturally summarized as:

`training history`

→ `persistent task-axis-specific reorganization`

→ `substantial carriage by the learned recurrent map`

→ `reproducible contribution from the input-facing recurrent transformation`

→ `ongoing heterogeneous native dynamics rather than generic fixed-point settling`

→ `task information becomes substantially easier for a linear reader when recent native motion is supplied`

→ `most of the one-step accessibility benefit is directional, not speed-based`

→ `a second vector difference exposes still more linearly accessible task structure`.

A compact mathematical description of the newest result is:

`h_t` < `[h_t, d_t / ||d_t||]` ≈ `[h_t, d_t]` < `[h_t, d_t, d_t-d_{t-1}]`

for simple linear task readout in the tested internal trajectory window, while

`[h_t, log ||d_t||]`

adds very little.

## What is still open

The project has **not** established that:

- trajectory history contains information beyond the complete Markov state-plus-map;
- temporal order is causally necessary;
- displacement direction or curvature is a universal neural code;
- the AD3/AD4 accessibility gains are causally necessary for the trained network's own task solution;
- changing phase/direction while controlling state will change function in the predicted way;
- these findings generalize to natural systems, transformers, LLMs, biological networks, or physical dynamical systems;
- the heterogeneous AD2 regimes themselves are functionally necessary;
- a practical architecture advantage has been demonstrated.

## Next justified experiment

The most direct next scientific step is a **causal motion intervention** that changes native direction / higher-order trajectory structure while controlling instantaneous-state displacement as tightly as possible, then measures whether task-relevant functional consequences change.

That experiment should be designed so a positive result cannot be reduced to simply moving the state farther across the frozen reader boundary.

A separate future engineering branch is preserved for **query-conditioned active interrogation of a learned recurrent vector field**. That should remain downstream of the current native-system causal tests rather than being used to force a preferred geometry into the present system.

## Authoritative result records

- `results/r8_ad1/aggregate/FINAL_RESULT.md`
- `results/r8_ad2/aggregate/FINAL_RESULT.md`
- `results/r8_ad3/aggregate/FINAL_RESULT.md`
- `results/r8_ad4/aggregate/FINAL_RESULT.md`

The experiment preregistrations, runners, classifiers, and workflows are preserved under `experiments/r8_ad1/` through `experiments/r8_ad4/` and `.github/workflows/`.

## Permanent boundary

> **A failed explanation is not the same as a failed phenomenon, and a positive accessibility result is not automatically a causal or ontological claim about where information “really lives.”**
