# Trajectory Information Project

Independent research on learned recurrent dynamics, trajectory structure, latent geometry, history-dependent computation, and active latent interrogation.

**Preservation rule:** preserve successful, negative, ambiguous, superseded, exploratory, and prototype work. A failed explanation is not the same as a failed phenomenon.

The current scientific focus is the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, causally tested, and eventually used practically?**

Observers are measurement tools. Perturbations are optional causal/control tools. Neither is assumed to be the source of trajectory information.

## Current source of truth

Start here:

- [`docs/R8_CURRENT_SYNTHESIS.md`](docs/R8_CURRENT_SYNTHESIS.md) — **current synthesis through R8-AD6**
- [`docs/R8_AD6_RESULT.md`](docs/R8_AD6_RESULT.md) — tangent/transverse recovery interpretation
- [`docs/R8_AD5_RESULT.md`](docs/R8_AD5_RESULT.md) — preserved negative causal-splice result and phase/tangent hypothesis origin
- [`docs/CURRENT_CLAIMS.md`](docs/CURRENT_CLAIMS.md) — older claim ledger and boundaries
- [`docs/EVIDENCE_LEDGER.md`](docs/EVIDENCE_LEDGER.md) — preserved evidence map, negatives, and protocol-limited branches
- [`RESEARCH_PRIORITIES.md`](RESEARCH_PRIORITIES.md) — scientific rules and experiment priorities
- [`docs/research_history.md`](docs/research_history.md) — preserved historical development

The newer R8 synthesis/result files supplement older claim/ledger files written before M9–M12 and AD1–AD6 completed.

## Current R8 system

The active R8 program uses ordinary neural-network components arranged as an autonomous latent dynamical system.

The task has **8 statistically symmetric categorical relations**, each taking one of **16 values**. A GRU encoder compresses the complete input into a **16-dimensional latent state** `h0`. After `h0`, the model receives **no new external input**. The same learned recurrent map is applied for 12 autonomous transitions:

```text
input -> GRU encoder -> h0 -> F -> F -> ... -> F -> h12 -> reader
```

with

```text
h_(t+1) = F(h_t)
```

The analysis tracks how natural task distinctions evolve through that trajectory and how controlled perturbations propagate through the frozen recurrent flow.

The architecture itself is not claimed to be radically novel. The research question is what ordinary training organizes inside it.

## Late R8 mechanistic sequence

### M8 — persistent history dependence

**Y3 — persistent history-dependent regime separation supported.** Opposite controlled demand histories left the same mature lineage in persistently different native organizations under matched present demand and matched training duration. About 90.7% of the midpoint separation remained after another 120 epochs of identical midpoint demand.

### M9 — recurrent-map carriage

**C2 — recurrent-map-carried contribution supported.** Most of the persistent history-specific contribution followed the learned recurrent map under component swaps. Optimizer state was not necessary under the tested controls.

### M10 — axis specificity

**S2 — strong axis specificity supported.** History targeted at one task axis reorganized that axis strongly, while matched off-axis history reorganized its own axis without comparable effect on the first.

### M11 — recurrent-map localization

**L1 — input-stage contribution supported.** The input-facing recurrent linear transformation made a reproducible causal contribution. This does not mean history is stored only there or that downstream components are irrelevant.

### M12 — failed simple susceptibility explanation

**U1 — no preregistered F1 susceptibility predictor.** The persistent specificity phenomenon replicated, but the preregistered simple pre-history F1 susceptibility predictor failed its frozen gate.

A failed predictor is not a failed phenomenon.

## AD1–AD6: native motion, accessibility, and causal anisotropy

### AD1 — no generic fixed-point settling

**A3 — fixed-point attractor behavior not supported.** Under the frozen 512-step native-state criterion, 0/12 mature families were classified FIXED; one was a short cycle and 11 remained unresolved at that horizon.

### AD2 — heterogeneous learned long-run regimes

**D4 — heterogeneous learned regimes.** Across 12 mature lineages:

- FIXED: 0/12
- PERIODIC: 2/12
- QUASIPERIODIC_LIKE: 5/12
- SENSITIVE: 1/12
- REGULAR_NONPERIODIC: 1/12
- MIXED: 3/12

These are finite-horizon descriptive classifications. `SENSITIVE` is not formal proof of chaos, and `QUASIPERIODIC_LIKE` is not a theorem about an invariant set.

### AD3 — one-step path history improves linear accessibility

**H1 — one-step path-history accessibility supported.** A closed-form linear ridge reader given `[h_t, h_t-h_(t-1)]` outperformed a reader given `h_t` alone by **+5.72 pp**, with 95% CI **[+5.39,+6.05] pp**, positive in **12/12 lineages**.

Against a dimension-matched shuffled-history control, the gain was **+5.88 pp**, CI **[+5.55,+6.21]**, also positive in **12/12**.

A larger quadratic static reader still beat the one-step representation by about 1.42 pp, so the supported claim is improved simple-readout accessibility rather than information unavailable from the instantaneous state under nonlinear readout.

### AD4 — direction carries nearly all of the one-step accessibility gain

**D1 — direction-preserving decomposition.** The AD3 effect replicated. Normalized displacement direction retained about **94.4%** of the one-step gain, while scalar speed retained about **4.8%**. Adding a vector second difference added another **+4.50 pp** of linear accessibility.

### AD5 — causal transition splices reverse the accessibility intuition

**K0 — neither preregistered causal contrast supported.** Direction rotations caused a smaller terminal accuracy drop (**+1.92 pp**) than matched magnitude/tangent changes (**+3.60 pp**). The preregistered `K_DIR` contrast was **-1.68 pp**, CI **[-2.21,-1.15]**, positive in only **1/12** lineages.

Likewise, turn-orientation interventions were less damaging than matched scalar-turn interventions. AD5 therefore established a **readability–causality dissociation** rather than the hoped-for direction-causality result.

Authoritative result: [`results/r8_ad5/aggregate/FINAL_RESULT.md`](results/r8_ad5/aggregate/FINAL_RESULT.md)

### AD6 — tangent persistence with transverse recovery

**P1 — tangent persistence with functional asymmetry supported.** AD6 directly tested the post-AD5 tangent/phase explanation using equal-norm perturbations from the same native state.

After three recurrent updates:

- tangent total-error retention: **1.231×**
- transverse total-error retention: **0.726×**
- `D_REC`: **+0.505**
- 95% CI: **[+0.409,+0.597]**
- positive: **12/12 lineages**

The stepwise recovery curves were:

```text
step 0: tangent 1.000x | transverse 1.000x
step 1: tangent 1.069x | transverse 0.646x
step 2: tangent 1.187x | transverse 0.686x
step 3: tangent 1.231x | transverse 0.726x
step 4: tangent 1.280x | transverse 0.751x
```

The signed along-flow offset remained about **0.536×** after three steps and **0.524×** after four.

Functionally:

- tangent perturbation terminal accuracy drop: **+4.02 pp**
- transverse perturbation terminal accuracy drop: **+2.13 pp**
- `D_FUNC`: **+1.90 pp**
- 95% CI: **[+1.29,+2.49] pp**
- positive: **11/12 lineages**

Thus the learned flow shows reproducible **tangent–transverse anisotropy**: transverse errors are preferentially corrected, while along-flow errors persist/amplify and have larger downstream task consequences.

Authoritative result: [`results/r8_ad6/aggregate/FINAL_RESULT.md`](results/r8_ad6/aggregate/FINAL_RESULT.md)

Interpretive note: [`docs/R8_AD6_RESULT.md`](docs/R8_AD6_RESULT.md)

## Current compact picture

The strongest current sequence is:

```text
training history
    -> persistent task-axis-specific reorganization
    -> substantial carriage by the learned recurrent map
    -> reproducible recurrent-component contribution
    -> ongoing heterogeneous native dynamics
    -> recent native motion improves simple task readout
    -> one-step accessibility is overwhelmingly directional, not speed-based
    -> a second vector difference exposes additional accessible structure
    -> causal splices reveal greater vulnerability along the native flow than under matched direction rotation
    -> transverse perturbations are preferentially corrected while tangent perturbations persist/amplify and cause larger functional damage
```

A concise present interpretation is:

> **The learned recurrent flow behaves locally like a computational path with transverse robustness and along-flow sensitivity.**

That is stronger than saying the trajectory is merely readable. It is still weaker than saying “the trajectory is the code.”

## What this does not establish

The project has **not** established that:

- trajectory history contains information beyond the complete Markov state-plus-map;
- chronology itself is an independent information source;
- displacement direction, second difference, or phase is a universal neural code;
- an independent hidden phase variable exists outside the state;
- AD6 tangent persistence is more than the dominant local Jacobian / finite-time sensitivity direction;
- heterogeneous AD2 regimes are themselves functionally necessary;
- the system exhibits formal bistability or strict dynamical-systems hysteresis;
- unresolved long-run dynamics are chaotic;
- the findings generalize to language models, transformers, biological systems, physical systems, or naturalistic tasks;
- the project has demonstrated a practical architecture advantage over conventional systems.

## Current next question

The next clean test is:

> **Is the AD6 tangent–transverse asymmetry specifically aligned with native trajectory progress, or is it simply the dominant local Jacobian/singular-vector direction of the recurrent map?**

That comparison can tell us whether the observed robustness/sensitivity is trajectory-organized in a stronger sense or reducible to ordinary local anisotropy of the learned map.

A separate future engineering experiment is preserved for **query-conditioned active interrogation of a learned recurrent vector field**. It remains downstream of the native-system mechanistic tests so it does not force a preferred geometry into the discovery system.

## Other preserved research programs

Observer/native-trajectory, ALI, causal-control/perturbation, reader-robustness, and historical attractor-era work remain preserved in the repository with their negative and protocol-limited outcomes.

The motivating historical observation remains simple: trajectories approaching the same stable endpoint could contain reproducible structure from which the original input could sometimes be recovered before convergence. The current R8 program tests related ideas under learned dynamics with much stronger controls.

## Reproducibility and scientific record

The repository intentionally keeps preregistrations before outcome inspection, deterministic/fresh probe or intervention definitions where appropriate, negative and protocol-limited outcomes, post-run audits labeled separately, workflows and authoritative result records, and explicit claim boundaries.

The project-wide rule remains:

> **A failed explanation is not the same as a failed phenomenon, and a promising secondary pattern is not promoted into a primary result after the fact.**
