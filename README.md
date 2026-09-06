# Trajectory Information Project

Independent research on learned recurrent dynamics, trajectory structure, latent geometry, history-dependent computation, and active latent interrogation.

**Preservation rule:** preserve successful, negative, ambiguous, superseded, exploratory, and prototype work. A failed explanation is not the same as a failed phenomenon.

The current scientific focus is the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, causally tested, and eventually used practically?**

Observers are measurement tools. Perturbations are optional causal/control tools. Neither is assumed to be the source of trajectory information.

## Current source of truth

Start here for the newest R8 synthesis:

- [`docs/R8_CURRENT_SYNTHESIS.md`](docs/R8_CURRENT_SYNTHESIS.md) — **latest synthesis through R8-AD4**
- [`docs/CURRENT_CLAIMS.md`](docs/CURRENT_CLAIMS.md) — older claim ledger and boundaries
- [`docs/EVIDENCE_LEDGER.md`](docs/EVIDENCE_LEDGER.md) — preserved evidence map, negatives, and protocol-limited branches
- [`RESEARCH_PRIORITIES.md`](RESEARCH_PRIORITIES.md) — scientific rules and experiment priorities
- [`docs/research_history.md`](docs/research_history.md) — preserved historical development

The newer synthesis supplements older claim/ledger files that were written before M9–M12 and AD1–AD4 completed.

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

The analysis tracks how natural task distinctions evolve through that trajectory. For inputs differing in one relation, the experiments follow

```text
Delta_t = h_t(x) - h_t(x')
```

and measure how distinctions survive, contract, expand, reorganize, and become accessible to simple readers.

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

## AD1–AD4: what the native motion is doing

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

**H1 — one-step path-history accessibility supported.** A closed-form linear ridge reader given

```text
[h_t, h_t - h_(t-1)]
```

outperformed a reader given `h_t` alone by **+5.72 percentage points on average**, with a 95% bootstrap CI of **[+5.39,+6.05] pp**, positive in **12/12 lineages**.

Against a dimension-matched shuffled-history control, the gain was **+5.88 pp**, CI **[+5.55,+6.21]**, also positive in **12/12**.

A much larger quadratic static reader still beat the 32-feature one-step representation by about 1.42 pp, so the supported claim is **improved simple-readout accessibility**, not information unavailable from the instantaneous state under nonlinear readout.

Authoritative result: [`results/r8_ad3/aggregate/FINAL_RESULT.md`](results/r8_ad3/aggregate/FINAL_RESULT.md)

### AD4 — direction carries nearly all of the one-step gain

**D1 — direction-preserving decomposition.** The AD3 effect replicated on fresh AD4 probe data.

Full signed displacement:

- gain over state: **+5.72 pp**
- CI: **[+5.39,+6.04] pp**
- positive: **12/12**

Normalized displacement direction:

- gain over state: **+5.41 pp**
- CI: **[+5.03,+5.81] pp**
- positive: **12/12**
- retained fraction of full gain: **~94.4%**

Scalar displacement magnitude:

- gain over state: **+0.28 pp**
- retained fraction: **~4.8%**
- frozen magnitude-preservation gate: **failed**

The higher-order preregistered test also passed. Adding the vector second difference

```text
a_t = (h_t - h_(t-1)) - (h_(t-1) - h_(t-2))
```

to the full one-step representation added another **+4.50 pp** of linear accessibility, CI **[+4.34,+4.66] pp**, positive in **12/12 lineages**.

By contrast, adding only a scalar cosine turning-angle measure added about **+0.08 pp**. The supported higher-order result is therefore about the **vector-valued second difference**, not a single turn-angle scalar.

Authoritative result: [`results/r8_ad4/aggregate/FINAL_RESULT.md`](results/r8_ad4/aggregate/FINAL_RESULT.md)

## Current compact picture

The strongest current mechanistic chain is:

```text
training history
    -> persistent task-axis-specific reorganization
    -> substantial carriage by the learned recurrent map
    -> reproducible recurrent-component contribution
    -> ongoing heterogeneous native dynamics
    -> recent native motion improves simple task readout
    -> one-step accessibility is overwhelmingly directional, not speed-based
    -> a second vector difference exposes additional accessible structure
```

For simple linear task readout in the tested internal trajectory window, the newest result can be summarized as:

```text
h_t
  < [h_t, normalized displacement direction]
  ~= [h_t, full displacement]
  < [h_t, full displacement, second vector difference]
```

while scalar speed adds little.

## What this does not establish

The project has **not** established that:

- trajectory history contains information beyond the complete Markov state-plus-map;
- chronology itself is an independent information source;
- displacement direction or second difference is a universal neural code;
- the AD3/AD4 accessibility features are causally necessary for the trained network's own task solution;
- heterogeneous AD2 regimes are themselves functionally necessary;
- the system exhibits formal bistability or strict dynamical-systems hysteresis;
- unresolved long-run dynamics are chaotic;
- the findings generalize to language models, transformers, biological systems, physical systems, or naturalistic tasks;
- the project has demonstrated a practical architecture advantage over conventional systems.

## Current next question

The next justified experiment is causal rather than another decoder comparison:

> **If native direction or higher-order trajectory structure is altered while instantaneous-state displacement and frozen-reader boundary distance are tightly controlled, does the system's functional consequence change?**

A positive result would move the project from **trajectory accessibility** toward **causal dynamical function**.

A separate future engineering experiment is preserved for **query-conditioned active interrogation of a learned recurrent vector field**. That experiment should remain downstream of the native-system causal tests so it does not force a preferred geometry into the discovery system.

## Other preserved research programs

### Observer / native-trajectory program

R2–R11 studied what information is accessible from evolving trajectory geometry, how recurrence changes generic versus trained-reader accessibility, how selective survival emerges, and where Euclidean preservation fails to predict functional usefulness.

Key boundaries:

- geometry-history accessibility can exceed endpoint snapshots;
- exact chronology was not established as essential;
- early transients can dominate some directional-history signals;
- recurrence can improve trained-reader compatibility while generic accessibility worsens;
- survival magnitude alone does not determine reader usefulness.

See [`docs/observer_program_r2_r11.md`](docs/observer_program_r2_r11.md).

### Active Latent Interrogation (ALI)

ALI studies whether controlled query-dependent perturbations of a frozen latent state can expose query-relevant information through the response of a frozen nonlinear transformation.

The reproducible ALI-N8-R1 result supports query-specific direction-dependent local responses under diagnostic decoders, but adaptive ALI did not beat direct memory readout and exhibited direction-only leakage. It does not establish that ALI replaces attention or provides a general memory architecture.

The future active-interrogation architecture idea is preserved separately and is not treated as an established result.

### Causal-control / perturbation program

R4B/R4C/R4D/R4E and JTP tested controllability, learned self-steering, low-dimensional control, state-conditioned intervention consequences, and instantaneous local-operator signatures.

Important negative boundaries are preserved: learned self-nudging did not produce a strong practical controller, R4E failed its primary Phase-I nonlinear gate, and JTP-1 did not find the preregistered seed-general instantaneous Jacobian-like trajectory-time signature.

### Reader robustness / affine geometry

AG3–AG5 studied why geometrically small representation errors can be functionally catastrophic and why larger errors can sometimes remain functionally acceptable. Local reader robustness matters; Euclidean distance alone is insufficient.

### Historical attractor-era work

The original attractor/trajectory prototypes motivated the project but contained claims that exceeded the available evidence. They remain preserved as project history rather than current proof.

The motivating observation remains simple: trajectories approaching the same stable endpoint could contain reproducible structure from which the original input could sometimes be recovered before convergence. The current R8 program tests related ideas under learned dynamics with stronger controls.

See [`archive/`](archive/README.md) and [`docs/research_history.md`](docs/research_history.md).

## Reproducibility and scientific record

The repository intentionally keeps:

- preregistrations before outcome inspection;
- deterministic/fresh probe or seed definitions where appropriate;
- negative and protocol-limited outcomes;
- post-run audits labeled separately from primary results;
- GitHub Actions workflows and authoritative result branches;
- claim boundaries separate from historical ideas.

The project-wide rule remains:

> **A failed explanation is not the same as a failed phenomenon, and a promising secondary pattern is not promoted into a primary result after the fact.**
