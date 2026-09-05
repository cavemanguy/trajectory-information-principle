# Trajectory Information Project

Independent research on learned recurrent dynamics, trajectory structure, latent geometry, and history-dependent computation.

**Preservation rule:** preserve successful, negative, ambiguous, superseded, exploratory, and prototype work. A failed explanation is not the same as a failed phenomenon.

The current scientific focus is the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, and eventually used practically?**

Observers are measurement tools. Perturbations are optional causal/control tools. Neither is assumed to be the source of trajectory information.

For the current source of truth, see:

- [`docs/CURRENT_CLAIMS.md`](docs/CURRENT_CLAIMS.md) — current supported claims and boundaries
- [`docs/EVIDENCE_LEDGER.md`](docs/EVIDENCE_LEDGER.md) — evidence map, negative results, and protocol-limited branches
- [`RESEARCH_PRIORITIES.md`](RESEARCH_PRIORITIES.md) — next experiments and permanent scientific rules
- [`docs/research_history.md`](docs/research_history.md) — preserved historical development

## Current R8 system

The active R8 program uses ordinary neural-network components arranged as a clean autonomous latent dynamical system.

The task has **8 statistically symmetric categorical relations**, each taking one of **16 values**. A GRU encoder compresses the complete input into a **16-dimensional latent state** `h0`. After `h0`, the model receives **no new external input**. The same learned recurrent map is then applied for 12 autonomous transitions:

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

and measure how strongly each relation survives, contracts, or expands under the learned recurrent dynamics.

The architecture itself is not claimed to be radically novel. The research question is what ordinary training organizes inside it.

## R8 discovery sequence

### R8-M1 — encoder–recurrence coadaptation

**M3 — supported.** Full relation-selective native survival required plasticity in both the encoder and recurrent map.

### R8-M2 — simple source tests

**S0 — neither simple source tracks winner.** Relation-specific initialization bundles and finite-sample data-column identity did not reliably determine which relation became dynamically favored.

### R8-M3 — optimization-path sensitivity

**T0 — preregistered early predictors not supported.** Changing only minibatch order altered commitment timing and changed final winner identity in 5/12 paired families.

### R8-M4 — functional contribution

**F3 — selective-specialization contribution supported.** Equalizing established relation-selective survival reduced terminal performance beyond a matched mean-survival control, with most loss concentrated in the dynamically favored relation.

### R8-M5 / M5R / M6 — capacity and workspace

- **M5: C0** — simple scarce-state-capacity account not supported.
- **M5R: R2** — wider state produced a fresh state-dimension-specific terminal benefit without requiring specialization to move in one predetermined direction.
- **M6: W0** — isolated recurrent-workspace attribution not supported because h0 equivalence failed after continued training.

### R8-M7R — demand sensitivity without free takeover

**D0 — reversible demand tracking not supported.** All 12 mature families were valid. Terminal demand toward the baseline loser moved native organization strongly toward that relation and improved its terminal performance, but complete specialist reassignment was rare.

Detailed result: [`docs/R8_M7R_RESULT.md`](docs/R8_M7R_RESULT.md)

### R8-M7I — inverted mirror

**V0 — baseline lineage reproduction failure.** The primary mirror result remains invalid because four paired lineages missed an overly tight preregistered numerical baseline-Q tolerance, despite matching maturity epoch and A/B identities. Post-primary diagnostics are preserved separately and are not used to rescue the frozen outcome.

- [`docs/R8_M7I_RESULT.md`](docs/R8_M7I_RESULT.md)
- [`docs/R8_M7I_POSTRUN_AUDIT.md`](docs/R8_M7I_POSTRUN_AUDIT.md)

### R8-M8 — persistent history-dependent regime separation

**Y3 — persistent history-dependent regime separation supported.**

Twelve fresh families were trained to maturity, then forked into opposite continuous demand histories. At the matched midpoint, both branches had:

- the same architecture;
- the same current demand `lambda=0.50`;
- the same cumulative post-maturity training duration;
- different controlled demand histories.

The preregistered native separation was:

```text
H_mid = Q_B-history(0.50) - Q_A-history(0.50)
```

with

- mean `H_mid = +1.331`
- 95% CI `[+0.644, +2.057]`

The full matched-demand sweep produced a positive signed separation area:

- mean `AREA = +1.240`
- 95% CI `[+0.589, +1.887]`

The midpoint states were then trained for another **120 epochs under identical current demand**. The separation remained:

- mean `H_hold120 = +1.208`
- 95% CI `[+0.499, +1.924]`
- retention ≈ **90.7%** of the original midpoint effect

The narrow supported statement is:

> **Within this symmetric synthetic autonomous recurrent system, opposite controlled demand histories can leave the same mature lineage in persistently different native dynamical organizations under the same current functional demand and matched training duration.**

Detailed result: [`docs/R8_M8_RESULT.md`](docs/R8_M8_RESULT.md)

Frozen preregistration: [`experiments/r8_m8/PREREGISTRATION.md`](experiments/r8_m8/PREREGISTRATION.md)

## What M8 does not establish

R8-M8 does **not** establish:

- mathematical bistability;
- formal thermodynamic or strict dynamical-systems hysteresis;
- conscious choice or intentional regime selection;
- information beyond the complete current state;
- essential chronology as a separate information channel;
- a universal trajectory-information principle;
- generalization to language models, transformers, biological systems, physical systems, or naturalistic tasks;
- practical superiority over conventional architectures.

The current safe terminology is **persistent history-dependent native organization** or **hysteresis-like operational path dependence**.

## Current next question

The next R8 priority is mechanism localization:

> **What maintains the persistent M8 history separation — encoder-side representation, recurrent-map geometry, or distributed encoder–recurrence coadaptation?**

Candidate confirmatory tests include encoder/recurrent cross-swaps, selective freezing during long identical-demand holds, longer hold durations, and transfer to unseen demand trajectories.

See [`RESEARCH_PRIORITIES.md`](RESEARCH_PRIORITIES.md) for the frozen scientific rules and next-study requirements.

---

# Other preserved research programs

The repository contains several related programs that remain part of the scientific record but are not substitutes for the current R8 claim.

## Observer / native-trajectory program

R2–R11 studied what information is accessible from evolving trajectory geometry, how recurrence changes generic versus trained-reader accessibility, how selective survival emerges, and where Euclidean preservation fails to predict functional usefulness.

Key boundaries:

- geometry-history accessibility can exceed endpoint snapshots;
- exact chronology was not established as essential;
- early transients can dominate some directional-history signals;
- recurrence can improve trained-reader compatibility while generic accessibility worsens;
- survival magnitude alone does not determine reader usefulness.

See [`docs/observer_program_r2_r11.md`](docs/observer_program_r2_r11.md).

## Active Latent Interrogation (ALI)

ALI studies whether controlled query-dependent perturbations of a frozen latent state can expose query-relevant information through the response of a frozen nonlinear transformation.

The reproducible ALI-N8-R1 result supports query-specific direction-dependent local responses under diagnostic decoders, but adaptive ALI did not beat direct memory readout and exhibited direction-only leakage. It does not establish that ALI replaces attention or provides a general memory architecture.

See:

- [`experiments/ali_n8_r1/PREREGISTRATION.md`](experiments/ali_n8_r1/PREREGISTRATION.md)
- [`docs/mechanism.md`](docs/mechanism.md)
- [`results/reproducible/ali_n8_r1/`](results/reproducible/ali_n8_r1/)

## Causal-control / perturbation program

R4B/R4C/R4D/R4E and JTP tested controllability, learned self-steering, low-dimensional control, state-conditioned intervention consequences, and instantaneous local-operator signatures.

Important negative boundaries are preserved: learned self-nudging did not produce a strong practical controller, R4E failed its primary Phase-I nonlinear gate, and JTP-1 did not find the preregistered seed-general instantaneous Jacobian-like trajectory-time signature.

## Reader robustness / affine geometry

AG3–AG5 studied why geometrically small representation errors can be functionally catastrophic and why larger errors can sometimes remain functionally acceptable. Local reader robustness matters; Euclidean distance alone is insufficient.

## Historical attractor-era work

The original attractor/trajectory prototypes motivated the project but contained claims that exceeded the available evidence. They remain preserved as project history rather than current proof.

See [`archive/`](archive/README.md) and [`docs/research_history.md`](docs/research_history.md).

## Reproducibility and scientific record

The repository intentionally keeps:

- preregistrations before outcome inspection;
- fresh seed definitions;
- negative and protocol-limited outcomes;
- post-run audits labeled separately from primary results;
- GitHub Actions workflows and result branches;
- current claim boundaries separate from historical ideas.

The project-wide rule remains:

> **A failed explanation is not the same as a failed phenomenon, and a promising secondary pattern is not promoted into a primary result after the fact.**
