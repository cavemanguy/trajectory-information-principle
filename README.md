# Trajectory Information Project

## Current research priority — September 2026 reset

**Preservation rule: preserve everything; delete nothing from the scientific record.** Prior successful, negative, ambiguous, superseded, exploratory, and prototype work remains part of the repository and research history.

The forward research focus returns to the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, and eventually used practically?**

Observers are measurement tools. Perturbations are optional causal/control tools. Neither is assumed to be the source of trajectory information.

### Prioritized phenomena

1. **R8–R10 learned selective preservation** — training transformed initially indiscriminate contraction into relation-dependent anisotropic preservation through encoder–recurrence coadaptation. **Immediate priority.**
2. **R6/R8 readout preparation** — recurrence can make information less generically accessible while making it more useful to its trained downstream reader.
3. **R2/R3 native transient trajectory structure** — indirect, reversal-dominated trajectories and nonmonotonic transient accessibility; directional history exists, chronology not established.
4. **Coupled local-observer collective behavior** — local components differentiated and coordinated through a shared dynamical substrate; chaos/exotic-attractor interpretations remain unsupported.
5. **Shared-observer transient factor** — historical early transient task signal peaked before the endpoint; requires clean reproduction.
6. **R7 reader-conditioned recurrent plasticity** — recurrence can adapt terminal format toward altered reader geometry under controlled conditions.
7. **R11 survival/use dissociation** — geometric survival magnitude alone does not determine reader usefulness.
8. **R4E state-conditioned intervention consequences** — preserved as a later practical control/logic branch; R4E itself closed at Phase I.

### Immediate experiment

**ND-R1 — Training-Emergent Selective Preservation and Transient Geometry**

Reproduce the R8–R10 transition from initially indiscriminate contraction to learned relation-dependent selective preservation, while measuring **R2-style native transient dynamics at the same frozen training checkpoints**.

**Primary ND-R1 uses no perturbations and no active controller.**

See:

- [`RESEARCH_PRIORITIES.md`](RESEARCH_PRIORITIES.md) — current priority list, practical-use directions, and reset rules
- [`docs/EVIDENCE_LEDGER.md`](docs/EVIDENCE_LEDGER.md) — preserved evidence/negative-result map
- [`docs/research_history.md`](docs/research_history.md) — historical path, including failed and ambiguous branches
- [`docs/research-plan.md`](docs/research-plan.md) — current plan with the previous ALI-focused plan preserved below it

---

# Active Latent Interrogation

**Status: early-stage independent research project. The name is provisional.**

Active Latent Interrogation (ALI) studies whether controlled, query-dependent perturbations of a frozen latent state can expose query-relevant information through the response of a frozen nonlinear transformation.

The current mechanism is:

> query → learned probe direction → frozen latent memory → controlled perturbation → local response → retrieval

This repository was previously organized around a broader **Trajectory Information Principle** and an attractor-curve demonstration. Those materials are preserved in [`archive/`](archive/README.md), but they are not evidence for the current mechanism.

## The idea in plain English

Suppose a model stores several facts in one fixed-width internal state. Instead of only reading that state directly, ALI asks a question, chooses a small direction in which to perturb the state, observes how a frozen nonlinear transformation responds, and tries to recover the requested fact from that response.

The central question is modest:

> **Can the query select a perturbation direction that preferentially exposes the requested information?**

The current experiments do not show that ALI replaces attention, improves real language models, proves a new law of dynamics, or provides a general-purpose memory architecture.

## Current reproducible experiment: ALI-N8-R1

ALI-N8-R1 is a preregistered experiment created from first principles rather than reconstructed or tuned to match the older historical aggregate result. The locked specification is in [`experiments/ali_n8_r1/PREREGISTRATION.md`](experiments/ali_n8_r1/PREREGISTRATION.md).

The task stores eight categorical relations in a 16-dimensional continuous latent state. The encoder and frozen nonlinear transformation `F` are pretrained only to preserve the facts, then frozen. The main response is the symmetric finite difference

$$
r(m,v)=\frac{F(m+\alpha v)-F(m-\alpha v)}{2\alpha}.
$$

R1 compares adaptive, query-only, query-blind, fixed, random, zero-perturbation, and direct-read controls. It also preregisters a full 8×8 query/direction swap and 64 independently trained diagnostic decoders.

Primary seeds were fixed in advance: **5, 17, and 31**.

## R1 result

All three primary seeds completed.

Seeds **17** and **31** remained test-blind through core selection, ALI/control selection, and all diagnostic-decoder selections. Seed **5** has a documented protocol deviation: temporary core heads were evaluated on test before downstream selection. The deviation is preserved in [`PROTOCOL_DEVIATIONS.md`](experiments/ali_n8_r1/PROTOCOL_DEVIATIONS.md) and is not hidden or repaired post hoc.

Across the three primary seeds:

| Metric | Mean | Sample SD |
|---|---:|---:|
| Query-only ALI `P(q)` | **26.5450%** | 0.8661 pp |
| Adaptive ALI `P(m,q)` | 66.7783% | 5.0036 pp |
| Direct read from `m` | **69.5700%** | 4.8116 pp |
| Direct read from `F(m)` | 61.9125% | 3.5753 pp |
| Zero perturbation | 6.1583% | 0.2475 pp |
| Adaptive direction-only leakage | 69.8167% | 5.5493 pp |

Chance is **6.25%**.

Adaptive ALI does not beat the strongest direct-memory control. It also has substantial direction-only leakage, so the adaptive policy cannot yet be interpreted as a clean interrogation mechanism even though the wrong-memory intervention shows very strong causal dependence on the memory-conditioned direction.

The cleaner test is the query-only system:

$$
v=P(q),
$$

which cannot inspect the current memory while choosing its direction.

### Preregistered query-only selectivity endpoints

The native 8×8 swap keeps the reader's true query fixed and substitutes only the perturbation direction. Across the three seeds:

- **`D_native = +19.4327` percentage points mean**
- sample SD **1.0719 pp**

Because the native reader could be specialized to its training direction, R1 also trains a separate diagnostic decoder for every relation/direction pair: **64 diagnostic decoders per seed**.

Across the three seeds:

- **`D_decode = +9.5751` percentage points mean**
- sample SD **0.8901 pp**
- relation-level independent-decoder diagonal advantage was positive in **24/24 seed/relation combinations**

The three primary seeds were:

| Seed | Query-only accuracy | `D_native` | `D_decode` |
|---:|---:|---:|---:|
| 5 | 25.5650% | +19.1168 pp | +10.3704 pp |
| 17 | 26.8625% | +18.5543 pp | +8.6136 pp |
| 31 | 27.2075% | +20.6271 pp | +9.7414 pp |

Under the claim hierarchy locked before the primary runs, this meets the preregistered **Level-4 evidence pattern** for this learned latent system.

The narrow supported statement is:

> **In this learned frozen latent system, query-specific perturbation directions reproducibly produce direction-dependent local responses that preferentially expose information associated with their respective queries under the preregistered diagnostic decoder class.**

## What R1 does not establish

R1 does **not** establish that:

- ALI beats direct reading or attention;
- the 16-dimensional state provides a demonstrated compression advantage;
- the directions are universal, discrete, or orthogonal neural addresses;
- the mechanism generalizes to language models or transformers;
- the result is a new computational primitive;
- the broader Trajectory Information Principle is proven.

The current result is deliberately narrower: query-specific perturbation geometry exists reproducibly in this trained frozen latent system under the specified experimental conditions.

## Historical evidence

Before R1, recovered aggregate CSVs showed a historical N=8 query-conditioned probing result around **64.67%**, causal degradation under wrong/shuffled/mean direction interventions, and a corrected positional-attention control around **84.81%**. The exact historical implementation and raw per-seed evidence were not recoverable, so those CSVs remain preserved as historical aggregate evidence rather than reproducible proof.

R1 is scientifically separate from that historical result and was not tuned to reproduce it.

## Repository map

- [`experiments/ali_n8_r1/PREREGISTRATION.md`](experiments/ali_n8_r1/PREREGISTRATION.md): locked R1 design
- [`experiments/ali_n8_r1/run_core.py`](experiments/ali_n8_r1/run_core.py): reproducible frozen-core training
- [`experiments/ali_n8_r1/run_r1.py`](experiments/ali_n8_r1/run_r1.py): staged ALI, controls, diagnostics, and final evaluation
- [`experiments/ali_n8_r1/PROTOCOL_DEVIATIONS.md`](experiments/ali_n8_r1/PROTOCOL_DEVIATIONS.md): preserved protocol deviations
- [`results/reproducible/ali_n8_r1/aggregate/`](results/reproducible/ali_n8_r1/aggregate/): three-seed R1 aggregate
- [`results/reproducible/ali_n8_r1/`](results/reproducible/ali_n8_r1/): frozen per-seed evidence records
- [`results/`](results/README.md): historical and reproducible result index
- [`docs/mechanism.md`](docs/mechanism.md): mechanism and information boundaries
- [`docs/research_history.md`](docs/research_history.md): research path and negative/ambiguous results
- [`archive/`](archive/README.md): superseded attractor-era prototype and claims

## Reproducibility status

The R1 repository now contains the locked preregistration, implementation, pinned dependencies, seed definitions, frozen compact per-seed records, checkpoint and dataset hashes, intervention/diagnostic evidence references, and the three-seed aggregate.

The original GitHub Actions artifacts remain authoritative for the full generated output bundles, including per-example predictions, logs, matrices, counts, environment metadata, and checkpoint hashes. Compact repository records tie each frozen seed to its workflow run, artifact ID, and artifact SHA-256.

The historical aggregate experiments remain non-reproducible from the repository because their original implementation was not recovered.

## Scope and next questions

The evidence still comes from one synthetic eight-relation task. The next scientific work should test **why** the query-only geometry appears and **how general** it is rather than tuning R1 further. Relevant follow-ups include larger relation counts, tighter latent bottlenecks, continuous information, naturally trained representations, and eventually language-model representations.

Any scientific design change belongs in a new experiment version rather than being patched into R1.

## Citation and contact

This is an evolving independent research prototype by **Zachary Daniels**. Cite the repository and a specific commit or experiment version rather than treating the project name or current interpretation as a settled universal result.

Issues and technically critical feedback are welcome.

## License

MIT. See [`LICENSE`](LICENSE).
