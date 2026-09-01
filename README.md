# Active Latent Interrogation

**Status: early-stage research project. The name is provisional.**

Active Latent Interrogation (ALI) studies whether a learned intervention can retrieve query-relevant information from a frozen, compressed latent memory by measuring the memory's dynamical response.

The current mechanism is:

> query → learned probe direction → frozen compressed latent memory → controlled perturbation → dynamical response → retrieval

This repository was previously organized around a broader "Trajectory Information Principle" and an attractor-curve demonstration. Those materials are preserved in [`archive/`](archive/README.md), but they are not evidence for the current mechanism.

## The idea in plain English

Imagine that a model compresses several facts into one fixed internal memory. Instead of reading that memory only once, ALI asks a query, chooses a small direction in which to nudge the memory, and watches how the frozen memory system responds. A decoder receives the query and that response, then tries to recover the requested relation.

The central question is modest: **does choosing the perturbation direction based on the query contribute useful addressing information?**

The present experiments do not show that ALI replaces attention, improves real language models, discovers a new law of dynamics, or provides a general-purpose memory system.

## Technical summary

Let an encoder compress an input memory into a latent state

$$m = E(x), \qquad m \in \mathbb{R}^{d_m}.$$

During the probing experiment, the compressed memory mechanism is frozen. A policy chooses a normalized direction

$$v = \frac{P(h,m,q)}{\lVert P(h,m,q)\rVert_2 + \varepsilon},$$

where $q$ is the query and $h$ is any policy/controller state used by the implementation. A controlled intervention of magnitude $\alpha$ is applied,

$$m^{+} = m + \alpha v.$$

A frozen transition or response operator produces a response, for example

$$r = F(m^{+}) - F(m),$$

and the readout predicts the requested target using only the response and query:

$$\hat y = R(r,q).$$

The exact response statistic must be reported with each experiment; a final-state difference, a multi-step trajectory, and a Jacobian-vector response are not interchangeable.

See [`docs/mechanism.md`](docs/mechanism.md) for the architecture, equations, information boundaries, and controls.

## Current reported result

On an 8-relation synthetic memory task, the current reported accuracies are:

| Method | Accuracy |
|---|---:|
| Native query-directed probing | **64.67%** |
| Wider direct control | 63.60% |
| Direct frozen-state readout | 62.40% |
| Query-blind probing | 62.15% |

The small gap between native probing and the wider direct control is not, by itself, strong evidence for a general advantage. Run-to-run variation, uncertainty intervals, matched optimization, and independent replication are not yet documented here.

The more informative evidence is the intervention on the learned probe direction:

| Probe-direction condition | Accuracy |
|---|---:|
| Native query-directed probing | **64.67%** |
| Wrong query used only for probe selection | 56.94% |
| Shuffled probe directions | 51.29% |
| Mean probe direction | 50.29% |

These interventions support a narrow causal statement: **in this trained system and task, retrieval performance depends on using the native query-conditioned probe direction.** They do not yet isolate query-only addressing, because the current policy may inspect the memory while selecting the direction: $v=P(h,m,q)$.

The next critical control is therefore

$$v=P(q), \qquad \hat y=R(r,q),$$

with the readout still restricted to the probe response and query. This tests whether the query can select an effective intervention without using memory content inside the addressing policy.

Full result interpretation and caveats are in [`docs/experiments.md`](docs/experiments.md).

## What is established—and what is not

### Plain English

The results suggest that the direction of the nudge matters. Giving the probe policy the wrong query, mixing directions among examples, or replacing directions with their average substantially lowers accuracy. That is useful evidence that the policy is not merely adding arbitrary noise.

However, the current policy sees both the query and memory. It may be solving part of retrieval while constructing the probe rather than using the query as an independent address. The query-only control is needed to distinguish those possibilities.

### Technical interpretation

The direction interventions manipulate $v$ while leaving the trained response/readout path otherwise fixed. Their accuracy drops are consistent with $v$ carrying task-relevant, example-specific information. They do not uniquely identify the source of that information because

$$I(v;m,q) \neq I(v;q).$$

With $v=P(h,m,q)$, the policy can perform content-dependent computation before the perturbation. A query-only policy constrains the addressing channel and tests whether useful selectivity remains when $I(v;m\mid q)$ is removed by construction.

No claim of statistical significance is made until repeated runs, uncertainty estimates, and a fixed evaluation protocol are available.

## Repository map

- [`docs/mechanism.md`](docs/mechanism.md): current mechanism, equations, architecture, and leakage boundaries
- [`docs/experiments.md`](docs/experiments.md): reported results, controls, interpretation, and missing evidence
- [`docs/research-plan.md`](docs/research-plan.md): next experiments and reporting checklist
- [`docs/research_history.md`](docs/research_history.md): the observer, trajectory, perturbation, and geometry experiments that narrowed the current hypothesis
- [`archive/`](archive/README.md): superseded attractor-era prototype and claims, retained as research history

## Reproducibility status

The current ALI experiment implementation, configuration, seeds, checkpoints, and raw result files are not present in this repository at this revision. Therefore the numerical results above should be treated as a **reported experimental checkpoint**, not as independently reproducible results from the checked-in code.

Before presenting the results as reproducible, the repository should include:

1. synthetic-task generator and fixed train/validation/test split;
2. model definitions and exact information access for every component;
3. training and evaluation commands;
4. parameter counts and matched-compute controls;
5. multiple seeds, per-seed scores, mean, dispersion, and confidence intervals;
6. saved configs or checkpoints and machine-readable raw outputs.

## Scope and limitations

- The evidence currently comes from one synthetic 8-relation task.
- Accuracy is modest and close to strong direct controls.
- The current addressing policy is not query-only.
- The compressed representation has not yet been characterized by rate, distortion, or an information bottleneck measurement; "compressed" currently describes the architecture, not a demonstrated compression advantage.
- Direction interventions show dependence within the trained model, not broad causal generalization.
- There is no demonstrated advantage over attention and no claim that ALI replaces it.
- Novelty relative to active sensing, learned querying, memory networks, perturbation methods, and dynamical readouts has not been established.

## Citation and contact

This is an evolving independent research prototype by Zachary Daniels. If discussing it, cite the repository and a specific commit rather than treating the working project name or current interpretation as a settled result.

Issues and technically critical feedback are welcome.

## License

MIT. See [`LICENSE`](LICENSE).
