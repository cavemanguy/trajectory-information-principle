# Experiments and Evidence Status

## 1. Task

### Plain English

The current test asks a model to store eight kinds of relationships in a compressed latent memory and retrieve the relationship requested by a query. This is useful for isolating mechanisms but much narrower than language understanding.

### Technical description

The benchmark is an **8-relation synthetic memory task**. The autonomous experiment used seeds **5, 17, and 31**, and recovered aggregate summaries are checked into [`../results/`](../results/README.md). The task generator, exact splits, full configuration, checkpoints, per-seed rows, and per-example predictions are not yet checked in, so this is an evidence record but not a complete reproduction package.

## 2. Main comparisons

### Plain English

Query-directed probing had the highest mean among these controls, but only slightly exceeded a wider direct decoder. Three runs show the native result is repeatable at roughly this level, while the missing dispersion for the wider control still prevents a clean uncertainty comparison.

### Reported results

| Method | Mean accuracy ± SD | Parameters | Difference from native mean |
|---|---:|---:|---:|
| Native query-directed probing | **64.67% ± 1.81%** | 3,365 | — |
| Wider direct control | 63.60% (SD not recovered) | not recovered | -1.07 points |
| Direct frozen-state readout | 62.40% ± 1.16% | 1,972 | -2.27 points |
| Query-blind probing | 62.15% ± 1.28% | 3,109 | -2.52 points |

### Technical interpretation

The aggregate summaries establish run-to-run dispersion across three seeds for native, direct, and query-blind conditions. Without per-seed rows and paired predictions, a paired confidence interval or condition-difference test cannot be reconstructed. The wider direct control also lacks a recovered SD and parameter count. Optimization budgets and stopping rules must still be documented. The justified conclusion remains descriptive rather than a reliable superiority claim.

## 3. Probe-direction interventions

### Plain English

The strongest evidence comes from deliberately damaging direction selection. A wrong query, shuffled directions, or an average direction causes a large drop. The native direction is doing task-relevant work.

### Reported results

| Direction condition | Mean accuracy ± SD | Drop from native mean |
|---|---:|---:|
| Native query-directed probing | **64.67% ± 1.81%** | — |
| Wrong query used only for probe selection | 56.94% ± 3.22% | -7.73 points |
| Shuffled probe directions | 51.29% ± 2.85% | -13.38 points |
| Mean probe direction | 50.29% ± 1.17% | -14.38 points |

### Technical interpretation

Let $v_i=P(h_i,m_i,q_i)$. The interventions use

$$v_i^{\mathrm{wrong-q}}=P(h_i,m_i,q_j),\ j\ne i,$$

$$v_i^{\mathrm{shuffle}}=v_{\pi(i)},$$

or a normalized training-set mean direction. Assuming the readout retains the correct $q_i$ and no other path changes, the drops support causal dependence on native direction assignment within this trained model.

They do **not** establish query-only addressing. Since $v_i$ depends on $m_i$ and possibly memory-derived $h_i$, it can contain content-specific computation. They also do not show that response-based decoding is superior to decoding the same computation directly from $v$, $m$, or policy state.

## 4. Probe-direction mechanics

### Plain English

The learned directions have visible structure. Different queries applied to the same memory tend to select substantially different directions. The same query tends to select a more consistent direction across examples. Meanwhile, the average directions associated with different queries are nearly orthogonal on average.

This bridges the earlier geometry work to the current causal result: queries are not merely changing probe magnitude or adding undifferentiated noise. They organize direction selection. It still does not prove that the query alone is doing the addressing, because the current policy can inspect memory.

### Measured geometry

| Measurement | Value |
|---|---:|
| Same-memory, cross-query cosine | 0.2090 |
| Same-memory, cross-query cosine distance $(1-\cos)$ | 0.7910 |
| Within-query direction consistency | 0.7976 |
| Query-centroid cosine | -0.0348 |

### Technical interpretation

For directions $v(m,q)$, the same-memory cross-query statistic compares

$$\cos\big(v(m,q_i),v(m,q_j)\big),\qquad i\ne j.$$

Its mean near 0.209 indicates substantial angular separation when the query changes while memory is held fixed. Within-query consistency near 0.798 indicates that directions for a given query align relatively strongly across examples. Query-centroid cosine near zero indicates that query-level mean directions are approximately orthogonal on average.

These are descriptive geometric measurements, not a proof of discrete address vectors. Their interpretation depends on the sampling, averaging, normalization, and layer at which directions were measured. The causal interventions in Section 3 are needed to show that this structure matters for performance.

## 5. Corrected attention comparison

### Plain English

A comparable corrected positional-attention model performed much better—about 85% versus about 65% for ALI. That means the current experiment is evidence about an addressing mechanism, not evidence that ALI is a better architecture than attention.

### Reported result

| Model | Relations | Mean accuracy ± SD | Range | Parameters |
|---|---:|---:|---:|---:|
| Corrected positional attention | 8 | **84.81% ± 7.06%** | 79.14%–92.71% | 938 |
| Native ALI probe | 8 | 64.67% ± 1.81% | 63.19%–66.69% | 3,365 |

### Technical interpretation

The attention result is the stronger predictive baseline in the recovered comparable $N=8$ summaries. It is not parameter-matched to ALI and its larger dispersion deserves investigation, but neither fact reverses the observed accuracy gap. This comparison directly rules out presenting the current ALI system as an attention replacement or performance improvement.

The scientific purpose of ALI remains mechanistic: isolate whether controlled directional interventions can retrieve information from frozen latent memory and determine whether the useful signal lies in query-conditioned direction, local response, or ordered dynamics.

## 6. Critical next experiment: query-only addressing

### Plain English

Train a probe chooser that never sees memory. It receives only the query; the decoder still receives only query and response. This cleanly tests whether queries can act as addresses into shared latent memory.

### Technical protocol

$$\tilde v=P_q(q),\quad v=\frac{\tilde v}{\lVert\tilde v\rVert_2+\varepsilon},$$

$$r=F(m+\alpha v)-F(m),\quad \hat y=R(r,q).$$

Hold frozen memory, perturbation budget, response representation, readout boundary, splits, tuning budget, stopping rule, and seeds constant. Report all per-seed scores, mean, standard deviation, and a paired bootstrap confidence interval over test examples.

## 7. Leakage and falsification checks

### Plain English

Make sure the answer comes through the response rather than an accidental shortcut.

### Technical checks

- Verify $R$ cannot access $m$, $v$, $h$, or targets.
- Evaluate $\alpha=0$ to detect another information path.
- Decode diagnostically from $v$ alone and $r$ without $q$.
- Preserve direction norms when shuffling or randomizing.
- Build wrong-query pairs without altering label frequencies.
- Compute mean directions from training data only.
- Ensure interventions do not alter batch statistics or stochastic state elsewhere.

## 8. Claims ledger

| Statement | Current status |
|---|---|
| Native directions matter on the reported synthetic task | Supported across three-seed aggregate intervention results; independent replication pending |
| Query-directed probing reliably beats direct readout | Not established |
| Query alone can address latent memory | Open; requires $v=P(q)$ |
| Memory is usefully compressed | Not established by rate/distortion analysis |
| The system is robust | Not established |
| Dynamics converge or use attractors | Not part of current evidence |
| ALI is novel | Not established; literature review required |
| ALI replaces or outperforms attention | Contradicted by the current corrected $N=8$ comparison; attention is substantially stronger |
| Results generalize to language | Not established |

## 9. Reproducibility requirements

Aggregate CSV summaries and the seed set are present. Add the exact generator, model code, configs, dependency lock, commands, per-seed rows, raw predictions, aggregation script, and environment notes. Tables should ultimately be generated from the checked-in machine-readable outputs to reduce transcription errors.
