# Experiments and Evidence Status

## 1. Task

### Plain English

The current test asks a model to store eight kinds of relationships in a compressed latent memory and retrieve the relationship requested by a query. This is useful for isolating mechanisms but much narrower than language understanding.

### Technical description

The benchmark is an **8-relation synthetic memory task**. The task generator, splits, class balance, chance level, protocol, seeds, and raw outputs are not yet checked in. This page therefore records reported point estimates and limited interpretations, not a reproduction package.

## 2. Main comparisons

### Plain English

Query-directed probing scored highest, but only slightly above a wider direct decoder. That difference could be meaningful, noise, or optimization. Multiple runs are needed.

### Reported results

| Method | Accuracy | Difference from native |
|---|---:|---:|
| Native query-directed probing | **64.67%** | — |
| Wider direct control | 63.60% | -1.07 points |
| Direct frozen-state readout | 62.40% | -2.27 points |
| Query-blind probing | 62.15% | -2.52 points |

### Technical interpretation

Without per-seed results or paired predictions, no confidence interval or significance test can be computed responsibly. Parameter counts, optimization budgets, and stopping rules must also be documented. The justified statement is descriptive: the reported native score is 64.67%, versus 62.15%–63.60% for these controls. A reliable advantage is not yet established.

## 3. Probe-direction interventions

### Plain English

The strongest evidence comes from deliberately damaging direction selection. A wrong query, shuffled directions, or an average direction causes a large drop. The native direction is doing task-relevant work.

### Reported results

| Direction condition | Accuracy | Drop from native |
|---|---:|---:|
| Native query-directed probing | **64.67%** | — |
| Wrong query used only for probe selection | 56.94% | -7.73 points |
| Shuffled probe directions | 51.29% | -13.38 points |
| Mean probe direction | 50.29% | -14.38 points |

### Technical interpretation

Let $v_i=P(h_i,m_i,q_i)$. The interventions use

$$v_i^{\mathrm{wrong-q}}=P(h_i,m_i,q_j),\ j\ne i,$$

$$v_i^{\mathrm{shuffle}}=v_{\pi(i)},$$

or a normalized training-set mean direction. Assuming the readout retains the correct $q_i$ and no other path changes, the drops support causal dependence on native direction assignment within this trained model.

They do **not** establish query-only addressing. Since $v_i$ depends on $m_i$ and possibly memory-derived $h_i$, it can contain content-specific computation. They also do not show that response-based decoding is superior to decoding the same computation directly from $v$, $m$, or policy state.

## 4. Critical next experiment: query-only addressing

### Plain English

Train a probe chooser that never sees memory. It receives only the query; the decoder still receives only query and response. This cleanly tests whether queries can act as addresses into shared latent memory.

### Technical protocol

$$\tilde v=P_q(q),\quad v=\frac{\tilde v}{\lVert\tilde v\rVert_2+\varepsilon},$$

$$r=F(m+\alpha v)-F(m),\quad \hat y=R(r,q).$$

Hold frozen memory, perturbation budget, response representation, readout boundary, splits, tuning budget, stopping rule, and seeds constant. Report all per-seed scores, mean, standard deviation, and a paired bootstrap confidence interval over test examples.

## 5. Leakage and falsification checks

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

## 6. Claims ledger

| Statement | Current status |
|---|---|
| Native directions matter on the reported synthetic task | Supported by reported interventions; replication pending |
| Query-directed probing reliably beats direct readout | Not established |
| Query alone can address latent memory | Open; requires $v=P(q)$ |
| Memory is usefully compressed | Not established by rate/distortion analysis |
| The system is robust | Not established |
| Dynamics converge or use attractors | Not part of current evidence |
| ALI is novel | Not established; literature review required |
| ALI replaces or outperforms attention | Not claimed or tested |
| Results generalize to language | Not established |

## 7. Reproducibility requirements

Add the exact generator, model code, configs, dependency lock, seeds, commands, raw predictions, aggregation script, and environment notes. Generate tables from machine-readable outputs to reduce transcription errors.
