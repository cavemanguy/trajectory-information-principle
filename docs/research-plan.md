# Research Plan

## Immediate priority: isolate addressing

### Plain English

The next job is to determine where useful information enters the probe. The cleanest test is a policy that selects direction from the query alone.

### Technical plan

Implement $v=P(q)$ beside $v=P(h,m,q)$. Use identical frozen memories, perturbation budgets, response features, readout access, splits, and examples. Log per-example predictions for paired analysis.

Every outcome is informative: if query-only approaches the current policy, memory inspection may be unnecessary; if it matches query-blind probing, content-dependent selection may account for the effect; an intermediate score suggests both contribute.

## Experiment sequence

### Plain English

Run the simplest tests that could disprove the preferred explanation before scaling up.

### Technical sequence

1. Reproduce the checked-in three-seed aggregate summaries from a fixed script.
2. Recover or rerun per-seed rows and retain paired per-example predictions for uncertainty on condition differences.
3. Add query-only $P(q)$ and zero-perturbation controls.
4. Decode from $v$, $r$, $m$, and allowed combinations to locate information channels.
5. Sweep perturbation magnitude $\alpha$.
6. Match parameter count and compute for direct controls.
7. Test held-out entity/relation combinations.
8. Only then vary memory load, relation count, latent width, and response horizon.

## Metrics

### Plain English

Accuracy alone can hide whether a result is stable, fair, or driven by one easy relation. Report overall scores and behavior across runs and relations.

### Technical reporting

- per-seed accuracy, mean, and standard deviation;
- paired confidence intervals for differences;
- accuracy by relation and class frequency;
- parameter count, steps, and approximate compute;
- perturbation- and response-norm distributions;
- chance, majority-class, and query-only-without-memory baselines;
- exact split construction and combination overlap.

Do not headline the best seed.

## Falsifiable hypotheses

### Plain English

The project needs predictions that can turn out to be wrong.

### Technical hypotheses

**H1 — Direction specificity.** Native $v_i$ beats norm-matched shuffled, mean, and random directions.

**H2 — Query-only addressability.** $P(q)$ beats query-blind $P(h,m)$ with the same response/readout boundary.

**H3 — Response necessity.** With $\alpha=0$, performance falls to the no-response baseline.

**H4 — Content-dependent advantage.** $P(h,m,q)$ beats $P(q)$ after capacity and optimization matching.

**H5 — Compositional generalization.** Any advantage persists on held-out relation/entity combinations.

## Deferred questions

### Plain English

Language models, hardware, attractors, and attention-replacement claims should wait until the basic information path is understood.

### Technical scope

Defer natural-language scaling, transformer-attention comparisons, physical analogies, convergence claims, and compression advantages until the mechanism is reproducible and query-only control is resolved. Before any novelty claim, review active sensing, adaptive measurement, memory networks, learned addressing, perturbation analysis, dynamical readouts, and control-theoretic observability.
