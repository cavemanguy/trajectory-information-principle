# ND-R1 Preregistration — Training-Emergent Selective Preservation and Native Transient Geometry

**Status:** frozen before primary execution  
**Date:** 2026-09-04  
**Research branch:** native-dynamics-r1

## 1. Purpose

ND-R1 is the first experiment after the September 2026 research reset.

It returns the primary object of study to the **unperturbed recurrent trajectory itself**.

The experiment asks whether ordinary task training reproducibly transforms an initially broadly contractive recurrent system into one that selectively preserves different natural task distinctions, and how that transition appears in the native transient geometry.

The motivating historical observations come from Observer R8–R10. Those observations are treated as prior evidence, not as new ND-R1 results.

## 2. Primary research question

> Does successful training reproducibly transform initially indiscriminate contraction into relation-dependent selective preservation of natural task distinctions?

Formally, for a recurrent trajectory

\[
h_{t+1}=F_\theta(h_t),\qquad t=0,\ldots,11,
\]

consider a controlled pair of **natural inputs** \(x,x'\) that differ only in relation \(r\). Their native state difference is

\[
\Delta_t^{(r)} = h_t(x)-h_t(x').
\]

Define natural distinction survival

\[
S_r(t)=\frac{\|\Delta_t^{(r)}\|_2}{\|\Delta_0^{(r)}\|_2+\eta}.
\]

This is not an injected perturbation. Both members of the pair are ordinary valid task inputs.

## 3. Secondary research question

> As selective preservation emerges during training, how do the unperturbed R2-style trajectory statistics change?

ND-R1 will measure native trajectory speed, radius, turning, reversals, path efficiency, directional summaries, and time-resolved linear accessibility at the same frozen training checkpoints.

These measurements are secondary/descriptive in ND-R1. They may motivate a later preregistered mechanism experiment, but they cannot rescue a failed primary selective-preservation result.

## 4. Explicit exclusions

The ND-R1 primary experiment contains:

- **no injected perturbations**;
- **no active controller**;
- **no observer feedback into recurrence**;
- **no learned intervention policy**;
- **no selection of time points from test outcomes**;
- **no post-hoc replacement of seeds**;
- **no claim that chronology is encoded unless separately tested later**.

Any observer-like calculation is a passive measurement performed on frozen recorded trajectories.

## 5. Model/task lineage

ND-R1 will use the recovered Observer-core lineage:

- 8 categorical relations;
- 16 values per relation;
- 16-dimensional recurrent latent state;
- 12 recurrent transitions (13 recorded states including \(h_0\));
- encoder → recurrent core → relation-specific final readers;
- deterministic synthetic task generation with train/validation/test separation.

Before execution, the implementation must document whether every architectural/training detail is directly recovered from preserved source or reconstructed from the closest reproducible lineage. Reconstruction choices must be committed before primary results and cannot be tuned to match the historical R8–R10 pattern.

## 6. Fresh primary seeds

Fresh confirmatory seeds are frozen as:

\[
\boxed{13,\;29,\;53}
\]

Historical Observer seeds 7, 19, and 43 may be used only for provenance/reproduction diagnostics and are not the fresh confirmatory cohort.

No failed fresh seed may be replaced.

## 7. Fixed training schedule

Primary models are trained for exactly 100 epochs unless a pre-execution source-recovery amendment proves that the preserved lineage used a materially different fixed schedule.

Dense checkpoints are frozen at epochs:

\[
\boxed{0,1,2,5,10,20,40,60,80,100}
\]

Epoch 0 is the initialized, untrained recurrent system.

All listed checkpoints are retained regardless of later performance.

## 8. Training competence gate

Mechanistic interpretation requires successful task training.

A fresh seed passes the competence gate if its **validation** accuracy under the native trained final reader is at least 50% at epoch 100.

All three fresh seeds must pass.

If one or more fresh seeds fail this gate, ND-R1 is classified **Outcome A — training reproduction failure**. The trajectory measurements are still preserved but no claim about training-emergent selective preservation is promoted.

Test data are not used to choose checkpoints, hyperparameters, seeds, or stopping time.

## 9. Controlled natural-pair bank

For each relation \(r\), construct deterministic held-out controlled pairs \((x,x')\) such that:

- the other seven relation values are identical;
- relation \(r\) differs;
- both examples are valid ordinary task inputs;
- no artificial latent displacement is added.

Use exactly 2,048 held-out controlled pairs per relation per seed, generated from a fixed analysis-bank RNG namespace committed in the implementation before execution.

The same pair bank is evaluated at every training checkpoint for that seed.

## 10. Primary measurements

For each epoch \(e\), relation \(r\), transition \(t\), and seed, compute the median pairwise survival

\[
\widetilde S_{r,e}(t)=\operatorname{median}_{\text{pairs}} S_{r,e}(t).
\]

### 10.1 Global contraction

At epoch 0 define

\[
C_0 = \frac{1}{8}\sum_r \log \widetilde S_{r,0}(12).
\]

Negative \(C_0\) indicates average terminal contraction of natural relation distinctions.

### 10.2 Relation-selectivity index

Define

\[
G_e = \operatorname{SD}_{r=1}^{8}\left[\log \widetilde S_{r,e}(12)\right].
\]

The primary training-emergence contrast is

\[
\Delta G = G_{100}-G_0.
\]

### 10.3 Early-establishment diagnostic

At epoch 100, compute Spearman correlation across relations between

\[
\log \widetilde S_{r,100}(2)
\]

and

\[
\log \widetilde S_{r,100}(12).
\]

This tests the historical observation that much of the relation-survival ranking was established in the first two recurrent transitions. It is secondary and does not determine the primary outcome.

## 11. Primary uncertainty procedure

For each seed, use a deterministic paired bootstrap over controlled natural pairs within relation, 5,000 resamples, recomputing relation medians and \(G_e\) for each resample.

Report the 95% bootstrap interval for

\[
\Delta G = G_{100}-G_0.
\]

The RNG namespace and exact implementation are frozen before results.

No bootstrap result is used to tune training.

## 12. R2-style native transient measurements

At every checkpoint record the following from ordinary unperturbed trajectories:

1. mean and median state radius \(\|h_t\|\);
2. transition speed \(\|h_{t+1}-h_t\|\);
3. unit direction \(u_t=(h_{t+1}-h_t)/(\|h_{t+1}-h_t\|+\eta)\);
4. consecutive-direction cosine \(u_t\cdot u_{t+1}\);
5. reversal fraction, preregistered as consecutive-direction cosine < 0;
6. total path length;
7. endpoint displacement;
8. endpoint/path-length efficiency;
9. integrated direction \(\sum_t u_t\);
10. first direction \(u_0\);
11. final direction \(u_{11}\);
12. linear relation/value accessibility from each \(h_t\), trained on training trajectories and evaluated on held-out test trajectories;
13. linear accessibility from first, final, and integrated direction summaries under the same split discipline.

These measurements characterize the native trajectory and are not interpreted as evidence for chronology by themselves.

## 13. Training-time co-organization analysis

After the primary result is classified, ND-R1 will report checkpoint-wise associations between \(G_e\) and the preregistered native transient measurements.

These are **secondary associations**, not causal claims.

Because only ten checkpoints exist, exact effect curves and per-seed values will be emphasized rather than relying on a single correlation coefficient.

## 14. Frozen outcome classification

### Outcome A — training reproduction failure

At least one fresh primary seed fails the epoch-100 validation competence gate.

Interpretation: ND-R1 cannot adjudicate the historical training-emergence phenomenon under this reproduction.

### Outcome B — competent training without reproducible selective-preservation emergence

All three seeds pass competence, but the preregistered selective-preservation criterion below fails.

Interpretation: the historical R8–R10 selective-preservation transition is not independently reproduced under ND-R1.

### Outcome C — reproducible training-emergent selective preservation

All three fresh seeds satisfy all of:

1. \(C_0<0\) at initialization;
2. point estimate \(\Delta G>0\);
3. bootstrap 95% interval for \(\Delta G\) has lower bound > 0.

Interpretation: successful training reproducibly transforms initially contractive native dynamics into a more relation-selective preservation regime for natural task distinctions.

There is deliberately no stronger primary outcome in ND-R1. Any striking transient geometry remains a separately documented secondary phenomenon until independently preregistered.

## 15. Claim boundaries

Even Outcome C does **not** establish that:

- recurrence creates new task information;
- trajectory order itself is encoded or necessary;
- selective preservation is the cause of downstream reader performance;
- the phenomenon is universal across recurrent architectures;
- an attractor or chaotic mechanism is involved;
- perturbations are required;
- the phenomenon improves over feed-forward or attention systems;
- the effect has practical utility yet.

Outcome C supports only a reproducible training-induced change in how **natural task distinctions survive native recurrent evolution** in the tested architecture.

## 16. Practical-use bridge

If Outcome C is obtained, practical follow-ups may test whether the learned dynamics can be exploited for:

- compact task-selective filtering;
- early-exit or few-step computation;
- dynamically specialized interfaces to small readers;
- selective forgetting/compression;
- state-dependent logic/control introduced only in a separate intervention experiment.

Engineering utility is a separate question from the ND-R1 scientific claim.

## 17. Preservation rule

All outputs are retained regardless of outcome, including failed training seeds, epoch-0 measurements, negative relations, unexpected transients, and protocol amendments.

No previous research branch is deleted or reclassified to make ND-R1 appear more successful.
