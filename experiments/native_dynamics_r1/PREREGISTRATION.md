# ND-R1 Preregistration — Training-Emergent Selective Preservation and Native Transient Geometry

**Status:** frozen before primary execution  
**Date:** 2026-09-04  

## Purpose

ND-R1 is the first experiment after the September 2026 research reset. It returns the primary object of study to the **unperturbed recurrent trajectory itself**.

Primary question:

> Does successful training reproducibly transform initially broadly contractive recurrent dynamics into relation-dependent selective preservation of natural task distinctions?

Secondary question:

> As that transition occurs, how do R2-style native transient trajectory statistics change?

The historical R8–R10 observations motivate the experiment but are not counted as new ND-R1 evidence.

## No-perturbation boundary

The primary experiment contains:

- no injected perturbations;
- no active controller;
- no observer feedback;
- no learned intervention policy;
- no time-point selection from test outcomes.

Observer-like calculations are passive measurements on frozen native trajectories only.

## Natural distinction survival

For two ordinary valid inputs `x,x'` that differ only in relation `r`, define

\[
\Delta_t^{(r)}=h_t(x)-h_t(x')
\]

and

\[
S_r(t)=\frac{\|\Delta_t^{(r)}\|_2}{\|\Delta_0^{(r)}\|_2+\eta}.
\]

No latent displacement is injected; both members are natural task inputs.

## Model/task lineage

Use the recovered Observer-core lineage:

- 8 categorical relations;
- 16 values per relation;
- 16-dimensional recurrent latent state;
- 12 recurrent transitions / 13 states including `h0`;
- encoder → recurrent core → relation-specific final readers;
- deterministic train/validation/test separation.

Before execution the implementation must state which architectural/training details are directly recovered and which are reconstructed. Reconstruction choices are committed before outcomes and cannot be tuned to reproduce historical R8–R10 curves.

## Fresh primary seeds

\[
\boxed{13,29,53}
\]

Historical seeds 7, 19, 43 are provenance-only and do not count as fresh confirmation. No failed seed may be replaced.

## Fixed training and checkpoints

Train exactly 100 epochs unless a pre-execution source-recovery amendment establishes that the preserved lineage requires a materially different fixed schedule.

Frozen checkpoints:

\[
\boxed{0,1,2,5,10,20,40,60,80,100}
\]

All checkpoints are retained.

## Competence gate

All three fresh seeds must reach at least **50% validation accuracy** under the native trained final reader at epoch 100.

If any seed fails, classify:

**Outcome A — training reproduction failure.**

Test data cannot select training duration, checkpoint, hyperparameters, or seeds.

## Controlled natural-pair bank

For each relation, generate exactly **2,048 held-out controlled natural pairs** per seed:

- other seven relation values identical;
- target relation value differs;
- both examples ordinary valid inputs;
- fixed deterministic RNG namespace committed before execution.

The same pair bank is used at every checkpoint for a seed.

## Primary metrics

For epoch `e`, relation `r`, and time `t`, let

\[
\widetilde S_{r,e}(t)=\operatorname{median}_{pairs} S_{r,e}(t).
\]

Initialization contraction:

\[
C_0=\frac{1}{8}\sum_r \log \widetilde S_{r,0}(12).
\]

Relation-selectivity index:

\[
G_e=\operatorname{SD}_{r=1}^{8}\left[\log\widetilde S_{r,e}(12)\right].
\]

Primary emergence contrast:

\[
\Delta G=G_{100}-G_0.
\]

## Primary uncertainty

For each seed, deterministic paired bootstrap over natural controlled pairs within relation, **5,000 resamples**, recomputing relation medians and `G` each resample.

Report 95% CI for `Delta G`.

## Secondary early-establishment diagnostic

At epoch 100 compute across-relation Spearman correlation between survival at transition 2 and survival at transition 12.

This tests the historical observation that much of the final preservation ranking was established in the first two recurrent transitions. It does not determine the primary classification.

## R2-style native transient measurements

At every checkpoint measure from ordinary unperturbed trajectories:

1. state radius;
2. transition speed;
3. unit direction;
4. consecutive-direction cosine;
5. reversal fraction (`cosine < 0`);
6. total path length;
7. endpoint displacement;
8. endpoint/path-length efficiency;
9. integrated direction;
10. first direction;
11. final direction;
12. time-resolved held-out linear accessibility from each `h_t`;
13. held-out linear accessibility from first, final, and integrated direction summaries.

These characterize native dynamics and do not establish chronology by themselves.

Checkpoint-wise associations between `G_e` and these transient measurements are secondary/descriptive and cannot rescue a failed primary result.

## Frozen outcomes

### Outcome A — training reproduction failure

At least one fresh seed fails the validation competence gate.

### Outcome B — competent training without reproducible selective-preservation emergence

All seeds pass competence, but the Outcome C criterion fails.

### Outcome C — reproducible training-emergent selective preservation

All three fresh seeds satisfy:

1. `C0 < 0` at initialization;
2. point estimate `Delta G > 0`;
3. bootstrap 95% CI for `Delta G` has lower bound `> 0`.

There is intentionally no stronger primary outcome in ND-R1. Strong secondary transient phenomena must receive a separate preregistered follow-up rather than being promoted post hoc.

## Claim boundaries

Outcome C would not establish that recurrence creates new information, chronology is encoded, selective survival causes reader performance, perturbations are required, attractors/chaos are involved, the phenomenon is universal, or practical utility is established.

It would support only:

> successful training reproducibly changes how natural task distinctions survive native recurrent evolution, from initially contractive dynamics toward a more relation-selective preservation regime in the tested architecture.

## Practical-use bridge

If Outcome C is obtained, separate engineering follow-ups may examine compact task-selective filtering, early-exit computation, reader-specific recurrent preprocessing, selective forgetting/compression, and later state-dependent control/logic.

Engineering utility remains separate from the ND-R1 scientific claim.

## Preservation

Retain all outputs regardless of outcome, including failed seeds, negative relations, initialization behavior, unexpected transient phenomena, and amendments. No previous research branch is deleted or reclassified to make ND-R1 appear stronger.
