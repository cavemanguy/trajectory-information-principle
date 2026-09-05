# R8-M9 Preregistration — Null-History Control for the R8-M8 Y3 Result

**Status: frozen before any fresh R8-M9 outcome is generated.**

This preregistration is fixed in advance. No threshold, seed, schedule, duration,
maturity rule, null-pair rule, or classification rule may be changed after fresh
outcomes are exposed.

## Motivation

R8-M8 produced **Y3 — persistent history-dependent regime separation supported**.
Two branches were forked from one mature lineage, driven through opposite
continuous demand sweeps, and compared at a matched midpoint (`lambda=0.50`,
120 post-maturity epochs on both sides). Mean `H_mid` was `+1.331`, and about
90.7% of the separation survived a further 120 epochs of identical demand.

R8-M8 contains no arm in which the two branches receive histories that differ in
**optimizer magnitude but not in scientific meaning**. Both M8 branches carry
their own AdamW moment estimates and weight-decay trajectories through opposite
schedules. A converged network under `LR=1e-3` with weight decay is expected to
drift slowly, so the observed persistence is also consistent with ordinary
optimizer inertia plus slow drift, with no history-specific content.

R8-M9 exists to separate those two accounts. It is a control study for a frozen
prior result. It does not revise, rescue, or restate the M8 Y3 classification,
which stands as recorded regardless of the R8-M9 outcome.

## Fresh seeds

R8-M9 uses twelve fresh families. No R8-M8 family is reused:

```text
631, 648, 664, 683, 701, 718, 736, 754, 771, 789, 806, 824
```

## Shared lineage protocol

Unchanged from R8-M8 and executed once per family:

- architecture, data generator, and optimizer settings are the frozen
  `m7r_base.Core` / AdamW configuration;
- baseline training runs to the M7R maturity trigger (competence and A/B
  stability across three consecutive recorded checks, first eligible at epoch 60,
  checked every 10 epochs from epoch 40, capped at epoch 400);
- at maturity epoch `M`, relation `A` is the survival winner and `B` the survival
  loser;
- the complete model and optimizer state at `M` is the single fork point for
  every branch in both arms.

## The two arms

Both arms fork from the same maturity state and use **identical lambda
schedules**, identical durations, identical batch order derivation, identical
optimizer configuration, and identical total post-maturity epoch counts.

Demand weighting is the frozen M8 form, generalized from the pair `(A,B)` to a
pair `(P,Q_r)`:

```text
w[P]   = 1 + 3*(1 - lambda)
w[Q_r] = 1 + 3*lambda
```

**TRUE arm** — an exact R8-M8 reproduction on fresh seeds. `(P,Q_r) = (A,B)`.

- `A_SWEEP`: lambda 0.00 (60 epochs), 0.25, 0.50, 0.75, 1.00 (30 epochs each)
- `B_SWEEP`: lambda 1.00 (60 epochs), 0.75, 0.50, 0.25, 0.00 (30 epochs each)

**NULL arm** — same schedules, meaningless target. `(P,Q_r) = (C,D)`, where `C`
and `D` are two relations that are neither `A` nor `B`.

- `C_SWEEP`: the `A_SWEEP` schedule
- `D_SWEEP`: the `B_SWEEP` schedule

`C` and `D` are selected deterministically from the maturity survival ranking:
rank all eight relations by terminal survival, remove `A` and `B`, and take two
adjacent relations from the middle of the remaining ordering, with their order
fixed by a seed-derived draw. Mid-ranked relations are used so that the null arm
receives neither the established specialist nor the weakest relation.

The null arm therefore experiences the same number of epochs, the same weighting
magnitudes on the same epochs, and the same optimizer-state divergence pressure
as the true arm. Only the identity of the demanded relations differs.

## Metric

Unchanged from R8-M7R/M7I/M8, and **measured on the `A`/`B` axis in both arms**:

```text
L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)
Q   = L_B - L_A
```

For each arm, with `fwd` the branch starting at lambda 0.00 and `rev` the branch
starting at lambda 1.00:

```text
H(lambda) = Q_rev(lambda) - Q_fwd(lambda)
H_mid     = H(0.50)
AREA      = integral_0^1 H(lambda) dlambda   (trapezoidal over the five levels)
```

Measuring the null arm on the untouched `A`/`B` axis is the point of the design.
The null arm perturbs `C`/`D` while `Q` continues to read `A`/`B`, so any `H`
it produces is separation not attributable to `A`/`B` demand history.

## Long hold

Identical to M8. At the first `lambda=0.50` checkpoint in each of the four
branches, the complete model and optimizer state is cloned and trained a further
120 epochs at fixed `lambda=0.50`, recording at +30, +60, +90, +120.

```text
H_hold120 = H(0.50) after 120 additional matched-demand epochs
```

## Primary contrast

```text
CONTRAST_mid = H_mid(TRUE) - H_mid(NULL)
```

evaluated per family and aggregated across the twelve families with a
deterministic 5,000-resample paired-family bootstrap that resamples families
jointly, preserving the TRUE/NULL pairing within each family.

## Preregistered gates

**True-arm replication gate.** The TRUE arm must reproduce M8 on fresh seeds:

1. mean `H_mid(TRUE) >= +0.50` with bootstrap 95% CI lower bound `> 0`; and
2. mean `AREA(TRUE) >= +0.25` with CI lower bound `> 0`; and
3. mean `H_hold120(TRUE) >= +0.25` with CI lower bound `> 0`.

**Null-flatness condition.** The NULL arm is flat when the bootstrap 95% CI lower
bound on mean `H_mid(NULL)` is `<= 0`.

**Contrast gate.** Specificity is supported when mean `CONTRAST_mid >= +0.50`
with paired bootstrap 95% CI lower bound `> 0`.

## Classification

Exactly one outcome is assigned, by the frozen rule in `classify_r8_m9.py`:

- **N0 — true-arm replication failure; null contrast uninterpretable.**
  The TRUE arm did not reproduce M8 on fresh seeds. No inertia conclusion is
  drawn in either direction. This is a replication-boundary outcome and must not
  be paraphrased as evidence against M8.

- **N1 — null reproduces the effect; M8 separation is consistent with optimizer
  inertia.** The TRUE arm replicated, but the contrast gate failed: driving two
  scientifically meaningless relations produced comparable `A`/`B` separation.
  The M8 Y3 classification still stands as its frozen record, but the persistent
  separation is then not established as history-specific, and the operational
  reading of M8 must be narrowed accordingly.

- **N2 — partial specificity; true arm exceeds a nonzero null baseline.**
  The contrast gate passed while the null arm was itself not flat. Part of the
  M8 separation is history-specific and part is generic post-fork divergence.

- **N3 — history specificity supported; M8 effect is not optimizer inertia.**
  The contrast gate passed and the null arm was flat. The M8 separation is
  specific to `A`/`B` demand history rather than to the fact of divergent
  optimizer state.

## Mandatory distribution reporting

The R8-M8 per-family `H_mid` values were bimodal: six families below `+0.23` and
six above `+1.65`, with mean `+1.331` against median `+0.943`, so the mean
described no observed family.

R8-M9 must therefore report, for `H_mid(TRUE)`, `H_mid(NULL)`, and
`CONTRAST_mid`: every per-seed value, the median, the minimum, the maximum, and
the count of positive families. This reporting is mandatory and independent of
the classification. A mean may not be published for these quantities without the
accompanying median and per-seed values.

Whether the TRUE arm reproduces the M8 bimodality is recorded as a descriptive
secondary observation. It does not enter any gate.

## Validity conditions

A family is valid only if all hold:

- the maturity trigger was reached before epoch 400;
- every branch fork reproduced its source state hash exactly;
- all recorded `Q`, `h0`, `h12`, and `G` values are finite;
- all four branches recorded 5 sweep checkpoints and 4 hold checkpoints.

If any family fails maturity, the run is `V0`. If a mature family fails any other
validity condition, the run is `V1`. In both cases no N-classification is
assigned. Following the project rule, a failed validity gate is preserved as the
frozen outcome and is not repaired after the fact.

## Secondary, non-gating observations

- midpoint latent `h0` and `h12` distances between paired branches, per arm;
- per-relation `h12` accuracy at each midpoint;
- the `C`/`D` identities selected per family;
- whether the null arm shifts `Q` on the `C`/`D` axis, confirming that the null
  demand had a real optimization effect somewhere.

The last item matters: if the null arm produced no measurable effect anywhere,
the null would be vacuous rather than matched. This observation is descriptive
and does not change any gate.

## Scope

R8-M9 is confined to the same synthetic 8-relation, 16-value, 16-dimensional
autonomous recurrent system. It cannot establish generalization to other
architectures, scales, data, or model families, and no R8-M9 outcome licenses
any claim about language models, biological systems, or physical systems.
