# R8-M10 Preregistration — Off-Axis History Specificity Control

**Status:** FROZEN BEFORE IMPLEMENTATION AND BEFORE ANY OUTCOME.

This document is committed before the implementation, before any run, and before
any result is observed. Nothing below is revised after outcomes are seen.

## 1. Motivation and the question this replaces

R8-M8 established that two lineages forked from one matured state, given
different demand histories and then returned to identical `lambda=0.50` demand,
retain a separation in native organization `Q` on the A/B axis.

R8-M9 then established that this separation survives resetting the AdamW state,
and survives crossing the AdamW state between lineages. Optimizer-state inertia
is therefore already ruled out as a necessary carrier of the effect, and this
preregistration does **not** re-ask that question.

The open question is different and stronger:

> Does **any** sustained differential demand history produce a comparable
> persistent A/B reorganization, or is the persistent A/B effect **specific to
> the axis that was actually demanded**?

If an arbitrary off-axis demand history — applied to two relations that are
neither A nor B — produces a comparable A/B separation, then the M8 effect is
generic post-fork divergence and carries no axis-specific content. If instead
off-axis history reorganizes its own C/D axis while leaving the A/B axis
comparatively flat, the M8 effect is specific to the historically demanded
computational distinction.

This is a specificity test, not an optimizer test.

## 2. Design

One matured baseline per family, per the frozen M7R lineage engine
(`m7r_base.py`, pinned byte-identical into this directory). Maturity uses the
frozen trigger: competence plus A/B stability across three consecutive checks,
first eligible epoch 60, checked every 10 epochs from epoch 40, cap 400.

At maturity we identify:

- `A` = baseline survival winner relation
- `B` = baseline survival loser relation
- `C`, `D` = two deterministically chosen mid-ranked relations, neither `A` nor `B`

`C` and `D` are selected by ranking relations by terminal survival, removing `A`
and `B`, and taking the two most central remaining relations, with ties broken by
relation index. This selection is a pure function of the baseline state and is
fixed before any post-fork training.

From that single maturity state, **four** lineages are forked, each carrying an
identical copy of the model state and the AdamW state:

| Arm | Lineage | Demand schedule applied to | Sweep |
|-----|---------|----------------------------|-------|
| TRUE | `TRUE_A` | A/B axis | `A_HISTORY` |
| TRUE | `TRUE_B` | A/B axis | `B_HISTORY` |
| NULL | `NULL_C` | C/D axis | `A_HISTORY` |
| NULL | `NULL_D` | C/D axis | `B_HISTORY` |

with the frozen schedules

```
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
```

Both arms therefore receive an identical schedule shape, identical epoch counts,
identical loss form, and identical demand weights `w = 1 + 3*(1-lambda)` and
`1 + 3*lambda`. The **only** difference is which pair of relations the demand
weights are attached to.

All four lineages end at `lambda=0.50` — matched present demand. Every reported
comparison is made at that matched midpoint, at post-maturity epoch 120.

## 3. Metrics

For every lineage, at every checkpoint, we record **both**:

- `Q_AB = alignment[B] - alignment[A]` — the original M8 axis
- `Q_CD = alignment[D] - alignment[C]` — the off-axis pair

Recording both in both arms is required, not optional. It lets us verify that
the C/D manipulation genuinely reorganized its own axis before we interpret
anything about whether it spilled onto A/B.

Primary quantities, all at post-maturity epoch 120:

```
H_true_AB = Q_AB(TRUE_B) - Q_AB(TRUE_A)     # M8 replication, on the demanded axis
H_null_AB = Q_AB(NULL_D) - Q_AB(NULL_C)     # off-axis spillover onto A/B
H_null_CD = Q_CD(NULL_D) - Q_CD(NULL_C)     # did the off-axis manipulation work at all
H_true_CD = Q_CD(TRUE_B) - Q_CD(TRUE_A)     # symmetric spillover descriptor
SPECIFICITY = H_true_AB - H_null_AB          # paired within family
```

`SPECIFICITY` is the primary estimand. It is paired within family: both arms
descend from the same maturity state in the same family, so the bootstrap
resamples families, never arms.

## 4. Secondary diagnostic — update magnitude

Identical loss weights on different relation pairs do **not** guarantee identical
gradient or update magnitude. A/B are the extreme-ranked relations and C/D are
mid-ranked, and they can produce different gradients under the same weighting.
We therefore do not assert that the arms are matched in optimization pressure.

Instead we **measure** it. For every lineage we accumulate, over all 120
post-fork epochs:

- `grad_norm_sum`, `grad_norm_mean` — pre-clip global gradient norm
- `update_norm_sum`, `update_norm_mean` — L2 norm of the realized parameter delta per step
- `clip_fraction` — fraction of steps where clipping was active
- `param_distance_from_fork` — final L2 distance from the shared fork state

These are reported as diagnostics for interpretation. They are **not** gates and
they do not alter the primary classification. Their purpose is to let a reader
see whether an observed null-arm flatness could be explained by the null arm
simply having been pushed less far.

## 5. Frozen decision rule

Bootstrap: 5000 resamples over families, paired within family.

### Replication gate (must pass first)

```
R : mean(H_true_AB) >= 0.50  AND  ci95_lower(H_true_AB) > 0
```

If `R` fails, no specificity claim is promoted. The off-axis contrast is
uninterpretable without a replicated effect to contrast against.

### Manipulation-check gate

```
MC : mean(H_null_CD) >= 0.50  AND  ci95_lower(H_null_CD) > 0
```

`MC` asks whether the off-axis demand actually reorganized the off-axis pair. If
`MC` fails, the null arm did not do its job, and a flat `H_null_AB` cannot be
read as evidence of specificity — it would be equally consistent with the C/D
manipulation having been ineffective. This gate is what makes a null result
interpretable, and it is why `Q_CD` is mandatory.

### Equivalence margin for null flatness

`CI lower bound <= 0` is not an equivalence test. A clearly negative effect would
satisfy it. We therefore preregister a two-sided equivalence margin:

```
DELTA = 0.25
FLAT(H_null_AB) : ci95_lower(H_null_AB) > -DELTA  AND  ci95_upper(H_null_AB) < +DELTA
```

The **entire** confidence interval must lie inside `(-0.25, +0.25)` for the null
arm to be called flat. `DELTA = 0.25` is the same magnitude threshold the M9
component-localization gates already use for "an effect worth promoting", so a
null-arm effect too small to have been promoted as an effect is what we are
willing to call flat.

### Separation criterion

```
SEP : mean(SPECIFICITY) >= 0.50  AND  ci95_lower(SPECIFICITY) > 0
```

### Classification

| Code | Condition | Meaning |
|------|-----------|---------|
| `S0` | `R` fails | True-arm replication failure; contrast uninterpretable |
| `S1` | `R` passes, `MC` fails | Off-axis manipulation ineffective; specificity untestable this run |
| `S2` | `R`, `MC` pass; `SEP` passes and `FLAT(H_null_AB)` | Strong specificity: demanded axis reorganizes, off-axis history does not spill onto A/B |
| `S3` | `R`, `MC` pass; `SEP` passes, `FLAT` fails | Partial specificity supported; null A/B effect is not established equivalent to zero |
| `S4` | `R`, `MC` pass; `SEP` fails | Specificity criterion not met; equivalence/comparability is not established by this failure |

`S4` is a real and publishable outcome. It would mean the preregistered degree
of axis specificity was not demonstrated in this run. It does **not** by itself
establish that the demanded-axis and off-axis effects are equivalent or
comparable, and it is not to be repaired, re-run, or reframed after the fact.

## 6. Mandatory distribution reporting

R8-M8 reported a mean `H_mid` of `+1.331` against a median of `+0.943`, over a
visibly bimodal set of families — roughly half near `0.05-0.23` and half near
`1.66-3.10`. The mean described no observed family.

Therefore, for **every** primary quantity, the classifier must report:

- all per-family values
- mean, median, min, max
- count of families with a positive value

A classification is not to be read without the distribution beside it. If the
per-family values are bimodal, that fact belongs in the result document
regardless of what the aggregate gate says.

## 7. Validity conditions

A family is valid only if all hold:

- maturity was reached within the epoch cap
- `A != B`, and `C`, `D` are disjoint from `A`, `B`, and from each other
- all four lineages verify byte-identical fork identity from the shared maturity
  state, by SHA256 over the model state dict and over the optimizer state
- all four lineages complete exactly 120 post-fork epochs
- all recorded values are finite

Families failing any condition are reported, not silently dropped. If any family
fails, the classification is a validity failure code, not a scientific result.

## 8. Fresh seeds

```
1061, 1078, 1094, 1113, 1129, 1146, 1164, 1183, 1201, 1218, 1236, 1254
```

These are disjoint from the R8-M9 seed set (631-821) and the R8-M11 seed set
(839-1042). Derived seeds use the namespace `R8-M10|{seed}|{name}`, so no
derived stream collides with any prior experiment.

## 9. Scope limits

This experiment does not establish:

- mathematical bistability or formal thermodynamic hysteresis
- information beyond the complete current state
- essential chronology
- a universal trajectory-information principle
- generalization beyond this tested synthetic recurrent system

It tests exactly one thing: whether persistent A/B reorganization is specific to
the historically demanded axis, or is produced comparably by arbitrary off-axis
demand history.
