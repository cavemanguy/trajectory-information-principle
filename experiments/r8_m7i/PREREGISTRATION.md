# R8-M7I Preregistration — Inverted-Demand Same-Lineage Mirror

**Status:** FROZEN BEFORE INVERTED OUTCOME INSPECTION

**Parent:** R8-M7R = D0 — reversible demand tracking not supported.

## Question

> Using the exact same training data, initialization, minibatch order, maturity checkpoint, and native A/B identities as R8-M7R, what happens if the post-maturity terminal-demand schedule is inverted from A → B → A to B → A → B?

This is a paired mirror experiment, not a fresh-seed replication. Its purpose is to test whether R8-M7R's resistance to B takeover was partly caused by first reinforcing the already-established specialist A for an additional 40 epochs before challenging it.

## Exact lineage reuse

Use the same 12 family seeds as R8-M7R:

`[214, 230, 247, 263, 279, 296, 313, 329, 346, 362, 378, 397]`

Reuse the exact R8-M7R deterministic seed namespace `R8-M7R` for:

- train/validation/test memories;
- validation/test presentation permutations;
- pair banks;
- parameter initialization;
- per-epoch presentation permutations;
- per-epoch minibatch ordering.

Use the exact R8-M7R architecture, optimizer, maturity trigger, competence thresholds, 40-epoch phase length, and 4.0-versus-1.0 normalized demand weighting.

Before any inverted post-maturity result can be interpreted, each family must reproduce the frozen `M7R_BASELINE_REFERENCE.json` entry:

- maturity epoch M exact;
- A exact;
- B exact;
- baseline Q within absolute tolerance 1e-5.

Any mismatch yields **V0 — baseline lineage reproduction failure** for the cross-family study. No post-hoc repair is permitted.

## Inverted post-maturity conditions

At M, copy the exact model and optimizer state into all conditions.

### MIRROR

Terminal h12 demand schedule:

- M+1 through M+40: **B**;
- M+41 through M+80: **A**;
- M+81 through M+120: **B**.

h0 loss remains symmetric.

### FIXB

Terminal h12 demand stays on **B** for all three 40-epoch phases. This controls for persistent loser-directed weighting and continued training.

### H0MIRROR

The B → A → B schedule is applied only to h0 loss. Terminal h12 loss remains symmetric.

## Native metric

Retain the R8-M7R definition:

`Q = L_B - L_A`

where positive Q favors B and negative Q favors A.

For MIRROR define:

- `Q_B1 = Q(M+40)`
- `Q_A = Q(M+80)`
- `Q_B2 = Q(M+120)`
- `BA_shift = Q_A - Q_B1`
- `AB_shift = Q_B2 - Q_A`

## Primary mirror-tracking criteria T

All CIs use deterministic 5,000-resample paired bootstraps over the 12 matched family seeds.

T is supported only if all six hold:

1. mean `Q_B1 >= +0.20` and its 95% CI lower bound > 0;
2. mean `BA_shift <= -0.75` and its 95% CI upper bound < 0;
3. mean `Q_A <= -0.20` and its 95% CI upper bound < 0;
4. mean `AB_shift >= +0.75` and its 95% CI lower bound > 0;
5. mean `Q_B2 >= +0.20` and its 95% CI lower bound > 0;
6. mean `Q_B2 - baseline_Q >= +0.75` and its 95% CI lower bound > 0.

This intentionally requires actual B-favored native organization, not merely movement toward B.

## Specificity controls S

### C1 — MIRROR versus FIXB during the A phase

At M+80 require:

`Q_A_MIRROR - Q_A_FIXB <= -0.50`

with 95% CI upper bound < 0.

This asks whether the middle A demand pulls the dynamics back toward A beyond what occurs under persistent B demand.

### C2 — terminal-demand versus h0-weighting

Define mirror amplitude:

`AMP = 0.5 * ((Q_B1 - Q_A) + (Q_B2 - Q_A))`.

Require:

`AMP_MIRROR - AMP_H0MIRROR >= +0.25`

with 95% CI lower bound > 0.

S is supported only if C1 and C2 both hold.

## Exact reassignment criterion E

For each MIRROR family record whether:

- B is exact terminal-survival winner at M+40;
- A is exact winner at M+80;
- B is exact winner at M+120.

E is supported if the full B → A → B exact-winner sequence occurs in at least 8 of 12 paired families.

## Frozen classifications

After validity:

- **I0 — inverted mirror tracking not supported:** T false.
- **I1 — inverted relative tracking supported:** T true but S or E false.
- **I2 — inverted demand-specific specialist reassignment supported:** T true, S true, E true.
- **V0 — baseline lineage reproduction failure.**
- **V1 — post-maturity execution validity failure.**

No threshold, phase duration, demand magnitude, baseline reference, or classification rule may be changed after inverted outcomes are exposed.

## Prespecified comparison to R8-M7R

Because the same baseline lineage is reused, report a paired descriptive comparison between:

- immediate B challenge in R8-M7I: `Q_B1` at M+40;
- delayed B challenge in R8-M7R: `Q_B` after 40 additional A-reinforcement epochs and then 40 B-demand epochs.

Report paired mean difference `Q_B1_I - Q_B_R7R` and a 95% bootstrap CI. This comparison is secondary and cannot change I0/I1/I2.

A positive difference would be consistent with the hypothesis that the extra A-reinforcement phase in R8-M7R increased resistance to later reassignment. It would not by itself prove formal hysteresis.

## Claim boundary

Under I2, the strongest allowed conclusion is:

> In the same mature recurrent lineages and exact same training datasets used by R8-M7R, reversing the order of controlled post-maturity terminal demand produces reversible B → A → B reassignment of native relation-selective preservation beyond matched controls.

Even I2 would not establish conscious choice, universal trajectory computation, strong emergence, formal hysteresis, essential chronology, or generalization beyond this synthetic system.
