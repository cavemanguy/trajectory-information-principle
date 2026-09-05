# R8-M9 Result — Component Localization of Persistent History

**Status:** FROZEN PRIMARY RESULT

## Primary classification

**C2 — recurrent-map-carried contribution supported**

Orthogonal optimizer result: **O supported — optimizer state is not necessary for 120-epoch persistence under the preregistered reset intervention.**

## Validity and replication

All 12 fresh M9 families completed the frozen protocol successfully.

Before any localization claim was interpreted, M9 required the R8-M8 matched-midpoint history effect to replicate on fresh lineages. It did:

- `H_parent` mean = **+0.594405**
- bootstrap 95% CI = **[+0.229694, +1.032561]**

Thus the persistent-history midpoint phenomenon replicated strongly enough for the preregistered localization analysis.

## Component-localization result

At the matched midpoint, the encoder block `E` and recurrent-map block `F` were transplanted between the A-history and B-history states in a frozen 2x2 design. Native organization `Q` was measured immediately, before any additional training.

### Encoder contribution

- `E_effect` mean = **+0.076675**
- 95% CI = **[+0.026277, +0.141389]**

The effect was direction-consistent but did **not** reach the preregistered minimum magnitude of +0.25. Therefore the encoder contribution criterion was not promoted.

This is not evidence that the encoder contributes nothing. It means the frozen criterion for an encoder-carried contribution was not met.

### Recurrent-map contribution

- `F_effect` mean = **+0.517730**
- 95% CI = **[+0.199047, +0.899724]**

The recurrent-map criterion passed.

Descriptively, the mean recurrent-map effect accounts for about 87% of the mean parent separation (`0.517730 / 0.594405`), while the encoder main effect accounts for about 13%. These fractions are descriptive, not separately preregistered estimands.

### Encoder-recurrence interaction descriptor

- `I_EF` mean = **-0.228696**
- 95% CI = **[-0.325413, -0.131323]**

The frozen interaction descriptor criterion was not promoted because the preregistered absolute-magnitude threshold was not met. Preserve the estimate as a secondary mechanistic descriptor.

## Optimizer-state persistence result

M9 tested whether AdamW state was necessary to maintain the history separation during another 120 epochs of identical `lambda=0.50` demand.

At +120 epochs:

- **INHERITED optimizer:** `H = +0.460842`, 95% CI `[+0.117534,+0.924632]`
- **RESET optimizer:** `H = +0.460241`, 95% CI `[+0.127948,+0.910702]`
- **CROSSED optimizer:** `H = +0.475399`, 95% CI `[+0.123491,+0.935879]`

The preregistered reset-persistence criterion passed.

Therefore, within this experiment, **history-specific AdamW state is not necessary for persistence over the tested 120-epoch identical-demand hold**. This does not prove optimizer state can never affect the phenomenon under other schedules or optimizers.

## Frozen interpretation

The strongest defensible statement is:

> **In this tested synthetic recurrent system, most of the localized causal contribution to the persistent matched-demand history effect follows the learned recurrent map rather than the encoder block, and the separation survives removal of history-specific AdamW state over a 120-epoch identical-demand hold.**

A concise mechanistic progression is now:

`controlled demand history -> learned recurrent-map differences -> persistent differences in native dynamical organization under matched present demand`

## Claim boundary

R8-M9 does **not** establish:

- that the encoder has no history-dependent role;
- that the recurrent map is the unique carrier in every architecture;
- that a particular weight, neuron, direction, eigenspace, or low-rank subspace inside `F` has yet been localized;
- mathematical bistability or formal thermodynamic hysteresis;
- information beyond the complete current state;
- essential chronology;
- a universal trajectory-information principle;
- generalization beyond this synthetic recurrent system.

The next justified mechanistic question is **where inside the recurrent map `F` the persistent history-dependent organization is carried.**

## Provenance

- Workflow run: `33985146347`
- Experiment branch: `r8-m9-component-localization`
- Frozen aggregate result branch: `r8-m9-results`
- Authoritative generated record: `results/r8_m9/aggregate/FINAL_RESULT.md`
