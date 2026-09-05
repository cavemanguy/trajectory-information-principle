# R8-M4 — Functional Necessity of Dynamical Specialization

## Frozen primary result

**Classification: F3 — selective-specialization contribution supported.**

All 12 fresh baseline families passed the lineage-validity gate. The selectivity-equalization intervention passed the preregistered manipulation gate:

- `G_E < G_B` in **12/12** seeds;
- mean `DeltaG(E-B) = -0.506315`, bootstrap 95% CI `[-0.562320, -0.447217]`;
- mean relative reduction in selectivity = **0.666**.

The primary epoch-100 test h12 endpoint then showed:

- mean `Delta(E-B) = -0.030413`, 95% CI `[-0.041504, -0.018127]`;
- mean `Delta(E-M) = -0.025746`, 95% CI `[-0.036619, -0.014523]`;
- mean `G_M-G_E = 0.577185`, 95% CI `[0.497767, 0.646360]`.

Under the frozen decision rule this satisfies **F3**: specifically suppressing relation-selective native survival impaired terminal task performance beyond the matched mean-survival control.

## Preregistered secondary: where the functional loss occurs

R8-M4 preregistered a secondary check of whether the baseline survival-winning relation suffers disproportionately under equalization. Using the preserved 12 fresh-family outputs:

- baseline survival winner mean h12 test accuracy: **0.7434**;
- mean h12 test accuracy across the other seven relations: **0.1393**;
- winner `Delta(E-B)`: **-0.30355**, bootstrap 95% CI `[-0.39019, -0.22393]`;
- non-winner mean `Delta(E-B)`: **+0.00861**, 95% CI `[-0.00393, +0.02060]`;
- paired winner-minus-nonwinner drop contrast: **-0.31216**, 95% CI `[-0.40472, -0.22802]`.

Against the matched mean-survival control:

- winner `Delta(E-M)`: **-0.32578**, 95% CI `[-0.40717, -0.24355]`;
- non-winner mean `Delta(E-M)`: **+0.01712**, 95% CI `[+0.00300, +0.02976]`;
- paired winner-minus-nonwinner contrast: **-0.34290**, 95% CI `[-0.43084, -0.25284]`.

This secondary result suggests that the functional contribution is highly concentrated in the dynamically favored relation rather than being a uniform effect across all eight channels. It motivates, but does not itself prove, a **capacity/resource-allocation interpretation**.

## Recovery note

The original aggregate GitHub Actions classifier job failed because `actions/download-artifact` used `merge-multiple: true`, flattening the 12 artifact directories and overwriting identically named `seed_summary.json` files. All 12 fresh-family training jobs had completed successfully and their individual artifact archives were intact.

The result was recovered by downloading those exact preserved artifacts individually and applying the unchanged frozen classifier logic from commit `77ad4069b090d3fd7d9c9c72edf387442a00ef40`. No model was retrained, no seed output was altered, and no scientific threshold or decision rule was changed.

## Claim boundary

R8-M4 establishes a functional contribution under one preregistered suppression intervention in one symmetric synthetic recurrent architecture. It does **not** establish universal necessity, strong emergence, essential chronology, practical superiority, or that Euclidean survival magnitude is itself the causal mediator.
