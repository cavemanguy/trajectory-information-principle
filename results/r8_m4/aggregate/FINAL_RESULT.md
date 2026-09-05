# R8-M4 Final Result

**Primary classification:** F3 — selective-specialization contribution supported

- Baseline validity: **True**
- Suppression gate: **True**
- E lower G count: **12/12**
- Mean DeltaG(E-B): **-0.506315**; 95% CI **[-0.562320, -0.447217]**
- Mean relative G reduction: **0.666**
- Mean h12 Delta(E-B): **-0.030413**; 95% CI **[-0.041504, -0.018127]**
- Mean h12 Delta(E-M): **-0.025746**; 95% CI **[-0.036619, -0.014523]**
- Mean G(M)-G(E): **0.577185**; 95% CI **[0.497767, 0.646360]**

## Interpretation

The selectivity-equalization intervention successfully suppressed relation-selective native survival in every fresh seed and by about 66.6% on average. Terminal h12 task accuracy was lower under the equalization intervention than under both the ordinary baseline and the matched mean-survival control, with the preregistered paired bootstrap intervals entirely below zero. The matched control retained substantially more selectivity than E.

Under the frozen R8-M4 decision rule, this satisfies **F3**: specifically suppressing relation-selective native survival impaired terminal task performance beyond the matched mean-survival control. This supports a functional contribution of the specialization in the tested synthetic recurrent system.

## Recovery note

The original aggregate GitHub Actions classification job failed before writing a result because `actions/download-artifact` used `merge-multiple: true`, flattening the 12 artifacts and overwriting identically named `seed_summary.json` files. All 12 fresh-family training jobs had completed successfully and their individual artifact archives remained intact.

This result was recovered by downloading those exact 12 preserved artifacts individually and applying the unchanged frozen classifier logic from commit `77ad4069b090d3fd7d9c9c72edf387442a00ef40`. No model was retrained, no seed output was altered, and no scientific threshold or decision rule was changed.

## Claim boundary

R8-M4 tests functional necessity under one preregistered training intervention in one symmetric synthetic recurrent architecture. F3 supports a functional contribution under this intervention, not a universal necessity theorem, proof of strong emergence, or proof that Euclidean survival magnitude itself is the causal mediator.
