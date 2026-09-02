# Observer-R11 — Does Selective Survival Causally Support Downstream Readout?

## Status

Primary seeds: **7, 19, 43**.

Preregistered classification: **Outcome G — shared-cause association**.

R11 tests whether the causal survival manipulation established in R10 also changes the frozen downstream reader's ability to use the corresponding task distinction.

The central claim boundary is strict:

> changing how much of a task-associated distinction survives recurrence is not sufficient evidence that the reader uses that distinction.

## Protocol

The initially preregistered full-pair rotation failed the semantic-preservation envelope on validation before any R11 test endpoint was evaluated. It changed immediate matching-relation linear predictions by roughly 10–13% even at the weakest nonzero strength.

That condition was rejected and preserved as a negative validation result. Per the locked fallback logic, R11 switched to the already validated R10 micro-intervention family with the strength ladder:

`epsilon = [0.0025, 0.005, 0.01, 0.02, 0.04]`

The largest strength satisfying the locked semantic-preservation thresholds in every seed was **epsilon = 0.01**.

At that strength, immediate matching-relation linear prediction changes stayed around 1%, full-task MLP prediction changes stayed below 1%, and immediate reader-margin changes remained small.

## R10 reproduction gate

The R10 orientation→survival effect reproduced on the R11 path across all three seeds.

Validation target-minus-random survival effect:

| Seed | High-reader / suppression | Low-reader / rescue |
|---:|---:|---:|
| 7 | -0.875 | +1.658 |
| 19 | -0.909 | +1.080 |
| 43 | -0.848 | +1.016 |

## Primary rescue result

For validation-selected low-reader relations, favorable orientation increased geometric survival relative to semantic-matched random controls by:

| Seed | Survival change | Matching-minus-off reader effect | 95% CI |
|---:|---:|---:|---:|
| 7 | **+1.708x** | **-0.001740** | [-0.002072, -0.001421] |
| 19 | **+1.071x** | **-0.000558** | [-0.000782, -0.000337] |
| 43 | **+1.038x** | **-0.001216** | [-0.001443, -0.000990] |

Matching reader-space pair separation also decreased in every seed.

Thus increased Euclidean survival did not improve native downstream use.

## Primary suppression result

For validation-selected high-reader relations, contractive orientation reduced geometric survival by:

| Seed | Survival change | Matching-minus-off reader effect | 95% CI |
|---:|---:|---:|---:|
| 7 | **-0.893x** | **+0.000462** | [+0.000182, +0.000743] |
| 19 | **-0.906x** | **+0.000576** | [+0.000212, +0.000940] |
| 43 | **-0.865x** | **+0.000687** | [+0.000383, +0.000998] |

Suppression therefore failed to impair the native reader and again produced a small opposite-sign continuous effect.

## Additional controls

The negative result survives the main preregistered checks:

- no beneficial monotonic survival dose-response;
- no hidden positive effect on the preregistered near-boundary subset;
- 8x8 reader matrix has relation-specific structure, but with the opposite sign from the survival→use hypothesis;
- rescue's negative reader effect is weaker at a later wrong stage, while suppression effects remain tiny;
- independent frozen h12 ridge probes do not materially benefit from rescue;
- fresh ridge and MLP64 readers trained on modified trajectories do not reproducibly exploit the additional survival;
- nearest-native-terminal distance changes are near zero, so the primary failure is not explained by a large target-vs-random off-manifold difference;
- a validation-derived native terminal-template correction does not make rescue useful;
- reader sensitivity calibration shows W_out could respond to terminal changes of this magnitude if they were aligned with reader-sensitive geometry.

## Interpretation

Raw geometric survival is not the reader-usable quantity.

Within the intervention states, Euclidean survival correlates weakly with matching reader-margin change, while terminal alignment and signed reader-projected survival correlate much more strongly.

The narrow supported conclusion is:

> **Geometric survival of a task-associated perturbation can be increased or decreased dramatically while immediate task representation remains closely matched, without producing the corresponding improvement or impairment in the frozen downstream reader. The natural association between relation survival and reader usefulness is therefore not explained by Euclidean survival magnitude alone. Training appears to co-organize survival geometry and reader-compatible terminal format.**

This does not show that selective survival is irrelevant. It shows that **survival magnitude alone is neither established as necessary nor sufficient for the current reader's use** under the tested interventions.

## Files

- `experiments/observer_r11/PREREGISTRATION.json` — locked R11 design
- `experiments/observer_r11/COMPUTATIONAL_AMENDMENT.md` — pre-test rejection of the full-pair intervention and frozen fallback
- `cross_seed_evidence_table.csv` — compact evidence classification
- `required_primary_table.csv` — primary cross-seed intervention outcomes and confidence intervals

The full generated R11 working bundle, raw arrays, checkpoints, figures, dose tables, and per-example outputs are not all checked into this repository. These compact records preserve the main scientific evidence and protocol boundary without implying complete end-to-end reproduction of R8–R11 from main alone.
