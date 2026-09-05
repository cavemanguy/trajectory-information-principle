# R8-M2 Result — Symmetry-Breaking Provenance

## Frozen outcome

**Primary classification: S0 — neither simple source tracks winner.**

R8-M2 tested whether seed-dependent native survival specialization in the symmetric synthetic recurrent system could be explained by two ordinary fixed asymmetry sources:

1. relation-specific initialization bundles;
2. finite-sample data-column identity.

The experiment remained no-perturbation and used eight fresh paired families.

## Training validity

All eight families passed the preregistered training-validity gate.

## Primary source-tracking results

### Initialization bundle

Initialization-bundle tracking was not supported.

- mapped winner matches: **3/8**
- mean aligned-minus-raw rank-correlation effect: **-0.030**
- bootstrap 95% CI: **[-0.360, 0.348]**

The eventual favored relation therefore did not reliably follow its relation-specific initialization bundle.

### Finite-sample data column

Data-column tracking was not supported.

- mapped winner matches: **1/8**
- mean aligned-minus-raw rank-correlation effect: **-0.080**
- bootstrap 95% CI: **[-0.610, 0.381]**

The eventual favored relation therefore did not reliably follow the finite-sample data-column identity.

## Commitment timing

The preregistered commitment descriptor identified **epoch 40** as the onset at which the final specialization became reliably established under the frozen criterion.

This does not mean the system contains no earlier bias. It means the final winner was not considered stably committed by the preregistered rule until epoch 40.

## Interpretation boundary

R8-M2 narrows the explanation of the R8 specialization phenomenon:

- R8-M1 established that full relation-selective native survival depends on encoder–recurrence coadaptation in this tested architecture.
- R8-M2 shows that the identity of the eventual specialist is not simply inherited from the tested relation-specific initialization bundle or the tested finite-sample data-column asymmetry.

The most defensible current description is **training-induced spontaneous dynamical specialization** or **training-induced symmetry breaking through encoder–recurrence coadaptation**.

However, **S0 does not prove strong emergence**, universality, practical advantage, or trajectory information as a new computational principle. Other ordinary optimization-path explanations remain open, including gradient interactions, minibatch-order path dependence, nonlinear parameter coupling, and transient representation geometry during training.

## Next question

The next R8 mechanism question is:

> **What changes around the commitment transition that selects and stabilizes the eventual dynamical specialist?**

The next study should remain native/no-perturbation and focus on the training transition itself rather than adding causal trajectory interventions.

Authoritative source result: `results/r8_m2/aggregate/FINAL_RESULT.md` on branch `r8-m2-results`, commit `20a45dd2bda73f3920740cc57acc6c4a334749c4`.
