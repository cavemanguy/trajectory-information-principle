# AR6 — Operator-Level Causal Control of Representation Reformatting

## Status
COMPLETE under the locked AR6 primary intervention family. This closes the current attractor-family mechanistic escalation.

## Primary classification

**Outcome C — Generic disruption / no demonstrated causal privilege for the AR4-derived transform mode.**

Secondary findings:

- **Outcome E (restricted): signed raw-coordinate control.** `+K` and `-K` produce nearly opposite immediate coordinate changes, but the preregistered “advance” direction did not advance later-stage reader compatibility.
- **Outcome H — content co-moves.** Canonicalized representation changed almost as much as raw representation, so a clean format/content dissociation was not obtained.
- **Outcome I (local/static equivalence).** At the intervention step, the update-space operator is itself an affine map of the native update, so the immediate coordinate effect has an exact matched static-affine equivalent by construction.
- **Modified-J causal advantage unresolved/non-replicating.** Modified-operator Jacobians predicted the two-step counterfactual response very accurately, but did not reproducibly outperform native-path Jacobians.

**Highest supported success level: Level 3 only in the narrow signed-coordinate sense. Level 4+ is not supported.** The mode-specific stage-format prediction required for a stronger Level-3/4 interpretation failed.

## Locked design

The preregistration was written and hashed before primary AR6 test runs. Five exact frozen AR2 systems were used: seeds 7, 19, 31, 43, 59. Upstream networks were not retrained.

Primary operator intervention used the native update `d = G(s,z)-s` and temporarily replaced it for two recurrent steps with:

`d' = M_y d`, where `M_y = I ± alpha K_y`.

`K_y` was derived from a train-only ridge map between adjacent AR4 directional stages within coarse answer Y, with no fine-condition C used to fit the operator. Candidate stage bins were 3, 4, 5 and alpha was selected from {0.005, 0.01, 0.02, 0.05, 0.10} using validation preservation and geometry-change criteria only.

All five seeds independently selected the middle stage (bin 4) and alpha = 0.10.

## Baseline reproduction

The frozen AR4 artifact retains the established cross-time locality, affine rescue, canonicalization rescue, transform-family structure, and evolving-J > frozen-J native prediction. The frozen AR5 artifact retains the small-state-perturbation result in which local Jacobian transport is nearly exact but native/frozen/shuffled Jacobian predictions are nearly indistinguishable. AR6 used the same frozen checkpoints and conventions.

A full seed-7 AR6 rerun reproduced the primary summary CSV byte-for-byte.

## P1 — Did AR6 actually change operator geometry?

Yes, by a large margin relative to AR5.

Mean validation-selected relative Jacobian change was **10.79%** (SD 1.10 pp). Mean test relative Jacobian change was **10.80%** (SD 1.19 pp).

AR5 changed the realized Jacobian path by only about **0.036%**. AR6 therefore exceeded the preregistered 0.36% geometry-separation gate by roughly thirty-fold and AR5's realized change by roughly three hundred-fold.

This is the clearest AR6 success: the experiment reached a genuinely changed-operator regime while keeping final computation extremely stable.

## Preservation

For the primary `+K` intervention across all five systems:

- coarse Y identity: **100%**
- endpoint-tolerance pass: **100%**
- mean terminal displacement: **6.20e-8** (cross-seed mean)

Thus the temporary operator modification did not produce an output or endpoint failure in the tested sample.

Implementation caveat: calibration rolled modified and native states for 60 post-window steps and directly checked endpoint/Y preservation. The calibration table's `convergence` field was set to 1.0 rather than independently recomputing the original convergence predicate. Because modified endpoints were ~1e-8 from the paired native converged endpoints, this does not change the endpoint result, but the convergence percentage should not be treated as an independently re-measured metric.

## P2/P4 — Did the operator cause signed representational change?

Yes.

Mean immediate normalized-direction displacement from native was:

- `+K`: **0.0814**
- `-K`: **0.0847**

The sign-reversal cosine, comparing `Δu(+K)` with `-Δu(-K)`, was **0.9963** (SD 0.0009) across systems.

So the operator intervention gives highly controlled, approximately antisymmetric coordinate motion rather than arbitrary numerical damage.

However, this is not enough to establish the AR4 stage-format mechanism because the direction of the *reader-compatibility* effect was not the preregistered one.

## P7 — Format advance / retard

The preregistered “advance” intervention was supposed to increase compatibility with the later-stage reader at matched intervention location.

It did not.

Relative to zero intervention, later-stage reader accuracy changed by:

- seed 7: -0.49 pp
- seed 19: -2.34 pp
- seed 31: -2.54 pp
- seed 43: -4.10 pp
- seed 59: -8.01 pp

Mean: **-3.50 pp**, same direction in 5/5 systems.

The matched random operator also reduced later-stage compatibility by **-3.36 pp** on average. Therefore the `+K` result is essentially indistinguishable in sign and magnitude from generic matched operator disruption for this primary readout.

The “retard” intervention reduced later-stage compatibility more strongly, mean **-9.45 pp** in 5/5 systems, but it did **not** reproducibly increase earlier-stage compatibility (mean +0.51 pp; only 3/5 strictly positive, one zero).

Therefore AR6 does not establish controllable advance/retard of representational stage.

## P6 — Modified-J versus native-J counterfactual prediction

Both models predicted the short-horizon counterfactual response very accurately:

- modified-J cosine: **0.99165**
- native-J cosine: **0.99158**
- mean difference: **+0.000071**

Seedwise modified-minus-native differences were approximately:

- +0.00170
- +0.00022
- +0.00004
- -0.00388
- +0.00228

Thus modified-J won in 4/5 systems, but the effect is tiny and heterogeneous, with one material reversal. This does not meet the intended standard for a meaningful modified-geometry causal advantage.

The important contrast with AR5 is that operator geometry was now changed by ~10.8%; nevertheless, simply substituting the modified Jacobian did not produce a robust improvement over the native Jacobian in predicting the tested short-horizon deviation.

## P8 — Format/content dissociation

This failed.

Mean raw coordinate change:

- `+K`: **0.0814**
- `-K`: **0.0847**

Mean change after frozen AR4 canonicalization:

- `+K`: **0.0738**
- `-K`: **0.0779**

Canonical change therefore remained roughly 91–92% as large as raw change. The intervention did not produce the desired regime of large raw-format movement with small canonical-content movement.

This supports Outcome H under the tested canonicalization: content-accessible geometry and raw coordinate format are not cleanly causally separable using this operator family.

## Random-control interpretation

The decisive negative result is not that operator changes fail to affect representation. They clearly do.

The problem is specificity. The AR4-derived `+K` mode did not produce the predicted later-format compatibility and performed similarly to a matched random operator on that key measure. Therefore the experiment cannot assign causal privilege to the AR4-derived stage transform family.

## Static-affine limitation

At the intervention step, `d' = M d` is exactly a static affine/linear transformation of the native update direction. A post-hoc application of the same `M` to that native direction reproduces the immediate representational effect by construction.

AR6 therefore cannot claim that dynamics are uniquely necessary to produce the local coordinate transformation. The scientific question was whether the temporary dynamic implementation would produce the *predicted downstream stage-format effect*. That stronger result did not occur.

## Final answers

1. **AR4/AR5 baseline?** Preserved and used from the frozen artifacts/checkpoints; no upstream retraining.
2. **Operator/Jacobian change?** ~10.8%, dramatically above AR5's ~0.036%.
3. **Safe geometry-change window?** Yes: middle stage, alpha .10, two-step transient intervention.
4. **Later raw representation altered?** Yes, measurably and with strong sign symmetry.
5. **Predicted AR4 format direction?** No. The “advance” intervention reduced later-reader compatibility in 5/5.
6. **Modified-J prediction accurate?** Yes (~0.992 cosine), but native-J was essentially as accurate.
7. **Modified-J > native-J?** Tiny mean advantage, 4/5 direction, not robust enough for the target claim.
8. **Dominant mode > random?** No on the key later-reader compatibility effect; +K and random produced nearly the same mean loss.
9. **Advance/retard achieved?** No. Retard suppresses later-reader compatibility, but advance does not advance it and retard does not reliably enhance earlier-reader compatibility.
10. **Y/endpoint preserved?** Yes, extremely strongly in the tested sample.
11. **Canonical C preserved?** No clean dissociation; canonical coordinates moved almost as much as raw coordinates.
12. **Static equivalent?** Immediate update-coordinate effect has an exact static affine equivalent by construction.
13. **Reproducibility?** Seed 7 rerun reproduced the summary exactly.

## Narrowest defensible conclusion

> **Directly modifying the local update operator by roughly ten percent can causally and directionally alter transient representational coordinates while leaving the coarse output and terminal state essentially unchanged. However, the AR4-derived stage-transform intervention does not produce the preregistered format-advance effect, performs similarly to a matched random operator on later-stage reader compatibility, does not cleanly preserve the canonical representation, and modified-operator Jacobians do not robustly outperform native Jacobians. AR6 therefore establishes operator-level control of transient coordinates, but not the stronger claim that AR4's progress-conditioned affine transformation family is the causal law generating stage-specific representational format.**

## Branch decision

Per the locked AR6 decision rule, **do not create AR7 to force stronger causality**.

AR6 closes the increasingly fine mechanistic attractor-family branch. The next scientifically appropriate move is architecture generalization, carrying forward both positive and negative boundaries:

- AR2: route is readable under endpoint control.
- AR3: order effect is largely progress binding plus temporal reformatting, not irreducible forward chronology.
- AR4: stage reformatting is strongly affine/canonicalizable and observationally associated with evolving Jacobian geometry.
- AR5: Jacobians are a genuine local causal transport law, but state perturbations barely change operator geometry.
- AR6: operator geometry can be changed strongly and transient coordinates can be controlled with output/endpoint preserved, but the specific AR4 transform family fails the stronger causal format-control prediction.

That is a clean stopping point rather than a failed branch.
