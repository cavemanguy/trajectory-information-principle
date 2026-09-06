# R8-AD6 Result — Tangent vs Transverse Error Dynamics

**Frozen classification:** **P1 — tangent persistence with functional asymmetry supported**

Run: `34015092693`

Authoritative frozen result: [`../results/r8_ad6/aggregate/FINAL_RESULT.md`](../results/r8_ad6/aggregate/FINAL_RESULT.md)

## Question

R8-AD5 produced **K0**: transition-direction rotations were not preferentially damaging; matched magnitude/tangent-like changes were more damaging. AD5 therefore generated, but did not promote, an exploratory explanation: **tangent/phase sensitivity with transverse robustness**.

AD6 tested that explanation directly. From the same frozen native state `h_t`, it applied equal-norm perturbations either along the local native flow tangent or into four orthogonal transverse directions, then tracked the perturbation for four recurrent updates and evaluated the existing frozen terminal task heads.

## Frozen primary result

Both preregistered gates passed.

### Recovery asymmetry

At the frozen primary recovery horizon `k=3`:

- tangent error retention: **1.230823×** initial perturbation norm
- transverse error retention: **0.725633×**
- `D_REC = RET_TAN - RET_TRANS = +0.505190`
- bootstrap 95% CI **[+0.409338, +0.596889]**
- positive in **12/12** lineages

Thus equal-size local tangent errors remained substantially farther from the same-time native trajectory than transverse errors after three recurrent updates.

The effect was stable across every preregistered intervention time and perturbation scale.

### Functional asymmetry

Terminal task accuracy drops were:

- tangent perturbations: **+4.024 pp**
- transverse perturbations: **+2.125 pp**
- `D_FUNC = +1.899 pp`
- bootstrap 95% CI **[+1.292, +2.491] pp**
- positive in **11/12** lineages

The cross-entropy contrast was also positive on average (`+0.053357`) but remained secondary under the preregistration.

## Recovery curve

Mean norm retention relative to the initial equal-size perturbation:

| recurrent steps after perturbation | tangent | transverse | tangent - transverse |
|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 0.000 |
| 1 | 1.069 | 0.646 | +0.423 |
| 2 | 1.187 | 0.686 | +0.501 |
| 3 | 1.231 | 0.726 | +0.505 |
| 4 | 1.280 | 0.751 | +0.529 |

This is not merely a terminal-reader boundary effect. The geometric separation appears immediately in the native recurrent flow: transverse errors are preferentially reduced, whereas tangent errors persist and on average grow relative to their initial norm.

## Phase/progress alignment

The preregistered sign-corrected along-flow diagnostic remained positive:

- `PHASE_RET(k=3) = +0.535540`
- bootstrap 95% CI **[+0.427770, +0.640651]**
- positive in **12/12** lineages

At `k=4`, mean phase/progress retention remained about **+0.524**.

This is compatible with a phase/progress-offset interpretation, but AD6 does not establish an independent hidden phase variable. The complete Markov state still determines the future under the fixed map.

## Relation to AD4 and AD5

The combined result is now more specific than either prior experiment alone:

- **AD4:** native displacement direction is highly useful to a simple diagnostic reader; scalar speed contributes little to that accessibility gain.
- **AD5:** rotating direction is not the greatest causal vulnerability; along-transition magnitude changes are more damaging than matched direction rotations.
- **AD6:** direct local tangent perturbations persist substantially more than true orthogonal perturbations and are more damaging to the terminal task.

Therefore the supported picture is not “direction is the causal code.” A better narrow description is:

> **The learned recurrent flow is strongly anisotropic: task-relevant trajectories are comparatively robust to transverse displacement but sensitive to along-flow progress/tangent displacement. Recent motion direction is highly readable, while along-flow displacement is more causally persistent.**

This is a readability–causality distinction rather than a contradiction.

## What AD6 establishes

Within the tested synthetic autonomous recurrent system, the same learned flow treats equal-size state errors differently according to their orientation relative to the native trajectory. Transverse perturbations are preferentially corrected, while local tangent perturbations persist and produce greater downstream functional damage.

This supplies a preregistered mechanism consistent with the AD5 reversal without changing AD5's frozen K0 outcome.

## Claim boundary

AD6 does **not** establish:

- information beyond the complete Markov state-plus-map;
- a state-independent hidden phase variable;
- essential chronology;
- that every AD2 regime is formally a phase oscillator;
- a universal trajectory code;
- generalization to transformers, biological systems, physical systems, or natural tasks;
- novelty relative to all prior literature;
- practical superiority over conventional architectures.

## Next justified question

The clean next question is whether the along-flow error is genuinely best described as a **progress/phase displacement on a learned low-dimensional trajectory/manifold**, rather than merely a locally high-gain Jacobian direction.

A future confirmatory test should compare tangent persistence against local Jacobian singular vectors, estimate rejoining to the native trajectory/manifold, and test whether a phase/progress coordinate predicts functional consequence better than Euclidean error alone. This should be preregistered separately and must not retroactively change AD6.
