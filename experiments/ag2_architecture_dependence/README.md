# AG2 — Architecture Dependence of Progress-Conditioned Representational Locality

AG2 follows AG1 and asks why progress-conditioned locality generalized across recurrent architectures while strong cross-stage affine alignability did not.

## Status

Completed comparative/descriptive experiment. AR1–AR6 remain closed. No AR7 and no causal state/gate/operator intervention was run.

**Highest supported level: Level 2 — Cross-task discriminator.**

**Primary classification: Outcome J — Task dependence dominates the alignability contrast, with a reproducible cross-task architecture difference in correspondence transport.**

## Main result

Adjacent-stage transport was highly predictable by simple affine maps in the attractor, GRU, and leaky-RNN systems even when AG1 far-stage affine rescue was near zero. Task-1 mean adjacent direction prediction quality (1-NMSE) was approximately 0.991 attractor, 0.981 GRU, 0.998 leaky RNN, and 0.857 vanilla RNN.

The strongest cross-task architecture discriminator was individual example correspondence retrieval after a global adjacent-stage affine map. Mean retrieval was:

| Architecture | Task 1 | Task 2 |
|---|---:|---:|
| Attractor reference | 85.0% | 70.5% |
| Leaky RNN | 75.3% | 68.2% |
| GRU | 61.4% | 41.5% |
| Vanilla RNN | 49.4% | 40.6% |

However, this did not become a task-independent explanation of affine rescue. Under the common AG1 protocol the attractor reference itself fell from ~40.9% rescue on Task 1 to ~2.6% on Task 2. A compact three-predictor model fit on Task 1 failed to generalize to Task-2 rescue.

## Interpretation

AG2 weakens the simplest successive-re-expression story. GRU and especially leaky RNN can preserve substantial local affine predictability, neighborhood structure, CKA similarity, and correspondence while still having weak distant-stage reader compatibility. The narrowest supported picture is therefore locally coherent representational transport whose long-range functional alignability is strongly task-dependent.

No tested Jacobian, gating, recurrent/input-drive, contraction, rank, neighborhood, or correspondence metric earned status as a task-independent causal or predictive law for affine rescue. All architecture-property relationships remain observational.

## Claim boundary

Do not claim information destruction/reconstruction, Jacobian or gate causality, universal coherent transport, or that correspondence determines affine rescue. AG2 contains no causal architectural intervention.

See `REPORT_FINAL.md` for the full 60-question report and `PREREGISTRATION.json` plus computational amendments for protocol details.
