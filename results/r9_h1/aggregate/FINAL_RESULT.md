# R9-H1 Final Aggregate — Hybrid Interaction Anomaly Screen

R9-H1 is an anomaly hunt, not a performance ranking. Accuracy is used only as the preregistered interpretation floor.

## Validity

| arm | validity | DATA | return-early |
|---|---:|---:|---:|
| ATTN_CONTROLLER | V+ | 0.8186 | 0.6624 |
| CONTROL_RNN | V- | 0.0634 | 0.0618 |
| CONTROL_T | V+ | 0.9718 | 0.9472 |
| INTERROGATOR | V+ | 0.2846 | 0.2370 |
| LOWRANK_MOD | V+ | 0.7703 | 0.6027 |
| MUTUAL | V+ | 0.7897 | 0.6249 |
| PARALLEL | V+ | 0.8590 | 0.7424 |
| RTR | V- | 0.0621 | 0.0630 |
| TEACHER | V- | 0.0613 | 0.0603 |
| TRT | V+ | 0.5865 | 0.3829 |
| UPDATE_GATE | V+ | 0.7968 | 0.6406 |

## Controller / teaching classifications

- ATTN_CONTROLLER: **A+**
- INTERROGATOR: **A+**
- LOWRANK_MOD: **A+**
- TEACHER: **A0**
- TEACHER persistence: **A0**

## Serial order

- ORDER_H: **A-**, mean -0.0342, 95% CI [-0.0431, -0.0219]
- ORDER_X: **A+**, mean +0.2539, 95% CI [+0.2063, +0.3004]

## Parallel specialization

- PAR_H: **A0**, mean -0.0019, 95% CI [-0.0082, +0.0039]
- RNN branch return contribution: **A+**
- Transformer branch return contribution: **A+**

## Perturbation response

- TEACHER: response-key gain **LOW_COMPETENCE_A+**, direction selectivity **LOW_COMPETENCE_A+**
- INTERROGATOR: response-key gain **A+**, direction selectivity **A+**

## Learning-trajectory coupling

- LEARN_BETA: **A-** (exploratory), mean -0.7286, 95% CI [-0.7540, -0.6972]

## Claim boundary

R9-H1 is an anomaly screen. Results do not establish an independent trajectory-information substance, universal phase code, architecture novelty, or architecture superiority.
