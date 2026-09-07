# R9 Current Synthesis — Temporal, Hybrid, and Swarm Experiments

Status: research synthesis, 2026-09-07. This document summarizes frozen records; it is not a new experiment, replication, or amendment. R8-AD6 remains the strongest established autonomous-flow result. R9 tests different, continuously driven organisms and does not retroactively change R8 classifications. Performance is an interpretation floor, not the project's primary objective.

## Source of truth and provenance

| Experiment | Frozen source / provenance | Authoritative result |
|---|---|---|
| R8-AD6 | Main through R8-AD6 | [R8 synthesis](R8_CURRENT_SYNTHESIS.md), [AD6 result](R8_AD6_RESULT.md) |
| R9-T1 | Preregistration `0923d3f368fb89f1ba661d5db8c1901c4223d246`; implementation `254379aa3d258065f9606201e934218d07687e33`; Actions `34019066391` | [Frozen aggregate](https://github.com/cavemanguy/trajectory-information-principle/blob/r9-t1-results/results/r9_t1/aggregate/FINAL_RESULT.md) |
| R9-H1 | Preregistration `c38b254a2cd7449db47c2a48bbea0da10aa82695`; implementation `b8147b6b1c51f0eb4b475ebc7fd07d1c0abd0a39`; Actions `34020620857` | [Frozen aggregate](https://github.com/cavemanguy/trajectory-information-principle/blob/r9-h1-results/results/r9_h1/aggregate/FINAL_RESULT.md) and JSON alongside it |
| R9-S1 | Preregistration `ca4da167c3be7f7b4ff98e11bf969ff41041eb94`; implementation `fdf4aa8395e803f3a9c410c032090f4ccc40410b`; Actions `34083819542`, attempt 1 | [Immutable run-specific aggregate](https://github.com/cavemanguy/trajectory-information-principle/blob/r9-s1-results/results/r9_s1/runs/34083819542-1-fdf4aa8395e8/FINAL_RESULT.md), JSON and complete family records alongside it |

The R9-S1 result commit is `92dcc6b` (short SHA). Its aggregate Actions artifact is `10004965073`, SHA256 `885bfd798820c79de8815093f0211903612153a9e98d9e87f00d978ce4a41111`. Original seed artifacts and checkpoints remain in their experiment records. No raw result is replaced by this synthesis. The R9-S1 archive was also inspected to verify the published summary and its source manifest.

## R9-T1 — Continuously driven temporal substrate

Frozen classification: **R9-D — primary GRR fails temporal competence**. Secondary sandwich: **S− — return disadvantage**. The temporal-competence gate failed; history transfer passed; anisotropy transfer failed. Eight fresh families completed.

The task supplies episode-specific regime keys, switching, distractors, and later reactivation by regime ID without repeating the key. On DATA events the target is `(x + k_r) mod 16`, with chance 6.25%. The primary GRR was a gated/residual recurrent cell, not the autonomous R8 MLP recurrence.

| Arm | DATA | Return-early |
|---|---:|---:|
| GRR | 20.97% | 14.33% |
| GRU | 63.95% | 48.52% |
| Causal Transformer | 100.00% | 100.00% |
| RNN–Transformer–RNN sandwich | 17.27% | 16.64% |

The GRR history gain was +2.41 pp, CI [+2.17,+2.66], positive 8/8; its gain against shuffled displacement was +2.71 pp. The tangent-minus-transverse retention contrast reversed sign: −0.1174, CI [−0.1915,−0.0496], positive only 1/8. Functional damage difference was approximately −0.0005 in accuracy units. Sandwich-minus-Transformer return accuracy was −83.36 pp, CI [−93.65,−63.14]. These are frozen negative and mixed results, not evidence that recurrence or all hybrids are inherently inferior. R8's tangent persistence did not transfer under this protocol.

## R9-H1 — Hybrid interaction anomaly screen

All eight families and the frozen classifier completed. The validity floor was mean DATA >=20% and return-early >=15%. It is not a mastery or ranking criterion. Eleven arms included plain controls, both serial orders, parallel fusion, perturbation teaching, learned interrogation, recurrent attention control, low-rank modulation, mutual teaching, and update-gated learning.

| Arm | DATA | Return-early | Validity |
|---|---:|---:|---|
| Plain RNN | 6.34% | 6.18% | V− |
| Plain Transformer | 97.18% | 94.72% | V+ |
| RNN→Transformer→RNN | 6.21% | 6.30% | V− |
| Transformer→RNN→Transformer | 58.65% | 38.29% | V+ |
| Parallel | 85.90% | 74.24% | V+ |
| Perturbation teacher | 6.13% | 6.03% | V− |
| Interrogator | 28.46% | 23.70% | V+ |
| Attention controller | 81.86% | 66.24% | V+ |
| Low-rank modulation | 77.03% | 60.27% | V+ |
| Mutual teaching | 78.97% | 62.49% | V+ |
| Update gate | 79.68% | 64.06% | V+ |

Frozen controller classifications: attention controller A+, interrogator A+, low-rank modulation A+, teacher A0, teacher persistence A0. The attention controller had hidden-key H1 gains of +2.65 pp in the RNN stream and +7.76 pp in the Transformer stream, both positive 8/8. The interrogator's native return accuracy exceeded shuffled, random, wrong-context, and zero-controller conditions by approximately 17.2–17.4 pp, all positive 8/8. Its response-key and direction-selectivity screens were A+.

**Critical answer-channel limitation:** the interrogator controller signal alone decoded the answer at 27.17%, close to the full arm's 28.46%. The frozen AX audit did not flag this, but that is not a clean exclusion of answer transmission. A later audit must not relabel the frozen result; a fresh experiment must isolate response-derived information beyond the controller's payload and state access. The attention controller signal was near chance in the same audit, making it a distinct candidate rather than proof of a common mechanism.

Serial order was asymmetric: ORDER_H = −3.42 pp, CI [−4.31,−2.19]; ORDER_X = +25.39 pp, CI [+20.63,+30.04]. These are R-T-R minus T-R-T contrasts on their specified diagnostic boundaries, not universal architecture comparisons. Both parallel branches contributed positively under return ablation, but the predicted RNN-versus-Transformer history specialization was A0. Teacher removal did not establish autonomous learning. The exploratory LEARN_BETA was −0.7286, CI [−0.7540,−0.6972]; it is not proof of beneficial learning-trajectory control.

## R9-S1 — Recurrent swarm dynamics

Eight fresh families and frozen classification completed. Eight 8-dimensional recurrent cells formed shared-weight disconnected, ring, random sparse, fully connected, and learned-routing swarms, plus heterogeneous ring/full variants and a 64-dimensional monolithic control. Fixed input ports distribute event types among cells. Shared graph arms report equal declared parameter counts, but disconnected active-parameter count differs; heterogeneous and monolithic capacities also differ. The exact R9-T1 generator was reused, with no architecture-specific curriculum or auxiliary dynamics loss.

| Arm | DATA | Return-early | Validity | Joint history | Complementarity | Communication |
|---|---:|---:|---|---|---|---|
| Disconnected | 6.25% | 6.16% | V− | LC-A+ | A0 | A0 |
| Ring | 6.95% | 6.63% | V− | LC-A+ | LC-A+ | A0 |
| Random sparse | 6.25% | 6.24% | V− | LC-A+ | LC-A+ | A0 |
| Full | 6.24% | 6.18% | V− | LC-A+ | LC-A+ | A0 |
| Learned routing | 6.26% | 6.20% | V− | LC-A+ | A0 | A0 |
| Heterogeneous ring | 16.66% | 13.51% | V− | LC-A+ | LC-A+ | LC-A+ |
| Heterogeneous full | 17.70% | 13.19% | V− | LC-A+ | LC-A+ | A0 |
| Monolithic | 51.40% | 40.02% | V+ | A+ | N/A | N/A |

LC denotes the original LOW_COMPETENCE qualification, not a new classifier label. Every swarm failed the interpretation floor. The heterogeneous ring's joint hidden-key H1 gain was +3.76 pp, CI [+2.75,+4.92], positive 8/8; its gain over shuffled displacement was +2.66 pp. Joint accessibility and complementarity are not information-theoretic synergy or evidence that communication created information.

The heterogeneous ring was the only communication A+ candidate: native-minus-zero return accuracy +6.77 pp, CI [+1.18,+13.43], positive 7/8; native-minus-shuffled +7.14 pp; native-minus-rewired +6.88 pp. Effects were uneven and concentrated in better-trained seeds. Independent disconnected training and off-manifold ablations do not isolate a unique architectural necessity. All other swarm communication gates were A0. The matched-history observations did not establish a reproducible functionally consequential hysteresis mechanism, collective wave, synchronization regime, or essential chronological code. The primary failure to learn remains central to interpretation.

## What survives and what does not

R8's native-flow anisotropy remains established only within its tested organism. R9 demonstrates that temporal architecture, interaction order, and some controller channels can produce different accessibility and functional effects, but not a universal trajectory mechanism. The positive H1 and S1 labels are anomaly-screen candidates requiring fresh confirmation, not independent discoveries. The failed teacher and shared-weight swarm results remain negative constraints. No result establishes hidden information beyond complete state plus transition map, general semantic recovery, universal phase encoding, a practical hybrid advantage, or novelty of the component architectures.

The next engineering question is whether a separate observer and controlled perturbation instrument can recover identifiable computations from a frozen black box. This is a new application branch, not a redefinition of trajectory information or a rescue of R9-H1/S1. See [Active Flow Observer preparation](ACTIVE_FLOW_OBSERVER.md) and its [literature review](ACTIVE_FLOW_OBSERVER_LITERATURE.md).