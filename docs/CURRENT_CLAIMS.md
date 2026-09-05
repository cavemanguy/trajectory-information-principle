# Current Claims and Claim Boundaries

**Status: September 2026 reset, updated through R8-M6 and R8-M5R.**

This file is the concise source of truth for what the project currently does and does not claim. It complements `EVIDENCE_LEDGER.md`, `RESEARCH_PRIORITIES.md`, and the detailed historical records. Older experiments are preserved and remain scientifically relevant, but they must not be paraphrased into stronger claims than their controls support.

## Core research intent

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, and eventually used practically?**

Working hierarchy:

1. **Native dynamics = phenomenon/computation under study.**
2. **Observer = measurement/diagnostic instrument.**
3. **Perturbation = optional causal intervention or control/logic input.**

The perturbation program is preserved, but perturbation response is not assumed to be where trajectory information resides.

## What has been established within the tested synthetic systems

- **Observer-relative trajectory accessibility:** R2/R3 showed that information available through evolving geometric/directional features can differ from information available from an endpoint snapshot. Exact chronology was not established, and much of the directional signal can be dominated by the earliest transient.
- **Readout preparation:** R6/R8 showed that recurrence can make task structure less accessible to generic readouts while improving compatibility with its trained downstream reader.
- **Selective empirical preservation:** R9/R10 showed strongly contractive, ill-conditioned recurrence in which different task distinctions survive by very different amounts. R10 causally established that orientation relative to learned local dynamical geometry changes distinction survival.
- **Training-induced encoder–recurrence coadaptation:** R8-M1 produced **M3 — encoder–recurrence coadaptation supported**. Full selective survival was substantially stronger than when either the recurrent map or encoder was frozen.
- **Symmetry-breaking source narrowing:** R8-M2 produced **S0 — neither simple source tracks winner**. Neither relation-specific initialization bundles nor finite-sample data-column identity reliably tracked winner identity.
- **Optimization-path sensitivity:** R8-M3 produced **T0 — neither preregistered epoch-20 predictor supported**. Changing only deterministic minibatch order shifted commitment timing and changed final winner identity in 5/12 paired families under identical initialization/data.
- **Functional contribution of native specialization:** R8-M4 produced **F3 — selective-specialization contribution supported**. Selectivity equalization reduced `G` in 12/12 fresh seeds by about 66.6% on average. Epoch-100 h12 test accuracy fell by 3.04 percentage points versus baseline and by 2.57 points versus the matched mean-survival control.
- **Specialist-concentrated functional loss:** the preregistered R8-M4 secondary analysis showed baseline survival-winning relations averaged 74.34% h12 test accuracy versus 13.93% across the other seven relations. Equalization reduced winner accuracy by about 30.36 points while non-winner accuracy was approximately preserved.
- **Simple tight-bottleneck allocation account rejected:** R8-M5 remains **C0 — simple capacity-allocation account not supported**. Widening full state from 16D to 32D improved h12 but did not reduce specialization; in that sample both `G` and specialist dominance increased. This rejects the specific prediction that wider state should relieve specialization.
- **Isolated recurrent-workspace account not supported:** R8-M6 remains **W0**. A function-preserving post-encoding expansion from 16D to 32D improved h12 versus both B16 and the near-parameter-matched P16 control, but h0 equivalence failed after continued training. Therefore the benefit cannot be attributed cleanly to recurrent workspace dimension alone under the frozen rule.
- **Fresh state-dimension-specific terminal benefit:** R8-M5R produced **R2 — state-dimension-specific terminal benefit supported** on 12 new families using the exact R8-M5 training manipulation but removing any directional success requirement on specialization. S32 improved h12 by 4.94 percentage points versus B16, 95% CI [+2.92,+6.53] pp, and by 3.00 points versus P16, CI [+0.60,+5.20] pp. Relative to B16, changes in `G` and `D` were mixed/indeterminate. Thus wider state can improve terminal performance without requiring specialization to increase or decrease in a fixed direction.
- **Survival/use dissociation:** R11 showed that increasing or decreasing Euclidean survival magnitude does not by itself cause the corresponding improvement or impairment in downstream reader use.
- **Query-specific perturbation geometry:** ALI-N8-R1 reproducibly showed query-specific direction-dependent responses under diagnostic decoders, while adaptive ALI did not beat direct memory readout and could leak information through the direction itself.
- **Negative causal-control boundaries:** R4C/R4D did not establish useful learned self-steering, R4E failed its Phase-I primary gate, and JTP-1 found no preregistered seed-general instantaneous local-operator marker of trajectory time under its controls.

These are real findings under their stated systems and controls. They are not equivalent to proof of the overarching trajectory-information hypothesis.

## What has not been established

The project has **not** established that:

- trajectory information is a new universal computational principle;
- a recurrent trajectory creates new task information independent of the complete earlier state;
- chronology itself carries the essential information in the tested systems;
- the observed organization is genuinely emergent in a strong theoretical sense rather than an ordinary consequence of optimization;
- relation-selective specialization is universally necessary or beneficial;
- the system explicitly or consciously "chooses" specialization;
- specialization strength is a simple scalar proxy for performance;
- specialization is caused by a tight latent/recurrent capacity bottleneck;
- the R8-M5R state-width benefit is caused specifically by one internal mechanism;
- the R8-M6 terminal benefit can be attributed cleanly to isolated recurrent workspace dimension;
- Euclidean survival magnitude itself mediates reader usefulness;
- trajectories provide a demonstrated practical advantage over conventional architectures;
- ALI replaces or outperforms attention or direct readout;
- the phenomena generalize to language models, transformers, naturalistic data, or physical systems;
- chaos, strange attractors, or exotic attractor dynamics explain the observed behavior;
- perturbation-response geometry is the location or definition of trajectory information.

The central trajectory-information hypothesis remains **open**, not proven and not cleanly falsified by the existing side-branches.

## R8 native-dynamics discovery sequence

ND-R1 remains formally **Outcome A — training reproduction failure** because all three fresh seeds missed its preregistered h12 competence gate. A post-run provenance audit showed that the threshold was miscalibrated above the historical Observer-R2 lineage itself; the frozen outcome is preserved and not repaired. Its post-primary selective-survival pattern generated the R8-M series.

R8-M1 established the narrow coadaptation mechanism: both encoder and recurrent-map plasticity are needed for the full learned selective-survival pattern.

R8-M2 ruled out two simple fixed explanations of winner identity: relation-specific initialization bundles and finite-sample data-column identity.

R8-M3 failed to identify the eventual winner with two simple preregistered epoch-20 predictors, while independently showing material optimization-path sensitivity under changed minibatch order.

R8-M4 moved from formation to function. Its **F3** result showed that specifically suppressing established relation-selective native survival impairs terminal task performance beyond a matched mean-survival auxiliary control, with most impairment localized to the dynamically favored relation.

R8-M5 tested the simple scarce-state-capacity interpretation and rejected its directional prediction. Wider state improved performance but did not cause specialization to relax.

R8-M6 attempted to isolate post-encoding recurrent workspace by a function-preserving fork. Terminal performance improved under X32, but continued training changed h0 enough that the preregistered isolated-workspace claim failed.

R8-M5R then repeated the full R8-M5 capacity manipulation on 12 new families without defining either more or less specialization as success. Its **R2** result established a state-dimension-specific terminal benefit while specialization response remained free to vary across optimization paths.

The most defensible compact description is now:

> **Training induces optimization-path-sensitive dynamical specialization through encoder–recurrence coadaptation. Established specialization can contribute functionally, but its strength is not a simple proxy for performance. Wider state improves terminal performance in fresh paired experiments even when specialization is allowed to vary rather than being forced toward a predetermined direction.**

This remains a local mechanistic result, not proof of strong emergence or a universal principle.

## Current scientific priority

The capacity-bottleneck branch is now sufficiently constrained. The next justified R8 test is not another width sweep. It is a harder test of **adaptive dynamical reorganization under controlled changing demand**:

> **With architecture, initialization lineage, and data generator fixed, does a controlled change in functional demand reliably reorganize native specialization toward the demanded relation, and does reversing the demand reverse or reassign that organization?**

The strongest version should use a reversible A→B→A demand schedule, preregister demand-tracking and reversibility criteria, include matched controls for mere extra training and loss weighting, and test whether blocking the demand-aligned reorganization selectively harms the phase in which it is useful.

This would test adaptive regime switching, not merely seed-to-seed heterogeneity. It must be frozen before fresh outcomes are inspected.

## Practical-use direction

The most promising practical interpretation remains **dynamical computation**, not perturbation-based retrieval:

> training can organize an encoder and recurrent map together so that task distinctions experience strongly different native survival through recurrent dynamics, and the amount/pattern of specialization need not be fixed across successful optimization paths.

Possible engineering branches include learned dynamical routing, selective memory/compression, sparse channel allocation, compact recurrent preprocessing, early-step computation, and reader-specific latent formatting. These remain engineering hypotheses until separately tested.

## Preservation rule

All previous work remains part of the scientific record. A failed explanation is not the same as a failed phenomenon, and a promising secondary pattern is not promoted into a primary result after the fact.

See:

- `../RESEARCH_PRIORITIES.md`
- `EVIDENCE_LEDGER.md`
- `ND_R1_POSTRUN_AUDIT.md`
- `R8_M1_RESULT.md`
- `R8_M2_RESULT.md`
- `R8_M3_RESULT.md`
- `R8_M4_RESULT.md`
- `R8_M5_RESULT.md`
- `R8_M6_RESULT.md`
- `R8_M5R_RESULT.md`
- `R8_NATIVE_DYNAMICS_INVESTIGATION.md`
- `observer_program_r2_r11.md`
- `research_history.md`
