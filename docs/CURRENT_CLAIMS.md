# Current Claims and Claim Boundaries

**Status: September 2026 reset.**

This file is the concise source of truth for what the project currently does and does not claim. It complements `EVIDENCE_LEDGER.md`, `RESEARCH_PRIORITIES.md`, and the detailed historical records. Older experiments are preserved and remain scientifically relevant, but they must not be paraphrased into stronger claims than their controls support.

## Core research intent

The central question is the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, and eventually used practically?**

The working hierarchy is:

1. **Native dynamics = phenomenon/computation under study.**
2. **Observer = measurement/diagnostic instrument.**
3. **Perturbation = optional later causal intervention or control/logic input.**

The perturbation program is preserved, but perturbation response is not assumed to be where trajectory information resides.

## What has been established within the tested synthetic systems

- **Observer-relative trajectory accessibility:** R2/R3 showed that information available through evolving geometric/directional features can differ from information available from an endpoint snapshot. Exact chronology was not established, and much of the directional signal can be dominated by the earliest transient.
- **Readout preparation:** R6/R8 showed that recurrence can make task structure less accessible to generic readouts while improving compatibility with its trained downstream reader.
- **Selective empirical preservation:** R9/R10 showed strongly contractive, ill-conditioned recurrence in which different task distinctions survive by very different amounts. R10 causally established that orientation relative to learned local dynamical geometry changes distinction survival.
- **Training-induced encoder–recurrence coadaptation:** R8-M1 passed its fresh-seed validity gate and produced **M3 — encoder–recurrence coadaptation supported**. Full selective survival was substantially stronger than when either the recurrent map or encoder was frozen. Terminal h12 supervision alone was not a stable cross-seed explanation.
- **Symmetry-breaking source narrowing:** R8-M2 produced **S0 — neither simple source tracks winner**. Neither relation-specific initialization bundles nor finite-sample data-column identity reliably tracked winner identity across eight fresh paired families. Stable commitment under its frozen descriptor appeared at epoch 40.
- **Optimization-path sensitivity:** R8-M3 produced **T0 — neither preregistered epoch-20 precommitment predictor supported**. Coadaptation synergy and shared-gradient alignment each carried some rank information but failed the frozen exact-winner criteria. Changing only deterministic minibatch order shifted commitment timing and often changed winner identity under identical initialization/data.
- **Functional contribution of native specialization:** R8-M4 produced **F3 — selective-specialization contribution supported**. Selectivity equalization reduced `G` in 12/12 fresh seeds, by about 66.6% on average. Epoch-100 h12 test accuracy fell by 3.04 percentage points versus baseline and by 2.57 points versus the matched mean-survival control, with paired bootstrap intervals entirely below zero.
- **Specialist-concentrated functional loss:** the preregistered R8-M4 secondary analysis showed that baseline survival-winning relations averaged 74.34% h12 test accuracy versus 13.93% across the other seven relations. Equalization reduced winner accuracy by about 30.36 points while non-winner accuracy was approximately preserved.
- **Simple tight-bottleneck allocation account rejected:** R8-M5 produced **C0 — simple capacity-allocation account not supported**. Widening the state from 16D to 32D improved mean h12 test accuracy by 5.55 percentage points, but increased survival-winner performance concentration by 0.257 and dynamical selectivity `G` by 0.098, with all three paired bootstrap intervals excluding zero in their observed directions. A near-parameter-matched 16D-state control differed from S32 in total parameter count by only about 2.31%, yet S32 still showed greater functional concentration and greater `G`. The simple explanation that specialization is merely a compensatory response to insufficient 16D state capacity is therefore not supported.
- **Survival/use dissociation:** R11 showed that increasing or decreasing Euclidean survival magnitude does not by itself cause the corresponding improvement or impairment in downstream reader use.
- **Query-specific perturbation geometry:** ALI-N8-R1 reproducibly showed query-specific direction-dependent responses under its diagnostic decoder class, while adaptive ALI did not beat direct memory readout and could leak information through the direction itself.
- **Negative causal-control boundaries:** R4C/R4D did not establish useful learned self-steering, R4E failed its Phase-I primary gate, and JTP-1 found no preregistered seed-general instantaneous local-operator marker of trajectory time under its controls.

These are real findings under their stated systems and controls. They are not equivalent to proof of the overarching trajectory-information hypothesis.

## What has not been established

The project has **not** established that:

- trajectory information is a new universal computational principle;
- a recurrent trajectory creates new task information independent of the complete earlier state;
- chronology itself carries the essential information in the tested systems;
- the observed dynamical organization is genuinely emergent in a strong theoretical sense rather than an ordinary consequence of optimization;
- relation-selective specialization is universally necessary or beneficial;
- the R8-M4 effect is mediated by Euclidean survival magnitude itself;
- specialization is caused by a tight latent/recurrent capacity bottleneck;
- the R8-M5 state-width effect is caused specifically by recurrent workspace dimension rather than the much stronger encoder-side representation produced by the wider model;
- trajectories provide a demonstrated practical advantage over conventional architectures;
- ALI replaces or outperforms attention or direct readout;
- the phenomena generalize to language models, transformers, naturalistic data, or physical systems;
- chaos, strange attractors, or exotic attractor dynamics explain the observed behavior;
- perturbation-response geometry is the location or definition of trajectory information.

The central trajectory-information hypothesis therefore remains **open**, not proven and not cleanly falsified by the existing side-branches.

## ND-R1 through R8-M5 discovery sequence

ND-R1 remains formally **Outcome A — training reproduction failure** because all three fresh seeds missed its preregistered h12 competence gate. A post-run provenance audit showed that the threshold was miscalibrated above the historical Observer-R2 lineage itself; the frozen outcome is preserved and not repaired.

Separately, ND-R1 generated the post-primary selective-survival observation: training transformed nearly uniform relation survival into strong seed-dependent relation selectivity.

R8-M1 independently established the narrow coadaptation mechanism: both encoder and recurrent-map plasticity are needed for the full learned selective-survival pattern.

R8-M2 ruled out two simple fixed explanations of winner identity: relation-specific initialization bundles and finite-sample data-column identity.

R8-M3 failed to identify the eventual winner with two simple preregistered epoch-20 predictors, while independently showing material optimization-path sensitivity under changed minibatch order.

R8-M4 moved from formation to function. Its **F3** result showed that specifically suppressing relation-selective native survival impairs terminal task performance beyond a matched mean-survival auxiliary control, with most of the impairment localized to the dynamically favored relation.

R8-M5 then tested the specifically motivated tight-capacity/resource-allocation account. Its frozen **C0** result rejected the predicted pattern: a 32D state improved h12 performance but strengthened, rather than weakened, both dynamical selectivity and survival-winner functional concentration. Because the full-width model also improved h0 accuracy by about 34.65 percentage points, R8-M5 does not isolate whether recurrent workspace dimension itself drives the stronger specialization.

The most defensible compact description is now:

> **training-induced, optimization-path-sensitive dynamical specialization produced through encoder–recurrence coadaptation, with a demonstrated functional contribution under controlled suppression; widening the full latent/state representation improves performance while strengthening rather than relieving specialization in the tested synthetic architecture.**

This remains a local mechanistic result, not proof of strong emergence or a universal principle.

## Current scientific priority

Priority 1 remains R8 native specialization for one final sharply isolated follow-up motivated by the R8-M5 reversal.

The next question is:

> **If the encoder output remains 16D, does a function-preserving expansion of only the autonomous recurrent workspace strengthen useful specialization and terminal performance?**

The planned R8-M6 design should begin from one common 16D model before stable commitment, then fork into:

- ordinary 16D continuation;
- a function-preserving 16D-to-32D recurrent-state expansion whose extra coordinates begin at zero and initially reproduce the original trajectory exactly;
- a near-parameter-matched wider-transition control that retains the 16D recurrent state and is also initialized to reproduce the original function exactly.

This directly separates post-encoding recurrent workspace dimension from the large h0 representation change that confounded the interpretation of R8-M5. A positive result would still be local to this architecture and would not imply a universal routing principle.

After this isolated follow-up, the planned trunk returns to R6/R8 readout preparation unless R8-M6 produces another uniquely justified result that changes that priority.

## Practical-use direction

The most promising practical interpretation currently is **dynamical computation**, not perturbation-based retrieval:

> training can organize an encoder and recurrent map together so that task distinctions experience strongly different native survival through recurrent dynamics.

R8-M4 shows that this organization can contribute functionally. R8-M5 shows that stronger specialization can coexist with better performance when state width increases, arguing against a simple scarce-capacity compromise. Possible engineering branches include learned dynamical routing, selective memory/compression, sparse channel allocation, compact recurrent preprocessing, early-step computation, and reader-specific latent formatting.

These remain engineering hypotheses until separately tested.

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
- `R8_NATIVE_DYNAMICS_INVESTIGATION.md`
- `observer_program_r2_r11.md`
- `research_history.md`
