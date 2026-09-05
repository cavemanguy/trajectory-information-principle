# Current Claims and Claim Boundaries

**Status: September 2026 reset.**

This file is the concise source of truth for what the project currently does and does not claim. It complements `EVIDENCE_LEDGER.md`, `RESEARCH_PRIORITIES.md`, and the detailed historical records. Older experiments are preserved and remain scientifically relevant, but they must not be paraphrased into stronger claims than their controls support.

## Core research intent

The central question is again the native dynamics:

> **What meaningful computational structure emerges in the evolving internal state itself, and can that structure be understood, measured, and eventually used practically?**

The working hierarchy is:

1. **Native dynamics = phenomenon/computation under study.**
2. **Observer = measurement/diagnostic instrument.**
3. **Perturbation = optional later causal intervention or control/logic input.**

The perturbation program is preserved, but perturbation response is not assumed to be where trajectory information resides.

## What has been established within the tested synthetic systems

The project has established several narrower findings:

- **Observer-relative trajectory accessibility:** R2/R3 showed that information available through evolving geometric/directional features can differ from information available from an endpoint snapshot. Exact chronology was not established, and much of the directional signal can be dominated by the earliest transient.
- **Readout preparation:** R6/R8 showed that recurrence can make task structure less accessible to generic readouts while improving compatibility with its trained downstream reader.
- **Selective empirical preservation:** R9/R10 showed strongly contractive, ill-conditioned recurrence in which different task distinctions survive by very different amounts. R10 causally established that orientation relative to learned local dynamical geometry changes distinction survival.
- **Training-induced encoder–recurrence coadaptation:** R8-M1 used matched initialization/data across five no-perturbation training conditions and passed its lineage-validity gate on fresh seeds 11/37/71. Joint training produced substantially more relation-selective native survival than either a frozen recurrent map or frozen encoder in all three seeds, with paired bootstrap intervals above zero. The preregistered outcome was **M3 — encoder–recurrence coadaptation supported**. Terminal h12 supervision alone was not a stable cross-seed explanation.
- **Survival/use dissociation:** R11 showed that increasing or decreasing Euclidean survival magnitude does not by itself cause the corresponding improvement or impairment in downstream reader use.
- **Query-specific perturbation geometry:** ALI-N8-R1 reproducibly showed query-specific direction-dependent responses under its diagnostic decoder class, while also showing that adaptive ALI did not beat direct memory readout and could leak information through the direction itself.
- **Negative causal-control boundaries:** R4C/R4D did not establish useful learned self-steering, R4E failed its Phase-I primary gate, and JTP-1 found no preregistered seed-general instantaneous local-operator marker of trajectory time under its controls.

These are real findings under their stated systems and controls. They are not equivalent to proof of the overarching trajectory-information hypothesis.

## What has not been established

The project has **not** established that:

- trajectory information is a new universal computational principle;
- a recurrent trajectory creates new task information independent of the complete earlier state;
- chronology itself carries the essential information in the tested systems;
- the observed dynamical organization is genuinely emergent in a strong theoretical sense rather than an ordinary consequence of optimization;
- trajectories provide a demonstrated practical advantage over conventional architectures;
- ALI replaces or outperforms attention or direct readout;
- the phenomena generalize to language models, transformers, naturalistic data, or physical systems;
- chaos, strange attractors, or exotic attractor dynamics explain the observed behavior;
- perturbation-response geometry is the location or definition of trajectory information.

The central trajectory-information hypothesis therefore remains **open**, not proven and not cleanly falsified by the existing side-branches.

## ND-R1 and the discovery path to R8-M1

ND-R1 was the first post-reset no-perturbation native-dynamics study. Its frozen primary classification remains:

> **Outcome A — training reproduction failure**

because all three fresh seeds missed the preregistered `h12 >= 0.50` competence gate.

A post-run provenance audit then showed that this competence threshold was miscalibrated above the historical Observer-R2 source lineage itself. The formal Outcome A is preserved and is not retroactively repaired.

Separately, all three ND-R1 fresh seeds showed a large secondary transition from nearly uniform relation survival at initialization to strongly relation-selective survival at epoch 100, with positive bootstrap intervals in every seed and different favored relations across seeds. That pattern was treated as post-primary hypothesis generation rather than a rescued result.

R8-M1 then tested the mechanism independently on new fresh seeds with a lineage-calibrated validity gate. Its formal **M3** result established the narrow coadaptation claim: full selective survival under the tested joint regime depends on both encoder and recurrent-map plasticity. See `R8_M1_RESULT.md`.

## Current scientific priority

Priority 1 remains R8 native specialization, but the question has advanced from whether the specialization appears to **why its symmetry breaks**.

The current preregistered experiment, R8-M2, asks whether the seed-dependent winning relation tracks ordinary asymmetry sources:

- relation-specific initialization bundles;
- finite-sample data-column identity;
- minibatch-order path sensitivity;
- early training commitment and initial shared-gradient asymmetry.

R8-M2 remains no-perturbation. Its primary source-tracking outcomes do not use the word emergence: even failure of the tested simple sources would not by itself prove strong emergence.

After source provenance is understood, the next R8 question is functional necessity: whether the specialization is actually required for task performance or is an optimization side effect.

## Practical-use direction

The most promising practical interpretation currently is **dynamical computation**, not perturbation-based retrieval:

> training can organize an encoder and recurrent map together so that task distinctions experience strongly different native survival through the recurrent dynamics.

Potential engineering branches include selective filtering/compression, sparse channel allocation, compact recurrent preprocessing, early-step computation, reader-specific latent formatting, and later state-dependent control/logic.

These remain engineering hypotheses until separately tested.

## Preservation rule

All previous work remains part of the scientific record. A failed explanation is not the same as a failed phenomenon, and a promising secondary pattern is not promoted into a primary result after the fact.

See:

- `../RESEARCH_PRIORITIES.md`
- `EVIDENCE_LEDGER.md`
- `ND_R1_POSTRUN_AUDIT.md`
- `R8_M1_RESULT.md`
- `R8_NATIVE_DYNAMICS_INVESTIGATION.md`
- `observer_program_r2_r11.md`
- `research_history.md`
