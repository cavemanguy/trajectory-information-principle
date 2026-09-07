# Active Flow Observer — Preparation and Experimental Design

Status: proposed engineering/research branch, 2026-09-07. No AFO training, semantic recovery result, or live black-box experiment has been run. This is not a rescue of R9-T1, H1, or S1 and does not change their frozen gates. It preserves the user's goal of an RNN continuously attached to state-space flow, an automated perturber, and a Transformer running alongside the system to produce a human-readable account of computation.

## Goal and falsifiable question

Can a causal observer of a frozen system's native state evolution, augmented by controlled responses to selected interventions, recover task-relevant and independently verifiable operations that cannot be recovered as well from simpler observational or information-matched baselines?

The target is a useful, calibrated semantic trace, not a claim to read thoughts, hidden consciousness, or an independent phase substance. A fluent explanation is not evidence that the described computation happened. Exact recovery is only a meaningful claim for operations with independently defined ground truth and an observation channel sufficient to distinguish them. When multiple mechanisms have identical accessible behavior, the observer must express uncertainty or abstain. The broad trajectory-information hypothesis remains open.

## Architecture and information flow

```
frozen, instrumented black box B
  input x_t -> native state z_t -> transition F_B -> output y_t
                     | snapshots, events, timestamps (read-only)
                     v
             causal observer RNN O
                     | compressed streaming history
                     +------------------------------+
                     |                              |
              perturbation policy P                 |
                     | bounded action p_t           |
                     v                              v
       isolated clone/replay of B             causal Transformer T
          z_t + p_t -> F_B                  native + response evidence
                     |                              |
            response differences -------------------+
                                                    |
                                      structured semantic event distribution
                                                    |
                                      verified/qualified text renderer
```

The native black box is trained first and frozen. Neither observer nor perturber updates its weights, optimizer state, mutable cache, or inputs on the primary native path. Interventions act on cloned states or a reproducible fork; the unperturbed continuation is retained as a paired control. The monitoring RNN reads the evolving state but is not assumed to be part of the phenomenon. The Transformer is an observer/semantic hypothesis generator, not a source of privileged truth. 'In tangent' here means a parallel observation process; mathematical tangent directions are separately defined and tested, never assumed.

For a discrete driven system, z_(t+1)=F_B(z_t,x_t). The observer state is o_(t+1)=G(o_t,c_t), where c_t is an explicitly permitted native observation. At a chosen checkpoint, P selects a bounded p_t using only its allowed history. A one-step response may be r_t=F_B(z_t+p_t,x_t)-F_B(z_t,x_t); longer paired continuations use identical future inputs. Symmetric finite differences are separate diagnostics, not automatically causal semantic explanations. The complete Markov state includes all mutable state necessary for continuation, not just an arbitrary hidden tensor. For Transformers, layer-depth trajectories and token-generation time are different axes and must be logged separately.

## Initial organism and ground truth

Begin with an owned, locally instrumented synthetic model/program whose operations are independently logged. Use a deterministic register-machine or compositional-rule task with a small operation vocabulary, explicit control flow, and semantically equivalent implementations. Operations may include selecting a register/key, retrieving, comparing, adding modulo a fixed value, conditional routing, updating memory, and emitting a result. The exact vocabulary, grammar, train/test generator, state visibility, and seeds must be frozen in a new preregistration before outcomes. The reference interpreter provides operation intervals, arguments where identifiable, and state transitions. These labels are never passed into the black-box forward inputs or observer at inference.

A trained recurrent black box is the first neural target. If the existing R9-T1 or other frozen models are reused, use a genuinely competent checkpoint under a predeclared rule; do not select an attractive seed or adapt a failed run after seeing its outcomes. A new source-model training experiment is permitted under its own preregistration. The R9-T1 task may provide a secondary known-rule test, but output correctness alone cannot establish internal implementation of a particular algorithm. A reference interpreter's semantic trace is task ground truth, not automatically the ground truth of a neural network's internal algorithm.

A later frozen Transformer target can expose residual streams and selected layer outputs through local inference hooks. A local model with accessible weights and controlled replay is preferable to an API that only supplies text. If the state/transition interface is unavailable, report that limitation rather than inventing a state-space trajectory from outputs. The initial adapter is not a general-purpose live probe of arbitrary remote systems.

## Staged protocol — draft, not an authorized scientific run

### Stage 0: Instrument and establish observability

Implement a typed state-capture and replay interface; validate deterministic continuation, full mutable-state coverage, no target-label access, causal-prefix invariance, timestamp units, and native-vs-instrumented equivalence. Preserve raw native traces, input/output streams, source hashes, checkpoint hashes, and separate ground-truth traces. Establish the ordinary baseline's task competence without treating it as the anomaly objective. Test legal interventions, state cloning, intervention identity (zero action), numerical validity, and bounded resource use. A failure stops the stage; do not fabricate a successful trace.

### Stage 1: Passive native-flow observation

Freeze B and first train/evaluate passive state-only, history-aware, and recurrent observers with no perturbations. Compare raw full-state readers, the observer's compressed state, matched-capacity history windows, output-only and input/output baselines, and simple local linear/state-conditioned response or predictive models. Include shuffled, reversed, integrated, first-only, and final-only history controls. Measure whether local motion improves accessibility and whether it adds anything beyond a complete-state reader with a suitable function class. Preserve native trajectories without forcing phase, attractors, waves, or semantic modes.

### Stage 2: Isolated automated perturbation

Train or evaluate a separate policy using only permitted state/history/query information. Begin with a deterministic fixed direction bank, matched random and shuffled actions, zero actions, and relevant local linear/Jacobian controls. Compare passive observation with native-plus-response, response-only, and direction/payload-only readouts under identical information and capacity budgets. Use disjoint train/validation/test direction banks and the state-by-direction generalization grid: seen/seen, unseen/seen, seen/unseen, unseen/unseen. Reuse directions across states within splits. Evaluate multiple bounded magnitudes and rollout horizons frozen in advance. Do not infer a Goldilocks zone, active advantage, or tangent semantics merely from a peaked scan.

For raw bit interventions, specify the exact representation and legal operation. An input-bit XOR and a latent floating-point perturbation are different treatments. Arbitrary IEEE-754 bit flips may create NaNs, infinities, or unrelated scale changes and are not equivalent to controlled Euclidean noise. Never write into a live external process or physical controller as part of the initial experiment. All perturbations must be reversible in the isolated copy, with explicit failure and validity records.

### Stage 3: Semantic Transformer and evidence stream

Train a causal Transformer on allowed observer/response features to predict a structured event distribution. The primary scientific output is a typed trace with event interval, operation/argument hypotheses, confidence or calibrated probability, evidence references, and an abstain/unknown option. Render plain language from the structured record. A free-form language generator may be evaluated separately, but it cannot validate its own explanation or silently replace unsupported claims with confident prose. Ground-truth traces are restricted to training/validation supervision and held-out scoring. The observer may receive natural-language task descriptions only if explicitly included in all relevant information-matched baselines.

Compare native-only, state-only, state-plus-history, state-plus-fixed probes, state-plus-learned probes, controller-payload-only, and response-only conditions. Include a privileged oracle trace reader as an upper bound, not a fair black-box baseline. Separate the semantic contribution of the monitoring RNN from that of the Transformer through matched bypass and ablation controls. A positive active result requires improvement beyond the controller's own answer information and beyond an information-matched passive reader. H1's controller-signal leakage makes this a mandatory test.

### Stage 4: Continuous integration and generalization

Only after the earlier stages meet their independently frozen gates, integrate continuous capture, policy scheduling, isolated response evaluation, and incremental semantic output. Test held-out programs, new operation compositions, seeds, changed surface encodings, longer histories, new source models, and semantic-equivalent implementations. Report coverage, event/argument accuracy, interval alignment, calibration, false-assertion rate, abstention quality, counterfactual consistency, task performance preservation, and latency. A later external live target requires a separate interface/safety protocol. None of these stages is authorized to proceed merely because a prior gate failed.

## Semantic and causal evaluation

Primary scientific question for the first confirmatory semantic test: does the native-plus-response observer improve held-out operation recovery over the strongest preregistered information-matched passive and controller-only baselines, with no unacceptable false-assertion increase? Freeze the exact metric, minimum effect, uncertainty procedure, seed set, and multiplicity strategy before the run. Do not select the best of many labels, probes, or architectures on test data.

Report event precision/recall/F1 and exact operation/argument accuracy where ground truth is identifiable; calibration (Brier/ECE where appropriate), coverage-risk and abstention; interval onset/offset error in native step units; and cross-run paired intervals. Causal explanations require interventions that discriminate competing hypotheses and predict held-out counterfactual consequences, not only correlate with traces. Distinguish geometric error, semantic accessibility, reader usefulness, and causal importance. An explanation that merely predicts the final answer, restates input syntax, or copies a controller payload does not establish internal logic recovery.

## Timing and the meaning of real time

Use event indexes and monotonic timestamps with documented units. Record source-step time, observation availability, probe dispatch, response completion, semantic emission, and measured lag. In a discrete neural model, one transition, one layer, one token, and one millisecond are not interchangeable. A 1-ms wall-clock sampling or explanation deadline is an optional future engineering requirement, not a demonstrated property of this project. Do not invent sub-step events from interpolation or report a delayed explanation as knowledge available at the earlier moment. If a semantic event depends on future observations, label it retrospective and report its lag. No claim of exact continuous-time logic is authorized without appropriate observability and temporal validation.

## Frozen boundaries and remaining preparation

No AFO empirical result, checkpoint, live capture, semantic decoder, or millisecond guarantee exists yet. Existing H1 interrogator code is a candidate reference, not a validated implementation for this protocol. Before execution: select and freeze the black-box organism and semantic ground truth; audit state hooks; finalize information partitions; choose the actual recurrent/Transformer architectures and capacity controls; freeze seeds, splits, budgets, metrics, hard stops, and implementation; pass outcome-free smoke; then run all preregistered families before opening the aggregate. Preserve raw and negative results and never modify R9 frozen branches to rescue an outcome.

The preparation scaffold is in `experiments/afo_1/`. It defines a trace contract and replay-safe intervention interface, not a trained observer or a launched experiment. Literature is reviewed in [ACTIVE_FLOW_OBSERVER_LITERATURE.md](ACTIVE_FLOW_OBSERVER_LITERATURE.md).