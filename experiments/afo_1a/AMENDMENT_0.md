# AFO-1A — Pre-outcome implementation clarification

This note resolves ambiguities before scientific source training. The preregistered task, seeds, budget, final-checkpoint rule, and gates are unchanged. It is not a rescue of an observed outcome.

1. The two 16-dimensional type and payload embeddings are concatenated, followed by LayerNorm32, as required by the stated GRU32->64 input dimension. The word “summed” in the preregistration is an inconsistent description, not an additional embedding operation.
2. Canonical held-out evaluation uses one episode at a time under deterministic CPU float32 execution. Replay identity compares the same one-episode computation with the same event sequence and model parameters. A cross-batch-size bitwise-equivalence claim is not required; floating-point kernels can differ with batch shape. Training retains batch64.
3. Scheduled training checkpoint records contain the loss and pre-clipping gradient norm every100 steps. The only ordinary source-model checkpoint is the exact final state_dict after1600 updates. Failure records preserve whatever partial state exists, without authorizing selection or continuation.
4. The source adapter's complete mutable continuation state is the hidden tensor and event index. It has no stochastic inference, mutable cache, optimizer, or external process. Forked models own independent parameters and state. The parameter fingerprint covers all tensor contents in state_dict.
5. The reference interpreter's operation labels are task semantics, not evidence of the learned neural implementation. They are held outside the source forward interface. Source qualification does not authorize an AFO-1B result or establish semantic recovery.

All clarifications precede the independent smoke gate and scientific outcomes. An implementation repair discovered before outcome inspection must be recorded and tested; post-outcome changes require a new experiment.
