# R9-H1 Implementation Notes

The implementation must preserve the frozen R9-T1 task semantics and may reuse its generator code. Architecture-specific controller channels must remain causal, bounded, and free of direct target-label input.

Smoke tests are outcome-free and limited to determinism, tensor shape, masking, artifact-schema, finite-loss, and target-leak structural checks.
