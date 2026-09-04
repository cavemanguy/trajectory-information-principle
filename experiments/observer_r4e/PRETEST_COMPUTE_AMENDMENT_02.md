# R4E Pre-Test Compute Amendment 02

**Locked before any R4E Phase-I test evaluation.**

For seed 7, family `J_a1g0`, width 64 completed validation evaluation and achieved validation MSE 0.0031723622, cosine 0.8947235, EV 0.8976970. The subsequent checkpoint write failed. Multiple deterministic regeneration attempts under the unchanged training recipe exceeded the available execution window before a checkpoint could be persisted.

No R4E test example or `V_test` rollout had been evaluated when this amendment was locked.

Therefore, for the concurrent mechanistic hierarchy only, seed 7 `J_a1g0` test evaluation uses the best successfully preserved checkpoint from the preregistered width set: width 32 (validation MSE 0.0035349457, cosine 0.8878750, EV 0.8860044). The width-64 validation result remains reported as a validation-only diagnostic and is not silently discarded.

This amendment does **not** alter the primary nonlinear `C_phi` model selection, the mandatory Phase-I controls, the Phase-I gate, epsilon, direction banks, source seeds, test split, metrics, or stopping rule.

Local amendment SHA-256: `52b35974ddb3a477621cc4c65f96ca07c490067d56a03b813083506f5759020c`.
