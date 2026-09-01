# ALI-N8-R1 Seed 31 Final Evidence

**Status: FROZEN — DO NOT MODIFY OR REPLACE**

This directory is the permanent compact repository record for primary seed 31. The authoritative full result bundle is the immutable GitHub Actions artifact from workflow run `33551465026`, artifact ID `9818401182`, named `ali-n8-r1-seed31-final`.

Artifact ZIP SHA-256: `77817ed04038ad5c1427d0f15d520b670431f0c950c805b6380dd80c7c77ba81`

Core checkpoint SHA-256: `1d03caee34a22c79d2340e2918f6c90db97f21f01d78284fd18f779b0c80fe6f`

Seed 31 is a clean primary execution. The core phase recorded `test_generated=false`, `test_evaluated=false`, and `status=core_complete_frozen_test_unseen`. The final workflow independently asserted that the test set was still unseen before running the single final test phase.

Key frozen results:

- adaptive ALI `P(m,q)`: 65.5950%
- query-only ALI `P(q)`: 27.2075%
- direct `m`: 68.1100%
- direct `F(m)`: 61.3650%
- zero perturbation: 6.1600%
- direction-only adaptive leakage: 67.9650%
- wrong-memory adaptive accuracy: 6.3700%
- wrong-memory prediction-change rate: 92.4075%
- `D_native`: +20.6271 percentage points
- `D_decode`: +9.7414 percentage points

All eight relation-level independent-decoder diagonal advantages are positive. This seed therefore reproduces the preregistered query-specific selectivity pattern, but it does not establish attention superiority, compression, universal addresses, or a new computational primitive by itself.

The full Actions artifact remains authoritative for per-example predictions, logs, all matrices, counts, environment metadata, and checkpoint hashes. This directory is a compact human-auditable freeze tied to that artifact.
