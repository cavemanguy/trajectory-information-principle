# ALI-N8-R1 Seed 17 — Frozen Evidence Record

Seed 17 is the first clean confirmatory R1 replication with the test split untouched until the final frozen evaluation.

## Provenance

- Workflow: `ALI-N8-R1 Seed 17 Full`
- Workflow run ID: `33548609053`
- Launcher commit on `main`: `ddead67e34970e7b713b8b3a2b4745d9686a40fb`
- R1 core test-blind patch: `6b9a648fafe400dfd3afedf9771fca5e25950c26`
- Final artifact: `ali-n8-r1-seed17-final`
- Final artifact ID: `9817398839`
- Final artifact SHA-256: `a336647ad372f005237edc44d19d8c6885477de558f525510365a5e7bdc0218b`
- Core checkpoint SHA-256: `0919baae436f62f78f3084585379345d61ee18db4a2e537e93847f5f502e27f6`
- Test dataset SHA-256: `1281dad6ddd04b01c5538bf4433d188f788db2be64737e138e22c3157a7a7526`
- Test latent SHA-256: `bf5d59f67eac05cdb008e5d46976470a42741004503afe6e5a0ab13c86bc446b`

The workflow explicitly verified `test_generated=false` and `test_evaluated=false` before the final phase. Test data were first generated/evaluated in the final frozen job.

## Key frozen results

- Adaptive ALI `P(m,q)`: 62.4725%
- Query-only ALI `P(q)`: 26.8625%
- Query-blind adaptive `P(m)`: 49.6725%
- Learned fixed direction: 21.0750%
- Random direction: 15.8475%
- Direct reader from `m`: 65.6575%
- Direct reader from `F(m)`: 58.6425%
- Zero perturbation: 5.9100%
- Direction-only leakage `P(m,q)`: 65.4300%
- Direction-only leakage `P(m)`: 64.2025%

Wrong-memory intervention:
- Native adaptive ALI: 62.4725%
- Wrong-memory direction: 6.4275%
- Prediction-change rate: 92.4150%
- Paired accuracy difference: +56.0450 pp

Query-only selectivity:
- `D_native = +18.5543 pp`
- `D_decode = +8.6136 pp`
- All eight per-relation `D_decode` values are positive.

## Interpretation boundary

The strong direction-only leakage means the adaptive `P(m,q)` result is not clean evidence of information being exposed only through the frozen transformation's response; the policy direction itself carries substantial target information. Under the preregistered hierarchy this supports adaptive computation but not clean content-dependent interrogation.

The query-only result is the stronger geometric result. Because `P(q)` cannot inspect memory and the independently trained 64-decoder matrix also shows positive diagonal selectivity, seed 17 reproduces the query-specific response geometry observed in seed 5 without seed 5's early-core-test procedural caveat.

This file is a compact repository record. The immutable GitHub Actions artifact above remains the authoritative complete machine output, including prediction files and checkpoint hashes. No result has been tuned, altered, or suppressed after test evaluation.
