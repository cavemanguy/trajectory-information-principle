# ALI-N8-R1 Three-Seed Aggregate

This directory aggregates the preregistered primary seeds **5, 17, and 31** without changing the locked scientific design or excluding any primary seed.

## Protocol status

Seeds **17** and **31** remained fully test-blind through core selection, ALI/control selection, and all 64 diagnostic-decoder selections; their test sets were first generated/evaluated in the final phase. Seed **5** has the documented protocol deviation that temporary core heads were evaluated on test before downstream model selection. That exposure did not change the selected core checkpoint, alpha, experimental design, or downstream selection rules, but it remains a caveat and must stay attached to the R1 record.

## Primary aggregate

Across seeds 5, 17, and 31:

- query-only ALI `P(q)`: **26.5450% mean**, sample SD **0.8661 pp**
- adaptive ALI `P(m,q)`: **66.7783% mean**, sample SD **5.0036 pp**
- direct `m`: **69.5700% mean**, sample SD **4.8116 pp**
- direct `F(m)`: **61.9125% mean**, sample SD **3.5753 pp**
- zero perturbation: **6.1583% mean**, sample SD **0.2475 pp** (chance = 6.25%)
- adaptive direction-only leakage: **69.8167% mean**, sample SD **5.5493 pp**
- wrong-memory adaptive accuracy: **6.3933% mean**, sample SD **0.0302 pp**
- wrong-memory prediction-change rate: **92.4042% mean**, sample SD **0.0128 pp**

Preregistered query-only selectivity endpoints:

- `D_native`: **+19.4327 pp mean**, sample SD **1.0719 pp**
- `D_decode`: **+9.5751 pp mean**, sample SD **0.8901 pp**

The independent-decoder diagonal advantage is positive for **24 of 24 relation-level seed/relation combinations** across the three primary seeds.

## Interpretation

R1 meets its preregistered **Level-4 evidence pattern**: query-only perturbation directions reproduce positive native-swap selectivity and positive independent-decoder selectivity across all three primary seeds. Under the preregistered diagnostic decoder class, the response produced by the direction associated with relation `i` makes relation `i` more decodable than responses produced by the other learned query-only directions.

The narrow supported claim is therefore:

> In this learned frozen latent system, query-specific perturbation directions reproducibly produce direction-dependent local responses that preferentially expose information associated with their respective queries under the preregistered diagnostic decoder class.

R1 does **not** establish that ALI beats direct reading or attention, that the latent state is compressed, that the learned directions are universal or orthogonal addresses, that the effect generalizes to language models, or that ALI is a new computational primitive.

The adaptive `P(m,q)` system also shows strong causal dependence on memory-conditioned directions, but high direction-only leakage means that result should not be interpreted as clean content-dependent interrogation.

`primary_seed_summary.csv` preserves the exact per-seed headline measurements used for this aggregate. Individual frozen seed directories and their authoritative GitHub Actions artifacts remain the source of truth for matrices, counts, predictions, hashes, and environment metadata.
