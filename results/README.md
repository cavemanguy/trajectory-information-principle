# Result Artifacts

## Recovered historical aggregate results

The CSV files stored directly in this directory are **recovered historical aggregate evidence**. They are preserved unchanged at their recorded precision.

The historical autonomous N=8 ALI experiment used seeds **5, 17, and 31**. The checked-in aggregate CSVs preserve measurements from those runs, but the exact historical dataset generator, implementation, complete training configuration, checkpoint-selection procedure, raw per-seed rows, checkpoints, and per-example predictions are unavailable.

Therefore the historical autonomous N=8 results are **not independently reproducible from this repository**. They must not be presented as if the present repository can regenerate the historical ~64.67% native result. The new reproducible R1 experiment was not reconstructed or tuned to match that number.

Historical files:

- `autonomous_geometric_addressability_summary.csv`
- `autonomous_geometric_causal_summary.csv`
- `autonomous_probe_direction_mechanics_summary.csv`
- `corrected_attention_n8_summary.csv`

These historical CSVs should remain unchanged.

## Reproducible ALI-N8-R1

The first reconstruction-independent preregistered experiment is **ALI-N8-R1**, specified in `experiments/ali_n8_r1/PREREGISTRATION.md`.

Primary seeds: **5, 17, 31**.

All three primary runs completed. Seeds **17** and **31** remained test-blind through core selection, ALI/control selection, and diagnostic-decoder selection. Seed **5** has a documented protocol deviation: temporary core heads were evaluated on test before downstream selection. See `experiments/ali_n8_r1/PROTOCOL_DEVIATIONS.md`.

Frozen seed records are under:

- `results/reproducible/ali_n8_r1/seed_5/final/`
- `results/reproducible/ali_n8_r1/seed_17/final/`
- `results/reproducible/ali_n8_r1/seed_31/final/`

The mechanically derived three-seed summary is under:

- `results/reproducible/ali_n8_r1/aggregate/`

### Three-seed headline results

Across seeds 5, 17, and 31:

- query-only ALI `P(q)`: **26.5450% mean**, sample SD **0.8661 pp**
- adaptive ALI `P(m,q)`: **66.7783% mean**, sample SD **5.0036 pp**
- direct `m`: **69.5700% mean**, sample SD **4.8116 pp**
- direct `F(m)`: **61.9125% mean**, sample SD **3.5753 pp**
- zero perturbation: **6.1583% mean**, sample SD **0.2475 pp**
- adaptive direction-only leakage: **69.8167% mean**, sample SD **5.5493 pp**
- wrong-memory adaptive accuracy: **6.3933% mean**, sample SD **0.0302 pp**
- wrong-memory prediction-change rate: **92.4042% mean**, sample SD **0.0128 pp**

Preregistered query-only selectivity endpoints:

- `D_native`: **+19.4327 pp mean**, sample SD **1.0719 pp**
- `D_decode`: **+9.5751 pp mean**, sample SD **0.8901 pp**

The relation-level independent-decoder diagonal advantage is positive in **24/24** seed/relation combinations.

Under the preregistered claim hierarchy, R1 therefore meets the **Level-4 evidence pattern** for this learned frozen latent system: query-specific directions produce reproducible direction-dependent responses that preferentially expose their associated relation under the preregistered diagnostic decoder class.

This does **not** establish that ALI beats direct reading or attention, that the latent state is compressed, that the directions are universal or orthogonal addresses, that the effect generalizes to language models, or that ALI is a new computational primitive.

The adaptive `P(m,q)` result remains mechanistically confounded by strong direction-only leakage even though the wrong-memory intervention shows strong causal dependence on the memory-conditioned direction.

For reproducible experiments, negative results, failed primary seeds, hashes, predictions, matrices, and protocol deviations remain part of the record. The full GitHub Actions artifacts are authoritative for generated per-example outputs and logs; the repository stores compact permanent evidence tied to those artifacts.
