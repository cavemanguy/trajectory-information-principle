# Result Artifacts

## Recovered historical aggregate results

The CSV files currently stored directly in this directory are **recovered historical aggregate evidence**. They are preserved unchanged at their recorded precision.

The historical autonomous N=8 ALI experiment used seeds **5, 17, and 31**. The checked-in aggregate CSVs preserve measurements from those runs, but the exact historical dataset generator, current-mechanism implementation, complete training configuration, checkpoint-selection procedure, raw per-seed rows, checkpoints, and per-example predictions are unavailable.

Therefore the historical autonomous N=8 results are **not independently reproducible from this repository**. They must not be presented as if the present repository can regenerate the reported ~64.67% native result. A new implementation must not be reconstructed or tuned to match that number.

Historical files:

- `autonomous_geometric_addressability_summary.csv`: recovered native query-conditioned, query-blind, and direct-readout aggregate comparisons
- `autonomous_geometric_causal_summary.csv`: recovered aggregate causal probe-direction interventions
- `autonomous_probe_direction_mechanics_summary.csv`: recovered aggregate direction-geometry measurements
- `corrected_attention_n8_summary.csv`: recovered aggregate positional-attention result for the 8-relation setting

These historical CSVs should remain unchanged. Documentation may round percentages for readability, but the machine-readable values are the preserved evidence record.

## New reproducible experiments

New experiments are scientifically and operationally separate from the recovered historical record. Their outputs belong under:

`results/reproducible/<experiment_version>/`

The first preregistered reconstruction-independent experiment is **ALI-N8-R1**, specified in `experiments/ali_n8_r1/PREREGISTRATION.md`. R1 is a new experiment from first principles and is not intended to reproduce the historical aggregate accuracy.

For reproducible experiments, retain per-seed configuration, training histories, checkpoints/hashes, predictions, intervention matrices, and mechanically generated aggregate summaries. Negative results and failed primary seeds remain part of the record.
