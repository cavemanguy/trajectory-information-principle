# Result Artifacts

This directory contains machine-readable aggregate summaries recovered from the experiments discussed in the documentation.

The autonomous ALI experiment used seeds **5, 17, and 31**. The checked-in CSVs record aggregate mean and standard deviation across those runs. They do not contain per-example predictions, checkpoints, or complete per-seed rows; those should be added when recovered.

Files:

- `autonomous_geometric_addressability_summary.csv`: native query-conditioned, query-blind, and direct-readout comparisons
- `autonomous_geometric_causal_summary.csv`: causal probe-direction interventions
- `autonomous_probe_direction_mechanics_summary.csv`: direction-geometry measurements
- `corrected_attention_n8_summary.csv`: recovered comparable positional-attention result for the 8-relation setting

Values are preserved at their recorded precision. Documentation rounds percentages to two decimal places for readability.
