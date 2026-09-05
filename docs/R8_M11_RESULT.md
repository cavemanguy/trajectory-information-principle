# R8-M11 Result — Recurrent Map Substructure Localization

## Frozen classification

**L1 — input-stage contribution supported**

R8-M11 used twelve fresh families and first required two replication gates before interpreting the finer recurrent-map partition.

- Parent matched-midpoint history replication R: **passed**.
- Whole recurrent-map carrier replication F: **passed**.
- Input-facing recurrent stage F1 criterion: **passed**.
- Output-facing recurrent stage F2 criterion: **did not pass**.
- F1 × F2 interaction descriptor: **did not pass**.

## Frozen statistics

- `H_parent`: mean `+0.649251`, median `+0.163183`, bootstrap 95% CI `[+0.161200,+1.300857]`, positive 12/12.
- `F_total`: mean `+0.453112`, median `+0.154239`, CI `[+0.108788,+0.904060]`, positive 10/12.
- `F1_effect`: mean `+0.261654`, median `+0.130698`, CI `[+0.087444,+0.471846]`, positive 9/12.
- `F2_effect`: mean `+0.191457`, median `+0.056462`, CI `[-0.003168,+0.474447]`, positive 10/12.
- `I12`: mean `+0.182707`, median `+0.003611`, CI `[-0.045470,+0.551428]`, positive 7/12.

## Interpretation

Within the frozen two-stage partition of the tested recurrent map, the input-facing `Linear(16,32)` stage makes a reproducible causal contribution to the persistent matched-demand history effect. The output-facing `Linear(32,16)` stage showed a positive mean and was positive in 10/12 families, but it did not satisfy the frozen support rule because its bootstrap interval crossed zero.

The family distribution is heterogeneous. Two families carried very large effects while several showed much smaller effects. Therefore the result supports an aggregate contribution from F1; it does not support the claim that F1 is the unique carrier or that every lineage has a large effect.

## Claim boundary

R8-M11 localizes causal contribution only to the frozen two-stage parameter partition inside the tested recurrent map. It does not establish unique storage, individual-neuron or low-rank causality, formal bistability/hysteresis, information beyond the complete state, or generalization beyond this synthetic architecture.
