# R8-M5 Preregistration — Capacity Allocation Test

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent results:** R8-M1 = M3; R8-M2 = S0; R8-M3 = T0; R8-M4 = F3  
**Primary study:** native training only; no latent-state perturbation, active observer, or trajectory steering

## 1. Scientific question

R8-M4 established a functional contribution of relation-selective native survival under a controlled suppression intervention. Its preregistered secondary analysis showed that the functional loss was highly concentrated in the baseline dynamically favored relation: the survival winner averaged 74.34% h12 test accuracy versus 13.93% across the other seven relations, and equalization reduced winner accuracy by about 30.36 percentage points while non-winner accuracy was approximately preserved.

R8-M5 asks:

> **Is this one-relation-dominant specialization a state-capacity/resource-allocation strategy created by the tight autonomous recurrent bottleneck, rather than merely a generic benefit of adding parameters?**

This is a new confirmatory experiment. The R8-M4 specialist-concentration result is motivation, not an R8-M5 outcome.

## 2. Guardrails

- Native task trajectories only; no hidden-state perturbations.
- Fresh seeds only.
- Same synthetic task/data generation and 12-step autonomous recurrence lineage as R8-M1 through R8-M4.
- Width conditions share the same train/validation/test memories, presentation permutations, and minibatch order within a family.
- All identically named tensors with identical shapes are copied from the 16-D baseline initialization into the other conditions before training, so common-shape initialization is matched where mathematically possible.
- The primary claim is about **state capacity and allocation pattern**, not about parameter efficiency or practical superiority.
- A larger model performing better is not sufficient for a positive result.
- R8-M4 remains F3 regardless of R8-M5 outcome.

## 3. Task and fixed architecture components

Use the same task lineage:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embedding width 8;
- encoder GRU hidden width 32;
- 12 autonomous recurrent transitions after encoding;
- separate h0 and h12 linear heads per relation;
- joint loss `mean_r[CE(h0_r)+CE(h12_r)]`;
- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- batch 256; gradient clip 1.0;
- 100 epochs;
- 20,000 train / 2,500 validation / 5,000 test memories;
- held-out natural-pair bank: 2,048 valid pairs per relation.

No external input enters after h0.

## 4. Capacity conditions

Each fresh family trains four conditions on identical data/order/permutations.

### B16 — lineage baseline

- latent/state width = 16
- recurrent hidden width = 32
- `F: 16 -> 32 -> 16`

This reproduces the R8-M1–M4 architecture.

### S24 — intermediate state capacity

- latent/state width = 24
- recurrent hidden width = 32
- `F: 24 -> 32 -> 24`

### S32 — wider state capacity

- latent/state width = 32
- recurrent hidden width = 32
- `F: 32 -> 32 -> 32`

### P16 — near-parameter-matched fixed-state control

- latent/state width = 16
- recurrent hidden width = 192
- `F: 16 -> 192 -> 16`

P16 keeps the 16-D autonomous state bottleneck but increases transition-network parameter capacity. Under the frozen implementation, P16 and S32 total trainable parameter counts must differ by less than 5%.

Purpose: distinguish a **state-width effect** from the weaker explanation that a larger parameter count simply improves training.

## 5. Fresh seeds

Twelve fresh family seeds are fixed:

`[18, 33, 46, 61, 76, 91, 106, 121, 141, 156, 173, 188]`

They do not overlap historical Observer seeds, ALI-N8-R1, ND-R1, R8-M1, R8-M2, R8-M3, or R8-M4 seed sets.

## 6. Checkpoints and evaluation

Save/evaluate at epochs:

`[0, 20, 40, 60, 80, 100]`.

At each checkpoint record:

- validation h0/h12/combined accuracy;
- per-relation validation h12 accuracy;
- relation-wise terminal survival `S_r(12)` on the held-out natural-pair bank;
- `G = SD_r(log(S_r(12)+eps))`;
- mean log survival `C`;
- survival winner relation.

At epoch 100 also record test h0/h12/combined accuracy and per-relation test accuracy.

For each condition define the **survival-winner performance gap**:

`D = Acc_h12(r_survival_winner) - mean_{j != r_survival_winner} Acc_h12(j)`.

D links the dynamical specialist to functional concentration. It is not the same as simply taking the maximum accuracy relation.

## 7. Training-validity gate

Every condition in every fresh family must satisfy at epoch 100:

- combined validation >= 0.38;
- h0 validation >= 0.55.

If any condition fails, classify **V — cross-capacity training validity failure** and stop primary capacity interpretation.

The gate is intentionally based partly on h0 so a positive capacity result cannot be produced by simply degrading the encoder-side task representation.

## 8. Frozen primary contrasts

All confidence intervals use deterministic 5,000-resample family bootstraps across the 12 paired seeds.

### A — wider-state allocation pattern: S32 versus B16

Define paired differences `S32 - B16`.

A is supported only if **all** hold:

1. **Terminal performance improves materially:** mean `Delta h12 >= +0.02` and bootstrap 95% CI lower bound `> 0`.
2. **Functional winner dominance decreases:** mean `Delta D <= -0.10` and 95% CI upper bound `< 0`.
3. **Dynamical selectivity decreases:** mean `Delta G < 0` and 95% CI upper bound `< 0`.
4. **h0 is not traded away:** mean `Delta h0 > -0.02` and 95% CI lower bound `> -0.03`.

A positive A result means wider state capacity changes the pattern in the direction predicted by a bottleneck-allocation account; it does not yet isolate state dimension from generic added parameters.

### S — state-dimension specificity: S32 versus P16

S is supported only if both hold:

1. mean `D_S32 - D_P16 <= -0.05` with 95% CI upper bound `< 0`;
2. mean `G_S32 - G_P16 < 0` with 95% CI upper bound `< 0`.

The implementation must also verify before fresh outcomes that P16 and S32 trainable parameter counts differ by <5%.

A positive S result means the reduction in specialist concentration/selectivity is not reproduced merely by giving a 16-D state a much larger transition MLP with nearly matched total parameter count.

## 9. Frozen classification

After validity:

- **C0 — simple capacity-allocation account not supported:** A is false.
- **C1 — wider-state allocation pattern supported, state specificity not established:** A true, S false.
- **C2 — state-dimension-specific capacity allocation supported:** A true, S true.
- **V — cross-capacity training validity failure.**

C2 is the strongest preregistered outcome. It supports a state-capacity/resource-allocation account in this architecture; it does not establish universality, optimality, or a new computational principle.

## 10. Secondary analyses

Report without changing the primary classification:

- S24 as an intermediate-width dose-response point;
- per-family monotonic Spearman trend across B16/S24/S32 for h12, D, and G;
- per-relation h12 distributions across widths;
- survival-winner/accuracy-winner agreement;
- relation-wise correlation between terminal log survival and h12 accuracy;
- h0 per-relation balance;
- epoch at which width conditions begin to diverge in D and G;
- total parameter counts and parameter-count ratios;
- whether widening state capacity increases the number of relations above fixed descriptive accuracy thresholds (reported descriptively only; no threshold can replace the primary D endpoint).

## 11. Claim boundaries

R8-M5 cannot establish:

- strong emergence;
- universal trajectory information;
- essential chronology;
- practical superiority over transformers/RNNs;
- that state capacity is the only source of specialization;
- that larger state width is always better;
- that Euclidean survival magnitude itself mediates reader usefulness;
- language-model or real-world generalization.

The strongest allowed conclusion under C2 is:

> **Within this symmetric synthetic autonomous recurrent system, increasing state dimension reduces one-specialist dynamical/functional concentration while improving terminal performance, and a near-parameter-matched wider transition network that retains the 16-D state bottleneck does not reproduce that redistribution. This supports a state-capacity/resource-allocation interpretation of the R8 specialization.**
