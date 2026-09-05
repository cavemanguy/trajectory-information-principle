# R8-M5R Preregistration — Free-Specialization Capacity Replication

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent study:** R8-M5 = C0 — simple capacity-allocation account not supported  
**Training implementation:** exact R8-M5 architecture/training code, reused without changing the model or optimization algorithm

## 1. Why this replication exists

R8-M5 was designed around a directional prediction: if the one-relation-dominant specialization were a scarce-state-capacity compromise, widening the autonomous state should improve performance while reducing dynamical selectivity and specialist dominance. That directional account failed. In the observed R8-M5 sample, wider state improved performance while selectivity and specialist dominance increased.

The failed R8-M5 classification remains **C0** and is not reinterpreted or repaired.

R8-M5R asks a narrower question without imposing a directional requirement on specialization:

> **Does increasing autonomous state dimension improve terminal task performance beyond both the 16-D lineage baseline and a near-parameter-matched 16-D control, when the learned system is free to specialize more, less, or not at all?**

The specialization response itself is measured, but its sign is not a success criterion.

## 2. Guardrails

- R8-M5 remains C0 regardless of this result.
- This is a new confirmatory replication on fresh seeds, not a reclassification of R8-M5.
- The exact R8-M5 training implementation and four architecture conditions are reused.
- Only the fresh seed bank / deterministic seed namespace and the frozen cross-family decision rule change.
- No outcome is required to increase or decrease `G` or `D`.
- A run with weak specialization is not treated as a failed run if it satisfies the ordinary training-validity gate.
- A run with strong specialization is not treated as more successful merely because `G` or `D` is larger.
- Specialization-performance association is secondary and descriptive; it does not establish why a given optimization path specialized.

## 3. Architecture and data

Use the exact R8-M5 lineage:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embedding width 8;
- encoder GRU hidden width 32;
- 12 autonomous recurrent transitions after encoding;
- separate h0 and h12 linear heads per relation;
- joint task loss `mean_r[CE(h0_r) + CE(h12_r)]`;
- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- batch 256; gradient clip 1.0;
- 100 epochs;
- 20,000 train / 2,500 validation / 5,000 test memories;
- natural-pair bank of 2,048 valid pairs per relation.

Conditions are unchanged:

- **B16:** state 16, recurrent hidden 32, `F: 16 -> 32 -> 16`;
- **S24:** state 24, recurrent hidden 32;
- **S32:** state 32, recurrent hidden 32;
- **P16:** state 16, recurrent hidden 192.

As in R8-M5, P16 and S32 total trainable parameter counts must differ by less than 5% before fresh outcomes can be interpreted.

## 4. Fresh seeds

Twelve new family seeds are fixed:

`[20, 41, 56, 69, 84, 99, 118, 133, 146, 161, 181, 199]`

They do not overlap the historical Observer/ALI/ND-R1/R8-M1/R8-M2/R8-M3/R8-M4/R8-M5/R8-M6 fresh-seed sets.

Each family uses identical train/validation/test memories, presentation permutations, and minibatch order across B16/S24/S32/P16, exactly as in R8-M5.

## 5. Training-validity gate

Every condition in every family must satisfy at epoch 100:

- combined validation >= 0.38;
- h0 validation >= 0.55.

If any condition fails, classify:

**V — cross-capacity training validity failure**

and do not promote the primary capacity comparison.

## 6. Primary endpoint

The primary functional endpoint is epoch-100 **test h12 accuracy**.

All intervals use deterministic 5,000-resample paired family bootstraps.

Define:

`Delta_B = h12(S32) - h12(B16)`

`Delta_P = h12(S32) - h12(P16)`

### Primary test P — wider-state terminal benefit

P is supported only if:

1. mean `Delta_B >= +0.02`; and
2. the 95% CI lower bound for `Delta_B` is `> 0`.

### Specificity test S — benefit beyond parameter-matched 16-D control

S is supported only if:

1. mean `Delta_P >= +0.015`; and
2. the 95% CI lower bound for `Delta_P` is `> 0`;
3. P16/S32 parameter-count relative difference is `< 0.05` in every family.

## 7. Frozen classification

After validity:

- **R0 — wider-state terminal benefit not supported:** P false.
- **R1 — wider-state terminal benefit supported, state-dimension specificity not established:** P true, S false.
- **R2 — state-dimension-specific terminal benefit supported:** P true, S true.
- **V — cross-capacity training validity failure.**

No classification depends on whether specialization rises or falls.

## 8. Specialization response — preregistered descriptive analysis

For every condition/family report:

- terminal selectivity `G = SD_r(log(S_r(12)+eps))`;
- survival-winner functional gap `D`;
- survival winner identity;
- per-relation h12 accuracy;
- h0 accuracy.

For S32-B16 and S32-P16, report paired mean differences and 95% CIs for:

- `Delta G`;
- `Delta D`;
- `Delta h0`.

Each specialization contrast is labeled only by the interval:

- **increase** if CI lower bound > 0;
- **decrease** if CI upper bound < 0;
- **indeterminate/mixed** otherwise.

These labels are descriptive and cannot change R0/R1/R2.

## 9. Adaptive-specialization secondary analyses

To evaluate the hypothesis that optimization paths may use specialization to different degrees, report without causal interpretation:

1. per-seed signs and magnitudes of `Delta G`, `Delta D`, and `Delta h12` for S32-B16;
2. across-seed Spearman correlation between `Delta G` and `Delta h12`;
3. across-seed Spearman correlation between `Delta D` and `Delta h12`;
4. within each condition, across-seed Spearman correlation between `G` and h12 performance;
5. within each condition, survival-winner versus accuracy-winner agreement;
6. S24 as an intermediate state-width point, including per-family width-order trends for h12, G, and D.

A positive association would be consistent with stronger specialization accompanying larger functional gains in this setting. A null or negative association would show that specialization strength is not a simple scalar proxy for performance. None of these correlations establishes that the system "needed" specialization on a particular run.

## 10. Claim boundary

The strongest allowed conclusion under R2 is:

> **Within this symmetric synthetic autonomous recurrent architecture, increasing state dimension improves terminal task performance beyond both the 16-D lineage baseline and a near-parameter-matched 16-D transition-capacity control. The accompanying specialization response is allowed to vary and is reported separately rather than being forced to increase or decrease.**

R8-M5R cannot establish:

- that specialization is always beneficial;
- that the system consciously or explicitly chooses specialization;
- a universal trajectory-information principle;
- strong emergence;
- essential chronology;
- language-model or real-world generalization;
- that Euclidean survival magnitude itself mediates reader usefulness.
