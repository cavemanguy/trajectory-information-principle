# R8-M6 Preregistration — Isolated Recurrent Workspace Expansion

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent results:** R8-M1 = M3; R8-M2 = S0; R8-M3 = T0; R8-M4 = F3; R8-M5 = C0  
**Primary study:** native training with a function-preserving architecture fork; no latent-state perturbation, active observer, or trajectory steering

## 1. Scientific question

R8-M5 rejected the simple tight-bottleneck allocation account: widening the full state from 16D to 32D improved terminal performance while strengthening, rather than weakening, both dynamical selectivity and survival-winner functional concentration. However, S32 also improved h0 test accuracy by about 34.65 percentage points, so R8-M5 did not isolate recurrent workspace dimension from encoder-side representation quality.

R8-M6 asks:

> **If the encoder output remains 16D, does a function-preserving expansion of only the autonomous recurrent workspace strengthen useful specialization and terminal performance?**

The key design feature is an exact precommitment fork. All conditions begin from the same trained 16D model and initially implement the same input-output function and the same native trajectory, up to zero-padding of the expanded state.

## 2. Guardrails

- No external information enters after h0.
- No hidden-state perturbations are injected.
- All fresh conditions are bit-identical through the common epoch-20 model.
- Architecture changes occur only at the frozen epoch-20 fork and are initialized to preserve the pre-fork function exactly on all inputs, up to floating-point tolerance.
- All three post-fork optimizers are reset to fresh AdamW states at epoch 21 so optimizer-state compatibility cannot favor one architecture.
- The encoder remains 16D in every condition.
- A positive workspace claim requires h0 performance to remain within a preregistered equivalence margin; if h0 materially diverges, recurrent-workspace isolation is not established.
- R8-M5 remains C0 regardless of R8-M6 outcome.

## 3. Shared task and pre-fork lineage

Use the same symmetric synthetic task:

- 8 categorical relations;
- 16 values per relation;
- relation/value embeddings width 8;
- encoder GRU hidden width 32;
- encoder output h0 width 16;
- 12 autonomous recurrent transitions;
- separate h0 and h12 relation heads;
- joint task loss `mean_r[CE(h0_r)+CE(h12_r)]`;
- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- batch 256; gradient clip 1.0;
- 20,000 train / 2,500 validation / 5,000 test memories;
- held-out natural-pair bank 2,048 valid pairs per relation.

Train one ordinary lineage model from epoch 0 through epoch 20 with:

- state width 16;
- recurrent hidden width 32;
- `F: 16 -> 32 -> 16` with GELU then tanh.

Epoch 20 is fixed because prior R8 studies place stable commitment later than this point in the baseline lineage.

## 4. Function-preserving fork

At epoch 20 create three continuation models.

### B16 — ordinary continuation

Exact copy of the common model.

### X32 — expanded recurrent workspace

Keep the encoder, h0, and h0 heads at width 16. Expand only the autonomous recurrent state to width 32.

- lift: `z0 = concat(h0, zeros_like(h0))`;
- recurrent map: `F_X: 32 -> 32 -> 32`;
- terminal heads read 32D state.

Initialize X32 by block-embedding the learned B16 recurrent map:

- first recurrent linear layer copies the B16 weights into the first 16 input columns and zeros the new input columns;
- second recurrent linear layer copies the B16 output into the first 16 state coordinates and zeros the new output rows;
- terminal heads copy the B16 weights into their first 16 columns.

The added input columns of the first recurrent linear layer receive deterministic dormant random weights with fixed scale `0.02`. Because all added state coordinates are exactly zero at the fork, these columns do not change the forward trajectory. Through later recurrent steps they provide a nonzero backward path into the zero-initialized added output rows, allowing the added state coordinates to become learnable after training resumes. Terminal-head columns on the added coordinates remain zero at the fork.

At the fork, for every input:

`z_t^X = concat(h_t^B, 0)` for `t=0..12`,

and h12 logits are equal within floating-point tolerance.

### P16 — near-parameter-matched wider-transition control

Keep the recurrent state at width 16 but expand the recurrent hidden layer:

- state width 16;
- recurrent hidden width 126;
- `F_P: 16 -> 126 -> 16`.

Initialize the first 32 hidden units by exact embedding of the learned B16 recurrent map. The remaining 94 first-layer rows are zero so their activations are exactly zero at the fork. The corresponding second-layer columns receive deterministic dormant random weights with fixed scale `0.02`; they do not alter the fork function but provide a gradient path that allows the added hidden units to become learnable.

P16 and X32 total trainable parameter counts must differ by less than 5% before fresh outcomes.

## 5. Fresh seeds

Twelve fresh family seeds are fixed:

`[22, 36, 49, 67, 82, 97, 113, 129, 144, 159, 177, 193]`

They do not overlap the historical Observer, ALI-N8-R1, ND-R1, or R8-M1 through R8-M5 seed sets.

## 6. Post-fork training

At epoch 21 reset AdamW optimizer state independently for B16, X32, and P16 using the same hyperparameters.

All three paths then use identical deterministic:

- task data;
- minibatch order;
- relation-presentation permutations;
- epoch count through 100.

No auxiliary loss is used.

## 7. Checkpoints and measurements

Save/evaluate at:

`[20, 25, 30, 35, 40, 50, 60, 80, 100]`.

At every checkpoint record:

- validation h0/h12/combined accuracy;
- per-relation validation h12 accuracy;
- relation-wise terminal natural-pair survival `S_r(12)`;
- `G = SD_r(log(S_r(12)+eps))`;
- mean log survival `C`;
- survival winner relation;
- survival-winner functional gap `D = Acc_h12(winner) - mean_{j != winner} Acc_h12(j)`.

At epoch 100 record test h0/h12/combined and per-relation test accuracy.

For X32 additionally record, at each transition, the median fraction of state squared norm in the added 16 coordinates. This is secondary and cannot alter the primary classification.

## 8. Fork-equivalence gate

Before any post-fork training, on a fixed diagnostic batch and the held-out natural-pair bank require:

- B16, X32, and P16 h0 logits identical within max absolute error `<= 1e-6`;
- B16, X32, and P16 h12 logits identical within max absolute error `<= 1e-6`;
- X32 first 16 trajectory coordinates match B16 within `<= 1e-6` and added coordinates are zero within `<= 1e-7`;
- P16 trajectory matches B16 within `<= 1e-6`;
- terminal survival vectors and `G` agree within `<= 1e-6`;
- X32/P16 parameter-count relative difference `< 0.05`;
- a fixed diagnostic backward pass gives nonzero gradient norm into the dormant added X32 state-output rows and nonzero gradient norm into the dormant added P16 hidden-unit input rows, confirming that the function-preserving expansion is not a permanently dead subspace.

Any failure is **V0 — fork-equivalence/design failure** and fresh scientific interpretation stops.

## 9. Training-validity gate

At epoch 100 every condition in every fresh family must satisfy:

- combined validation `>= 0.38`;
- h0 validation `>= 0.55`.

Any failure is **V1 — post-fork training validity failure**.

## 10. Bootstrap convention

All primary uncertainty intervals use deterministic 5,000-resample paired family bootstraps over the 12 fresh seeds.

## 11. Primary W test — recurrent-workspace expansion pattern

For X32 minus B16, W is supported only if **all** hold:

1. **Terminal benefit:** mean `Delta h12 >= +0.02` and 95% CI lower bound `> 0`.
2. **Functional concentration strengthens:** mean `Delta D >= +0.10` and 95% CI lower bound `> 0`.
3. **Dynamical selectivity strengthens:** mean `Delta G > 0` and 95% CI lower bound `> 0`.
4. **Encoder-side performance equivalent:** `abs(mean Delta h0) <= 0.02` and the entire 95% CI lies inside `[-0.03, +0.03]`.

The h0 equivalence requirement is essential. Without it, R8-M6 cannot isolate recurrent workspace from encoder-side quality.

## 12. Primary S test — state-dimension specificity versus parameter-matched control

For X32 minus P16, S is supported only if **all** hold:

1. mean `Delta h12 >= +0.015` and 95% CI lower bound `> 0`;
2. mean `Delta D >= +0.05` and 95% CI lower bound `> 0`;
3. mean `Delta G > 0` and 95% CI lower bound `> 0`;
4. `abs(mean Delta h0) <= 0.02` and the entire 95% CI lies inside `[-0.03, +0.03]`;
5. the frozen parameter-count match remains `<5%`.

## 13. Frozen classification

After the two validity gates:

- **W0 — isolated recurrent-workspace account not supported:** W false.
- **W1 — workspace-expanded continuation pattern supported, state specificity not established:** W true, S false.
- **W2 — recurrent-state-dimension-specific workspace effect supported:** W true, S true.
- **V0 — fork-equivalence/design failure.**
- **V1 — post-fork training validity failure.**

W2 is the strongest allowed outcome. It would show that, from an initially identical precommitment function and a fixed 16D encoder output, added autonomous recurrent state dimension causally changes the learned continuation toward better terminal performance and stronger specialization beyond a near-parameter-matched 16D-state control.

## 14. Secondary analyses

Report without changing the frozen classification:

- new-coordinate energy in X32 over recurrent time and training checkpoints;
- terminal-head weight norm on added X32 coordinates;
- winner identity agreement across B16/X32/P16;
- per-relation h12 changes;
- relation-wise correlation between log survival and h12 accuracy;
- number of relations above descriptive h12 accuracy thresholds;
- commitment timing for B16/X32/P16 using the prior rank/winner descriptor;
- whether gains are concentrated in the X32 survival winner or spread across relations;
- epoch at which X32 first diverges from B16/P16 in h12, D, and G.

These are descriptive only.

## 15. Claim boundaries

R8-M6 cannot establish:

- a universal trajectory-information principle;
- strong emergence;
- essential chronology;
- language-model or transformer generalization;
- practical superiority over conventional architectures;
- that state dimension is the only mechanism controlling specialization;
- that Euclidean survival magnitude itself mediates reader usefulness.

A W2 result would be a local causal architecture result about post-encoding recurrent workspace dimension in this synthetic autonomous recurrent system.
