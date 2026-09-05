# R8-M1 Preregistration — Objective and Plasticity Decomposition of Native Selective Survival

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION  
**Parent question:** R8 training-induced native dynamical specialization  
**Primary study:** no latent perturbations, no active observer, no controller  

## 1. Scientific question

The post-reset R8 investigation found a reproducible exploratory pattern in frozen ND-R1 checkpoints: initially near-uniform destruction became strongly relation-selective survival during training, and post-hoc component swaps suggested the trained encoder was necessary for relation-specific placement while matched recurrent training amplified the effect.

R8-M1 asks the simplest unresolved mechanistic question:

> **Which ordinary training signal and which trainable component are necessary for relation-selective native survival to form?**

This experiment does not test whether the behavior is a new universal computational principle and does not use the word “emergent” as an outcome label.

## 2. Guardrails

- Native trajectories are observed without injected perturbation.
- Observer/decoder measurements do not feed back into the dynamics.
- All conditions, thresholds, seeds, checkpoints, and contrasts are fixed before fresh outcomes.
- A failed gate is preserved; no threshold may be changed after outcome inspection.
- Secondary patterns cannot rescue a failed primary contrast.
- Practical implications remain engineering hypotheses.

## 3. Architecture and task

Use the recovered Observer-R2 / ND-R1 core lineage unchanged:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation embedding width 8;
- value embedding width 8;
- encoder GRU hidden width 32;
- latent width 16;
- recurrent map `F: 16 -> 32 -> 16` with GELU then tanh;
- 12 recurrent transitions;
- separate linear h0 and h12 heads for each relation.

Data sizes per seed:

- train: 20,000 memories;
- validation: 2,500 memories;
- test: 5,000 memories reserved and not required for primary training selection;
- natural-pair analysis: 2,048 valid input pairs per relation.

Training:

- fixed 100 epochs;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch 256;
- gradient clipping 1.0;
- deterministic algorithms where available.

## 4. Fresh primary seeds

Fresh R8-M1 seeds are fixed as:

`[11, 37, 71]`

These do not overlap the historical Observer seeds 7/19/43, ALI-N8-R1 seeds 5/17/31, or ND-R1 seeds 13/29/53.

## 5. Critical paired-initialization rule

Within each seed, all five conditions start from **the exact same complete epoch-0 parameter state** and use the same memories, per-epoch input permutations, and batch order.

This paired design isolates training objective/plasticity rather than initialization differences.

No condition is reinitialized independently.

## 6. Training conditions

### J — Joint baseline

Train encoder, recurrent map `F`, h0 heads, and h12 heads.

Loss:

`L_J = mean_r [CE(head0_r(h0), y_r) + CE(headT_r(h12), y_r)]`

This reproduces the original core training objective.

### H0 — h0-only

Train encoder and h0 heads only.

Freeze recurrent map `F` and h12 heads at the shared initialization.

Loss:

`L_H0 = mean_r CE(head0_r(h0), y_r)`

This tests whether static representation learning alone can organize relation placement such that a fixed random recurrence becomes selectively survivable.

### HT — h12-only

Train encoder, recurrent map `F`, and h12 heads.

Freeze h0 heads at initialization.

Loss:

`L_HT = mean_r CE(headT_r(h12), y_r)`

This tests whether terminal recurrent supervision is sufficient without the auxiliary h0 objective.

### FF — joint objective with F frozen

Freeze `F` at the shared initialization. Train encoder, h0 heads, and h12 heads.

Use the same joint loss as J.

This tests whether encoder adaptation to a fixed recurrent field can generate the full specialization effect.

### EF — joint objective with encoder frozen

Freeze relation/value embeddings, encoder GRU, and `to_h` at the shared initialization. Train `F`, h0 heads, and h12 heads.

Use the same joint loss as J.

This tests whether recurrent plasticity can generate relation-specific survival from a fixed initial representation.

## 7. Frozen checkpoints

Save every condition at epochs:

`[0, 2, 10, 20, 40, 60, 100]`

Epoch 0 is the shared identical initialization and must hash identically across all five conditions within a seed.

## 8. Primary native-survival measurement

For each relation `r`, generate 2,048 deterministic valid memory pairs `(x, x')` that differ **only** in relation `r` while all other relations are identical.

Encode and roll out both naturally with no latent intervention.

For transition time `t`:

`S_r(t) = median_pairs ||h_t(x)-h_t(x')|| / (||h_0(x)-h_0(x')|| + eps)`

Terminal relation selectivity:

`G = SD_r( log(S_r(12)+eps) )`

Mean terminal contraction/expansion:

`C = mean_{r,pair} log(S_{r,pair}(12)+eps)`

Record all `S_r(t)` for `t=0..12`.

## 9. Baseline lineage-validity gate

The J condition must show source-like successful training before its mechanistic contrasts are interpreted.

Historical Observer-R2 selected epoch-100 combined validation metric `(mean h0 + mean h12)/2` ranged from approximately 0.4297 to 0.4626.

The frozen validity envelope is therefore:

- J epoch-100 combined validation metric >= **0.38** in every seed; and
- J epoch-100 mean h0 validation accuracy >= **0.55** in every seed.

The 0.38 combined threshold is deliberately below the historical minimum to allow ordinary fresh-seed variation while still rejecting clear training failure. It is frozen before R8-M1 execution.

If either condition fails in any seed, classify R8-M1 as **V — baseline validity failure** and do not promote mechanistic contrasts, though all measurements remain preserved.

## 10. Primary paired contrasts

For each seed and each contrast, use deterministic paired bootstrap over natural-pair indices within relation, 5,000 resamples. The same resampled pair indices are used for both conditions before recomputing relation medians and `G`.

A contrast is supported only if the observed difference is positive in all three seeds and the 95% percentile bootstrap interval lower bound is >0 in all three seeds.

### H1 — terminal recurrent supervision contributes to specialization

Primary contrast:

`G_HT - G_H0 > 0`

If supported, terminal supervision produces more native survival specialization than static h0-only representation learning under matched initialization/data.

### H2 — recurrent-map plasticity contributes beyond encoder adaptation

Primary contrast:

`G_J - G_FF > 0`

If supported, adapting the encoder to a fixed random recurrence is not sufficient to reproduce the full joint specialization.

### H3 — encoder plasticity contributes beyond recurrent adaptation

Primary contrast:

`G_J - G_EF > 0`

If supported, adapting the recurrence to the fixed initial representation is not sufficient to reproduce the full joint specialization.

### H4 — full encoder–recurrence coadaptation

H4 is supported only if **both H2 and H3** are supported in all three seeds.

This is the primary test of the coadaptation interpretation.

## 11. Secondary preregistered analyses

These cannot rescue failed primary hypotheses.

1. **Static-learning selectivity:** `G_H0(epoch100) - G_shared_epoch0`. Tests whether encoder learning alone produces some relation asymmetry through a fixed recurrence.
2. **Terminal-only sufficiency relative to joint:** compare `G_HT` with `G_J`; report difference without an equivalence claim unless the preregistered absolute difference is <=0.10 in every seed.
3. **Winner identity agreement:** compare the highest-survival relation across J, HT, H0, FF, and EF within each seed.
4. **Early recurrent establishment:** Spearman rank correlation between relation survival at transition 2 and transition 12 for every condition.
5. **Training-time onset:** report `G` at all frozen training checkpoints.
6. **Generic state accessibility:** deterministic ridge decoding from h0 and h12 at epoch 100, reported separately from geometric survival.
7. **Native geometry:** path length, endpoint/path efficiency, radius, speed, and consecutive-direction cosine for J only.

## 12. Outcome summary

After the baseline validity gate:

- **M0 — no primary decomposition supported:** H1, H2, H3 all fail.
- **M1 — terminal supervision effect only:** H1 supported; H4 fails.
- **M2 — component-plasticity effect without full coadaptation:** one of H2/H3 supported, but H4 fails.
- **M3 — encoder–recurrence coadaptation supported:** H2 and H3 supported. Report H1 independently as terminal-supervision supported/not supported.

Do not reinterpret M0–M3 as proof of strong emergence.

## 13. Claim boundaries

Even M3 would establish only:

> under this symmetric synthetic recurrent task, full relation-selective native survival depends on plasticity in both the encoder and recurrent map under the tested joint training regime.

It would not establish:

- trajectory information as a universal principle;
- creation of new information;
- chronology as essential;
- practical advantage;
- language-model generalization;
- attractor/chaotic dynamics;
- perturbation necessity;
- a mechanism beyond ordinary gradient-based coadaptation.
