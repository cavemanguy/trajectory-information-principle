# R8-M1 Computational Details

Frozen before fresh-seed execution. These details implement the preregistration without changing its scientific hypotheses.

## Determinism and pairing

- All conditions within a seed clone one shared epoch-0 `state_dict`.
- Training memories, validation memories, pair banks, per-epoch relation-order permutations, and batch orders are shared across conditions within a seed.
- Condition names never enter a training-data or batch-order RNG seed.
- Conditions may differ only through frozen parameter sets and objective terms.

## Trainable parameter sets

- `J`: all parameters trainable.
- `H0`: `rel_emb`, `val_emb`, `enc`, `to_h`, and `head0` trainable; `F` and `headT` frozen.
- `HT`: `rel_emb`, `val_emb`, `enc`, `to_h`, `F`, and `headT` trainable; `head0` frozen.
- `FF`: `rel_emb`, `val_emb`, `enc`, `to_h`, `head0`, and `headT` trainable; `F` frozen.
- `EF`: `F`, `head0`, and `headT` trainable; encoder-side modules frozen.

Frozen parameters are excluded from the optimizer so AdamW weight decay cannot move them.

## Loss scale

For comparability with the recovered core lineage:

- joint losses sum h0 and h12 cross-entropy over relations and divide by `N_REL`;
- single-endpoint losses sum the relevant cross-entropy over relations and divide by `N_REL`.

No extra factor of 1/2 is introduced into the joint objective.

## Checkpoint analysis

At each frozen checkpoint, natural-pair survival is measured on the same pair bank for all conditions.

Primary bootstraps use epoch-100 per-pair terminal survival arrays. For each resample:

1. within each relation, sample pair indices with replacement;
2. use the identical sampled indices for both conditions in a contrast;
3. compute relation medians for each condition;
4. compute `G` for each condition;
5. record the paired difference.

Bootstrap RNG namespace: `R8-M1|seed|contrast_name|bootstrap`.

## Ridge accessibility secondary

- ridge lambda: `1e-3`;
- training features: deterministic training split;
- evaluation features: deterministic validation split;
- fit separate multiclass one-hot linear ridge readout jointly over the 8 relations, matching the ND-R1 diagnostic style;
- report h0 and h12 only at epoch 100 for each condition.

## Native geometry secondary

Native path geometry is measured only for condition `J` on the validation split at epoch 100 and includes path length, endpoint displacement, endpoint/path efficiency, radius, speed, turn cosine, and reversal fraction.

## Result preservation

Per-seed outputs include training histories, checkpoint metrics, pair-survival arrays, bootstrap contrasts, shared-initialization hash, and an environment manifest. Aggregate classification is produced by a separate frozen script.