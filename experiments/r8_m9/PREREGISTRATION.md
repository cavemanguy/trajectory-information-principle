# R8-M9 Preregistration — Component Localization of Persistent History

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION

## Question

> After opposite demand histories create the persistent matched-demand separation established by R8-M8, which trained component carries that separation: the encoder-side representation, the autonomous recurrent map, their joint organization, or optimizer state?

R8-M9 is a mechanistic follow-up to R8-M8. R8-M8 produced **Y3 — persistent history-dependent regime separation supported** in the tested 16-D autonomous recurrent system. M9 does not re-test the broad claim with another sweep. It first recreates the matched midpoint history effect on fresh families and then performs component transplants and optimizer-state controls at the formed midpoint.

## Architecture and inherited machinery

Use the same R8 architecture and data-generating process as R8-M8:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embeddings of width 8;
- encoder GRU hidden width 32;
- `to_h` projection to a 16-D autonomous state;
- recurrent map `F: 16 -> 32 -> 16` with GELU/Tanh;
- 12 autonomous recurrent transitions after h0;
- separate h0 and h12 readers;
- AdamW, lr `1e-3`, weight decay `1e-4`;
- batch 256; gradient clip 1.0;
- 20,000 train / 2,500 validation / 5,000 test memories;
- natural-pair bank 2,048 per relation.

No new external information enters after h0.

The component partition is frozen as:

- **Encoder block E:** `rel_emb`, `val_emb`, `enc`, `to_h`;
- **Recurrent block F:** `F`;
- **Reader block R:** `head0`, `headT`.

The native organization metric Q depends only on E and F, not on R.

## Fresh family seeds

Twelve fresh families are fixed:

`[631, 648, 664, 681, 699, 716, 733, 751, 768, 784, 802, 821]`

Use a new deterministic namespace `R8-M9` for data, initialization, pair banks, presentation permutations, and minibatch ordering.

## Maturity trigger

Train each family under the ordinary symmetric objective. Evaluate every 10 epochs beginning at epoch 40, hard maximum epoch 400.

A checkpoint is competent iff:

- combined validation >= 0.38; and
- h0 validation >= 0.55.

Maturity M is the first checkpoint for which:

1. the current and preceding two 10-epoch checks are all competent; and
2. terminal-survival winner A and loser B are unchanged between current and immediately preceding checkpoint.

If any family fails to reach M by epoch 400, the cross-family frozen outcome is **V0 — maturity validity failure**.

At M define A = survival winner and B = survival loser. A and B must be distinct.

## Recreate the R8-M8 matched midpoint histories

Fork the exact M model and optimizer state into two histories.

### A_HISTORY

- lambda=0.00 for 60 epochs;
- lambda=0.25 for 30 epochs;
- lambda=0.50 for 30 epochs.

### B_HISTORY

- lambda=1.00 for 60 epochs;
- lambda=0.75 for 30 epochs;
- lambda=0.50 for 30 epochs.

The terminal h12 weighting is exactly the M8 rule:

- `w_A = 1 + 3*(1-lambda)`
- `w_B = 1 + 3*lambda`
- every other relation weight = 1
- h0 remains symmetric.

Both histories therefore reach lambda=0.50 after exactly 120 post-maturity epochs.

Let the two complete midpoint parameter states be A-state and B-state.

## Native organization metric

For relation r:

`L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)`

and

`Q = L_B - L_A`.

Positive Q favors the baseline loser B; negative Q favors baseline winner A.

Define the replicated parent history separation:

`H_parent = Q_B-state - Q_A-state`.

## Phenomenon-replication gate R

Mechanistic localization is interpreted only if the fresh M9 families reproduce the M8 matched-midpoint phenomenon:

1. mean `H_parent >= +0.50`; and
2. deterministic 5,000-resample paired-family bootstrap 95% CI lower bound > 0.

If validity passes but R fails, the frozen classification is:

**R0 — persistent-history midpoint phenomenon not replicated strongly enough for localization.**

No component-localization claim is promoted under R0.

## 2×2 encoder/recurrent transplant

At the matched midpoint, construct four models by exact parameter-block transplantation:

- `Q_AA`: E from A-state, F from A-state;
- `Q_AB`: E from A-state, F from B-state;
- `Q_BA`: E from B-state, F from A-state;
- `Q_BB`: E from B-state, F from B-state.

For native Q evaluation, all four receive the same frozen A-state reader block. Reader choice cannot change Q; a common reader avoids accidental implementation differences.

Exact block hashes are recorded before evaluation. No additional training occurs before these four native measurements.

Define the encoder contribution:

`E_effect = 0.5 * [(Q_BA - Q_AA) + (Q_BB - Q_AB)]`

and recurrent contribution:

`F_effect = 0.5 * [(Q_AB - Q_AA) + (Q_BB - Q_BA)]`.

By construction, `E_effect + F_effect = Q_BB - Q_AA = H_parent` for each family.

Define the 2×2 interaction descriptor:

`I_EF = (Q_BB - Q_AB) - (Q_BA - Q_AA)`.

Interaction is descriptive unless its two-sided bootstrap CI excludes zero and `abs(mean I_EF) >= 0.25`; it does not override the component classifications below.

### Encoder contribution criterion E

E is supported iff:

1. mean `E_effect >= +0.25`; and
2. bootstrap 95% CI lower bound > 0.

### Recurrent contribution criterion F

F is supported iff:

1. mean `F_effect >= +0.25`; and
2. bootstrap 95% CI lower bound > 0.

## Optimizer-state persistence test O

Optimizer state cannot affect the immediate midpoint Q, but it could maintain the separation during continued training.

From the untouched full A-state and B-state midpoint models create three matched lambda=0.50 hold pairs, each trained for 120 additional epochs with identical data/presentation/minibatch schedules:

1. **INHERITED:** each model keeps its own history-specific AdamW state;
2. **RESET:** both models receive newly initialized AdamW state with the same hyperparameters;
3. **CROSSED:** A-state model receives B-history optimizer state and B-state model receives A-history optimizer state, mapped by identical parameter ordering.

Record Q at +30, +60, +90, +120.

Primary optimizer-necessity descriptor:

`H_reset120 = Q_B_RESET(+120) - Q_A_RESET(+120)`.

O is supported iff:

1. mean `H_reset120 >= +0.25`; and
2. bootstrap 95% CI lower bound > 0.

When O passes, optimizer state is **not necessary** for 120-epoch persistence under this reset intervention. When O fails, no claim is made that optimizer state is sufficient; only that necessity was not ruled out.

The inherited and crossed holds are prespecified secondary comparisons. Report `H_inherited120`, `H_crossed120`, and their full decay curves.

## Reader-transfer secondary analysis

Reader parameters are not part of native Q, so reader localization is secondary and cannot change the primary component classification.

At the untrained midpoint, evaluate h12 relation-wise accuracy for:

- A dynamics (E_A,F_A) with reader A and reader B;
- B dynamics (E_B,F_B) with reader A and reader B;
- both cross-hybrid dynamics with reader A and reader B.

Report overall h12 accuracy, A-relation accuracy, B-relation accuracy, and the lambda=0.50 weighted h12 accuracy for every dynamics/reader combination.

This asks whether functional differences follow the history-specific dynamics, the reader, or their compatibility. It does not redefine the native history effect.

## Execution validity

Every family must:

- reach M;
- fork identical A_HISTORY/B_HISTORY starting states and optimizers at M;
- complete both 120-epoch histories;
- reproduce exact block hashes in every transplant;
- construct all four 2×2 E/F combinations;
- complete INHERITED, RESET, and CROSSED holds through +120;
- contain no NaN/Inf in losses, Q, survival, or validation metrics.

Any post-maturity violation yields **V1 — post-maturity localization execution failure**.

## Frozen component classifications

After validity and R:

- **C0 — component localization unresolved:** neither E nor F criterion passes;
- **C1 — encoder-carried contribution supported:** E passes, F does not;
- **C2 — recurrent-map-carried contribution supported:** F passes, E does not;
- **C3 — distributed encoder + recurrent contribution supported:** both E and F pass.

Optimizer result O is reported orthogonally to C0–C3.

Examples of allowed combined wording:

- `C3 + O: distributed encoder/recurrent contribution; optimizer state not necessary for 120-epoch persistence.`
- `C2, O failed: recurrent contribution supported; optimizer-state necessity not ruled out.`

## Secondary descriptors

Report regardless of classification:

- M, A, B per family;
- parent `Q_AA`, `Q_BB`, `H_parent`;
- all four transplant Q values;
- `E_effect`, `F_effect`, `I_EF` per family;
- relative fractions `E_effect/H_parent` and `F_effect/H_parent` when defined;
- survival vectors, G, C, and exact winner for all transplant states;
- h0/h12 latent distances between parent histories;
- reader-transfer functional matrix;
- inherited/reset/crossed optimizer hold curves at +30/+60/+90/+120;
- exact winner identity during each hold;
- test-set metrics only after all fixed training/evaluation decisions are complete.

## Strongest allowed conclusions

Under C3 + O, the strongest allowed conclusion is:

> In this tested synthetic recurrent system, the persistent matched-demand history effect is distributed across encoder-side representation and recurrent-map parameters, and the separation remains after optimizer state is reset, showing that AdamW history is not necessary for persistence over the tested 120-epoch hold.

Under C1 or C2, wording must say **contribution supported**, not that the other component contains no history unless its effect is additionally bounded by a separately justified equivalence test.

R8-M9 does not establish molecular/biological memory, formal bistability, formal thermodynamic hysteresis, a universal trajectory-information principle, essential chronology, information beyond the complete state, or generalization beyond this synthetic architecture.

## Permanent interpretation rule

A component transplant localizes causal contribution to the learned parameter partition under this architecture. It does not prove that the same partition is the unique mechanistic carrier in other architectures or tasks.
