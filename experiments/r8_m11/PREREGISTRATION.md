# R8-M11 Preregistration — Recurrent Map Substructure Localization

**Status:** FROZEN BEFORE FRESH-SEED OUTCOME INSPECTION

## Question

> R8-M9 localized most of the persistent matched-demand history effect to the learned recurrent map `F`. Which learned stage inside `F` carries that contribution: the input-facing `16 -> 32` linear stage, the output-facing `32 -> 16` linear stage, or both?

R8-M11 is a mechanistic follow-up to R8-M9. It does not test Luke's off-axis/null-history control and deliberately uses a separate seed set, namespace, branch, and scientific question.

## Inherited architecture and training protocol

Use the same synthetic recurrent architecture and data-generating process as R8-M8/M9:

- 8 statistically symmetric categorical relations;
- 16 values per relation;
- relation/value embedding width 8;
- encoder GRU hidden width 32;
- `to_h` projection to a 16-D autonomous state;
- recurrent map `F = Linear(16,32) -> GELU -> Linear(32,16) -> Tanh`;
- 12 autonomous recurrent transitions after `h0`;
- separate h0 and h12 readers;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch 256, gradient clip 1.0;
- 20,000 train / 2,500 validation / 5,000 test memories;
- natural-pair bank 2,048 per relation.

No new external information enters after h0.

## Fresh family seeds

Twelve fresh families are fixed:

`[839, 856, 872, 891, 907, 926, 944, 963, 981, 1003, 1021, 1042]`

These do not overlap the completed R8-M9 families or the seed set proposed in Luke's off-axis control PR.

Use deterministic namespace `R8-M11` for initialization, data, pair banks, permutations, minibatch order, and bootstrap resampling.

## Maturity trigger

Train each family under the ordinary symmetric objective. Record checkpoints every 10 epochs beginning at epoch 40, hard maximum epoch 400.

A checkpoint is competent iff:

- combined validation >= 0.38; and
- h0 validation >= 0.55.

Maturity `M` is the first checkpoint for which:

1. current and preceding two 10-epoch checks are all competent; and
2. terminal-survival winner and loser are unchanged between the current and immediately preceding checkpoint.

At M define:

- `A` = terminal-survival winner;
- `B` = terminal-survival loser.

If any family fails to reach M by epoch 400, the cross-family result is **V0 — maturity validity failure**.

## Recreate the matched midpoint histories

Fork the exact mature model and optimizer state into two branches.

### A_HISTORY

- lambda 0.00 for 60 epochs;
- lambda 0.25 for 30 epochs;
- lambda 0.50 for 30 epochs.

### B_HISTORY

- lambda 1.00 for 60 epochs;
- lambda 0.75 for 30 epochs;
- lambda 0.50 for 30 epochs.

The terminal weighting is unchanged from M8/M9:

- `w_A = 1 + 3*(1-lambda)`;
- `w_B = 1 + 3*lambda`;
- all other relations weight 1;
- h0 remains symmetric.

Both histories reach the matched midpoint after exactly 120 post-maturity epochs.

Let the complete midpoint parameter states be `A-state` and `B-state`.

## Native organization metric

For relation r:

`L_r = log(S_r(12)+eps) - mean_{j != r} log(S_j(12)+eps)`

and

`Q = L_B - L_A`.

The parent midpoint separation is:

`H_parent = Q(B-state) - Q(A-state)`.

## Gate R — fresh replication of the parent phenomenon

Localization is interpreted only if:

1. mean `H_parent >= +0.50`; and
2. deterministic 5,000-resample family bootstrap 95% CI lower bound > 0.

If validity passes but this gate fails, classification is:

**R0 — matched-midpoint history effect not replicated strongly enough for substructure localization.**

No intra-F claim is promoted under R0.

## Gate F — replicate the R8-M9 recurrent carrier

Before subdividing `F`, repeat the frozen M9 whole-block encoder/recurrent transplant using a common A-state reader:

- E_A + F_A
- E_A + F_B
- E_B + F_A
- E_B + F_B

Define exactly as M9:

`F_total = 0.5 * [(Q(E_A,F_B)-Q(E_A,F_A)) + (Q(E_B,F_B)-Q(E_B,F_A))]`.

Gate F passes iff:

1. mean `F_total >= +0.25`; and
2. bootstrap 95% CI lower bound > 0.

If R passes but F fails, classification is:

**F0 — parent history effect replicated, but the recurrent-map carrier did not replicate strongly enough for intra-F localization.**

This protects M11 from explaining a component-level effect that is absent on its own fresh families.

## Frozen recurrent-map partition

The learned recurrent map is partitioned into only two parameter blocks:

- **F1:** `F.0.weight` + `F.0.bias`, the input-facing `Linear(16,32)` stage;
- **F2:** `F.2.weight` + `F.2.bias`, the output-facing `Linear(32,16)` stage.

`GELU` and `Tanh` have no learned parameters and are unchanged in every hybrid.

Weights and biases are kept together at this stage. R8-M11 does not separately localize weight versus bias, individual hidden units, singular directions, or low-rank subspaces.

## 2 x 2 x 2 substructure transplant

At the matched midpoint, construct all eight combinations:

- encoder background E from A-state or B-state;
- F1 from A-state or B-state;
- F2 from A-state or B-state;
- reader block fixed to the A-state reader in all eight models.

No additional training occurs before measurement.

Denote native organization as:

`Q[e,f1,f2]`, where each source is `A` or `B`.

Exact hashes for E, F1, F2, and reader blocks are recorded and verified for every hybrid.

## Primary F1 and F2 effects

For each family define the input-stage contribution:

`F1_effect = mean over e,f2 of [Q[e,B,f2] - Q[e,A,f2]]`.

With two encoder backgrounds and two F2 backgrounds, this is the average of four paired contrasts.

Define the output-stage contribution:

`F2_effect = mean over e,f1 of [Q[e,f1,B] - Q[e,f1,A]]`.

These definitions satisfy, per family:

`F1_effect + F2_effect = F_total`,

where `F_total` is the average matched whole-F A->B transplant effect across the two encoder backgrounds.

### F1 support criterion

F1 is supported iff:

1. mean `F1_effect >= +0.15`; and
2. bootstrap 95% CI lower bound > 0.

### F2 support criterion

F2 is supported iff:

1. mean `F2_effect >= +0.15`; and
2. bootstrap 95% CI lower bound > 0.

The +0.15 minimum was frozen using the scale of the prior M9 recurrent-map effect (~+0.52) so that a statistically nonzero but very small transplant leakage is not promoted as the carrier.

## F1 x F2 interaction descriptor

For each encoder background compute:

`I12(e) = [Q[e,B,B]-Q[e,A,B]] - [Q[e,B,A]-Q[e,A,A]]`.

Average over encoder backgrounds to obtain `I12` per family.

The interaction is reported as supported descriptively only if:

- the two-sided bootstrap 95% CI excludes zero; and
- `abs(mean I12) >= 0.15`.

Interaction does not override the primary L0-L3 classifications.

## Frozen primary classifications

After validity, R, and F gates:

- **L0 — intra-F localization unresolved:** neither F1 nor F2 criterion passes;
- **L1 — input-stage contribution supported:** F1 passes, F2 does not;
- **L2 — output-stage contribution supported:** F2 passes, F1 does not;
- **L3 — distributed two-stage contribution supported:** both F1 and F2 pass.

Strong wording is always **contribution supported**, not exclusive storage, unless a future equivalence test justifies absence in the other block.

## Secondary descriptors

Report regardless of L-classification once validity permits:

- M, A, B per family;
- parent `H_parent`;
- whole-block `F_total`;
- all eight `Q[e,f1,f2]` values;
- `F1_effect`, `F2_effect`, and `I12` per family;
- descriptive fractions `F1_effect/F_total` and `F2_effect/F_total` when defined;
- survival vector, `G`, `C`, and exact winner for every hybrid;
- h0 and h12 validation accuracy for every hybrid;
- A-relation and B-relation h12 accuracy for every hybrid;
- A-vs-B parameter-difference Frobenius norm for F1 and F2;
- singular-value spectra of `Delta F1.weight` and `Delta F2.weight` as exploratory descriptors only;
- final test-set hybrid metrics only after all fixed training and primary evaluation paths are complete.

The SVD descriptors may motivate a later low-rank causal study but cannot be used to redefine M11's primary localization after outcomes are seen.

## Hybrid compatibility boundary

Cross-layer hybrids may be less functionally compatible than native A/A or B/B recurrent maps. This is scientifically informative but is not a validity failure by itself. Report hybrid h12 performance and native-survival metrics so that a large Q change can be distinguished from simple catastrophic incompatibility.

Only non-finite values, missing blocks, failed hashes, or incomplete execution invalidate the run.

## Execution validity

Every family must:

- reach M;
- reproduce identical model and optimizer fork states at M;
- complete both 120-epoch midpoint histories;
- verify exact E/F1/F2/reader hashes in all eight hybrids;
- compute all preregistered native metrics;
- contain no NaN/Inf in primary metrics or validation measures.

Any mature-family violation yields **V1 — post-maturity substructure execution failure**.

## Strongest allowed conclusions

Under L1:

> Within this tested architecture, the dominant localized contribution inside the recurrent map follows the input-facing 16->32 learned stage under the frozen transplant analysis.

Under L2:

> Within this tested architecture, the dominant localized contribution inside the recurrent map follows the output-facing 32->16 learned stage under the frozen transplant analysis.

Under L3:

> Within this tested architecture, both learned linear stages of the recurrent map contribute to the persistent history-dependent organization.

R8-M11 cannot establish that an identified block is a unique memory substrate, that individual neurons or singular directions are causal, formal bistability/hysteresis, information beyond the complete state, essential chronology, universal trajectory computation, or generalization beyond this synthetic architecture.

## Permanent interpretation rule

M8 established persistent operational history dependence. M9 localized most of its causal contribution to the recurrent map and ruled out history-specific AdamW state as necessary over the tested hold. M11 is allowed only to refine that recurrent-map mechanism one level deeper; it may not retroactively strengthen M8 or M9 beyond their frozen claims.
