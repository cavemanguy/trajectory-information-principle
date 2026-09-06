# R8-AD3 Preregistration — One-Step Path-History Accessibility

## Question

In the mature learned R8 recurrent system, does adding one step of native trajectory history make task-relevant memory values more linearly accessible than the instantaneous latent state alone?

This experiment tests **readout accessibility**, not information beyond a complete Markov state. For the autonomous map `h_{t+1}=F(h_t)`, future motion is determined by the current state. The one-step history feature used here is

`delta_t = h_t - h_{t-1}`

and `[h_t, delta_t]` is linearly equivalent to `[h_t, h_{t-1}]`. Therefore, a positive result means that recent path history improves simple linear decoding relative to the current state alone; it does not prove that velocity is a separate ontological code, that chronology is essential, or that the full state-plus-map lacks the same information.

## Engine and lineages

Use the exact `experiments/r8_m10/m7r_base.py` engine on the branch under test.

Reuse the 12 R8-AD2 lineage seeds intentionally:

`[1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986]`

Each lineage is deterministically retrained from its seed to the same frozen maturity criterion used by R8-M10/R8-AD2, then frozen. No model weights are updated during the accessibility analysis. Reusing AD2 lineages allows a later descriptive linkage between accessibility and the already-frozen long-run regime labels; that linkage is secondary and not a primary gate.

## Probe data

For each frozen lineage generate disjoint fresh memory/permutation splits not used in core training:

- probe train: 12,000 examples
- probe validation: 4,000 examples
- probe test: 8,000 examples

The eight relation values are the probe targets; each target has 16 classes.

## Native trajectory window

Run the ordinary native trajectory through `h_12`.

The **primary internal window** is `t = 1..11`, excluding the two states directly trained for linear readout (`h_0` and `h_12`).

`t=12` is retained as a prespecified secondary endpoint analysis because `h_12` is directly trained by the native terminal heads and may be near a linear-readout ceiling.

For each `t`, define:

- `STATE = h_t`
- `VELOCITY = delta_t = h_t - h_{t-1}`
- `STATE_VELOCITY = [h_t, delta_t]`
- `STATE_SHUFFLED_VELOCITY = [h_t, shuffled(delta_t)]`, where the shuffle is deterministic and performed independently within each probe split and time step
- `STATE_QUADRATIC = [h_t, {h_t[i] h_t[j] for i<=j}]`

`STATE_QUADRATIC` contains 152 static features versus 32 for `STATE_VELOCITY`, making it a deliberately generous static nonlinear control.

## Frozen probe family

Use deterministic ridge least-squares multiclass probes. For each representation and time step:

1. standardize non-bias features using probe-train mean/std only;
2. append a bias column;
3. encode the eight 16-way targets as one concatenated 128-dimensional one-hot target;
4. fit ridge regression in closed form;
5. choose lambda by validation mean accuracy across the eight relations from the frozen grid

`lambda in [1e-6, 1e-4, 1e-2, 1, 100]`;

6. evaluate the chosen probe once on the untouched test split.

No probe hyperparameter is selected from test outcomes.

## Primary family summaries

For each lineage, average test accuracy over the 11 internal times and eight relations for each representation.

Define:

`DELTA_STATE = Acc(STATE_VELOCITY) - Acc(STATE)`

`DELTA_SHUFFLE = Acc(STATE_VELOCITY) - Acc(STATE_SHUFFLED_VELOCITY)`

The shuffled control has exactly the same dimensionality as the true one-step-history representation and tests whether any gain is specific to correctly paired trajectory history rather than merely extra features/parameters.

## Frozen primary gates

Across the 12 lineage-level paired effects, use a deterministic 10,000-resample percentile bootstrap over lineages.

`G_STATE` passes iff all are true:

1. mean `DELTA_STATE >= +0.005` absolute accuracy (+0.5 percentage points);
2. bootstrap 95% CI lower bound for mean `DELTA_STATE` is > 0;
3. at least 10/12 lineages have `DELTA_STATE > 0`.

`G_SHUFFLE` passes under the same three criteria for `DELTA_SHUFFLE`.

## Frozen classification

- `H1 — one-step path-history accessibility supported`: `G_STATE` and `G_SHUFFLE` both pass.
- `H2 — partial/ambiguous path-history accessibility`: exactly one of `G_STATE`, `G_SHUFFLE` passes.
- `H3 — no preregistered path-history accessibility`: neither gate passes.
- `H4 — invalid/incomplete`: fewer than 12 valid mature lineages or any required probe split/result is incomplete/nonfinite.

## Secondary analyses (not gates)

Report:

- `VELOCITY`-only accuracy;
- `STATE_VELOCITY - STATE_QUADRATIC`;
- per-time (`t=1..11`) accessibility curves;
- the prespecified `t=12` endpoint comparison;
- per-relation effects;
- descriptive results grouped by the already-frozen AD2 regime labels.

No secondary result can rescue a failed primary classification.

## Claim boundary

A positive H1 result would establish that, in this synthetic R8 system, correctly paired one-step native trajectory history makes task values more accessible to a simple linear probe than the instantaneous state alone and than a dimension-matched shuffled-history control.

It would **not** establish information beyond the complete Markov state-plus-map, causal necessity of trajectory history, essential temporal order, a universal trajectory code, or generalization to natural systems/LLMs. A negative result would reject this particular one-step linear-accessibility hypothesis, not the broader possibility that longer paths, nonlinear trajectory descriptors, or causal dynamical interventions matter.
