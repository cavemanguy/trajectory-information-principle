# R8-AD4 Preregistration — Motion-Component Decomposition

## Question

R8-AD3 established that, in the mature learned R8 system, a simple linear reader given the current latent state plus the correctly paired one-step displacement

`d_t = h_t - h_{t-1}`

has substantially better task-value accessibility than a reader given `h_t` alone or `h_t` plus a shuffled displacement.

R8-AD4 asks a narrower mechanistic question:

**What part of that one-step displacement carries the accessibility gain?**

The primary decomposition is between displacement **direction** and displacement **magnitude**. Curvature/second-difference and coordinate-level analyses are prespecified follow-ups.

This is a readout-accessibility experiment. It does not test information beyond a complete Markov state-plus-map, causal necessity, or a universal trajectory code.

## Important equivalence

`[h_t, d_t]` and `[h_t, h_{t-1}]` are related by an invertible linear transformation. Therefore R8-AD4 does **not** treat “previous state” and “state plus velocity” as distinct linear mechanisms.

## Engine and lineages

Use the exact `experiments/r8_m10/m7r_base.py` engine and the same 12 deterministic mature lineages used by R8-AD2/R8-AD3:

`[1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986]`

Each lineage is reconstructed deterministically to the same frozen maturity criterion and then frozen. No model weights are changed during AD4.

Reusing the lineages is intentional: AD4 decomposes the already-established AD3 phenomenon rather than asking whether a new set of networks exhibits it.

## Fresh probe data

Use disjoint deterministic AD4 namespaces, not AD3 probe samples:

- train: 12,000 memories
- validation: 4,000 memories
- test: 8,000 memories

Targets are the eight 16-way relation values.

## Native trajectory window

Run the ordinary native trajectory through `h_12`.

Primary first-difference window: `t = 1..11`.

Curvature window: `t = 2..11`, because second difference requires `h_{t-2}`.

For each sample and time:

`d_t = h_t - h_{t-1}`

`m_t = ||d_t||_2`

`u_t = d_t / max(m_t, 1e-8)`

`a_t = d_t - d_{t-1}`

`turn_t = cosine(d_t, d_{t-1})`

## Frozen representations

Primary representations:

- `STATE = h_t`
- `FULL = [h_t, d_t]`
- `SHUFFLED_FULL = [h_t, shuffled(d_t)]`, deterministic independent shuffle within split/time
- `DIRECTION = [h_t, u_t]`
- `MAGNITUDE = [h_t, log(m_t + 1e-8)]`

Prespecified secondary representations:

- `ABS_PATTERN = [h_t, |d_t|]`
- `SIGN_PATTERN = [h_t, sign(d_t)]`
- `FULL_CURVATURE = [h_t, d_t, a_t]`
- `FULL_TURN = [h_t, d_t, turn_t]`

Coordinate attribution is basis-dependent and therefore secondary. At each time, evaluate all 16 `STATE_PLUS_COORD_j = [h_t, d_t[j]]` candidates on validation, choose the best candidate using validation only, and report its untouched test performance. Also choose the four coordinates with highest individual validation scores, fit `STATE_PLUS_TOP4`, and report untouched test performance.

## Probe family

Use the same deterministic closed-form ridge multiclass probe family as AD3:

1. standardize using train mean/std only;
2. append bias;
3. concatenate the eight 16-way targets into 128 one-hot outputs;
4. ridge regression in closed form;
5. choose lambda from `[1e-6, 1e-4, 1e-2, 1, 100]` by validation mean accuracy only;
6. evaluate once on test.

No hyperparameter or feature subset is selected from test outcomes.

## Family summaries

Average each first-difference representation across `t=1..11` and all eight relations.

Define:

`DELTA_FULL = Acc(FULL) - Acc(STATE)`

`DELTA_SHUFFLE = Acc(FULL) - Acc(SHUFFLED_FULL)`

`DELTA_DIR = Acc(DIRECTION) - Acc(STATE)`

`DELTA_MAG = Acc(MAGNITUDE) - Acc(STATE)`

For each lineage with positive `DELTA_FULL`, define retained fractions:

`RET_DIR = DELTA_DIR / DELTA_FULL`

`RET_MAG = DELTA_MAG / DELTA_FULL`

For curvature over `t=2..11`:

`DELTA_CURV = Acc(FULL_CURVATURE) - Acc(FULL)`

`DELTA_TURN = Acc(FULL_TURN) - Acc(FULL)`

## Bootstrap

Use a deterministic 10,000-resample percentile bootstrap over the 12 lineage-level values.

## Frozen gates

### G_FULL_REPLICATION

Pass iff both AD3-style contrasts pass:

1. mean `DELTA_FULL >= +0.005`;
2. bootstrap 95% CI lower bound for mean `DELTA_FULL > 0`;
3. at least 10/12 lineages have `DELTA_FULL > 0`;
4. mean `DELTA_SHUFFLE >= +0.005`;
5. bootstrap 95% CI lower bound for mean `DELTA_SHUFFLE > 0`;
6. at least 10/12 lineages have `DELTA_SHUFFLE > 0`.

### G_DIRECTION_PRESERVES

Pass iff all are true:

1. mean `DELTA_DIR >= +0.005`;
2. bootstrap 95% CI lower bound for mean `DELTA_DIR > 0`;
3. at least 10/12 lineages have `DELTA_DIR > 0`;
4. mean `RET_DIR >= 0.75`;
5. bootstrap 95% CI lower bound for mean `RET_DIR > 0.50`.

### G_MAGNITUDE_PRESERVES

Identical criteria using `DELTA_MAG` and `RET_MAG`.

### G_CURVATURE_ADDS

Prespecified independent gate over `t=2..11`, pass iff:

1. mean `DELTA_CURV >= +0.005`;
2. bootstrap 95% CI lower bound for mean `DELTA_CURV > 0`;
3. at least 10/12 lineages have `DELTA_CURV > 0`.

`FULL_TURN`, `ABS_PATTERN`, `SIGN_PATTERN`, best-single-coordinate, and top-4-coordinate results are secondary and cannot alter the primary decomposition classification.

## Frozen primary classification

If fewer than 12 valid mature lineages or required values are incomplete/nonfinite:

- `D0 — invalid/incomplete`

Else if `G_FULL_REPLICATION` fails:

- `D5 — AD3 one-step accessibility did not replicate on fresh AD4 probes`

Else:

- `D1 — direction-preserving decomposition`: direction passes, magnitude fails
- `D2 — magnitude-preserving decomposition`: magnitude passes, direction fails
- `D3 — direction and magnitude both preserve substantial accessibility`: both pass
- `D4 — full displacement effect not preserved by direction or magnitude alone`: neither passes

`G_CURVATURE_ADDS` is reported separately as a prespecified modifier and does not change D1–D4.

## Claim boundary

A D1 result would support the claim that normalized displacement direction retains most of the established one-step linear-accessibility benefit while scalar speed does not. D2 would support the converse. D3 would indicate substantial redundant/accessibly overlapping contributions. D4 would indicate that the useful linear feature depends on the full signed displacement in a way not captured by either normalized direction or scalar magnitude alone.

A positive curvature gate would show that a second trajectory difference adds linear accessibility beyond the already-established full one-step displacement.

Coordinate-level findings are explicitly basis-dependent. No outcome establishes information beyond the complete state-plus-map, causal necessity, essential chronology, formal trajectory coding, or generalization beyond this synthetic R8 system.