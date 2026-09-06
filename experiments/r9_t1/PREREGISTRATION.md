# R9-T1 Preregistration — Continuously Driven Temporal Substrate

**Frozen before implementation or outcome inspection.**

## Motivation

R8 used a deliberately small autonomous recurrent map after a one-shot encoder. R9-T1 asks whether the most important R8 dynamical signatures survive in a substantially more capable and genuinely temporal setting where the recurrent system receives new input continuously, must learn and retain episode-specific rules, switch between them, ignore distractors, and recover old competence when a previously defined regime returns.

The experiment does **not** force attractors, oscillations, trajectory direction, curvature, phase, Jacobian spectra, or any favored dynamical geometry during training.

A secondary architecture arm tests the user's proposed `RNN -> Transformer -> RNN` sandwich. It is exploratory/engineering-oriented and is not the primary scientific hypothesis.

## Task — episodic regime-key memory

Each episode contains three regime IDs `r in {0,1,2}`. At the start of the episode, each regime is assigned an independent random key `k_r in {0,...,15}`. Keys are resampled every episode and are not global model parameters.

For a DATA value `x in {0,...,15}`, the required answer under active regime `r` is

`y = (x + k_r) mod 16`.

The stream contains five event types:

- `REGIME(r)`: identifies a regime.
- `KEY(k)`: immediately after the first `REGIME(r)` for a regime, defines that regime's episode-specific key.
- `ACTIVATE(r)`: reactivates a previously defined regime but does **not** repeat its key.
- `DATA(x)`: requires prediction of `y` under the currently active regime.
- `DISTRACTOR`: carries no target and must not erase useful memory.

The fixed semantic schedule is:

1. define regime 0 -> data block
2. define regime 1 -> data block
3. reactivate regime 0 -> data block
4. define regime 2 -> data block
5. reactivate regime 1 -> data block
6. reactivate regime 2 -> data block

Within data blocks, deterministic seeded jitter changes block lengths and inserts distractors. Episodes are padded to a common maximum sequence length. The exact generator is frozen in implementation before outcome inspection and must preserve this semantic schedule.

The important distinction is that a returning regime provides only its ID; correct behavior requires retention/retrieval of the earlier episode-specific key.

## Fresh seeds

Eight independent family seeds:

`[2111, 2131, 2153, 2179, 2203, 2221, 2243, 2267]`

Training, validation, test, probe, and intervention streams use deterministic disjoint seed namespaces.

## Architectures

All models receive the same event/payload representation and predict a 16-way class at every DATA step.

### A. Gated Residual Recurrent Network (`GRR`, primary)

A continuously driven custom recurrent cell with hidden size 64. Conceptually:

`c_t = tanh(W_c [LN(h_{t-1}), e_t])`

`g_t = sigmoid(W_g [LN(h_{t-1}), e_t])`

`h_t = LN(h_{t-1} + g_t * (c_t - h_{t-1}))`.

This is a richer recurrent substrate than R8's repeated autonomous MLP: it is input-driven, gated, residual, normalized, and persistent.

### B. Standard GRU baseline

Single-layer hidden size 64.

### C. Causal Transformer baseline

Two causal self-attention blocks, model width 64, four heads, feed-forward width 128.

### D. `RNN -> Transformer -> RNN` sandwich (secondary)

- front GRR contextualizes the raw stream locally and persistently;
- a causal Transformer globally mixes the front-RNN state sequence;
- back GRR converts the globally mixed sequence into a second persistent evolving state used for prediction.

The sandwich uses width 48 for recurrent/Transformer states and two attention blocks. Exact parameter counts are reported; performance differences are not interpreted as pure architectural efficiency unless parameter counts are comparable.

## Training

- PyTorch, Python 3.11.
- AdamW.
- Fixed optimizer-step budget; no outcome-dependent extension.
- Cross-entropy only on DATA events.
- Gradient clipping is permitted and frozen in code before outcomes.
- No auxiliary dynamical loss.
- No trajectory regularizer.
- No explicit memory-state supervision.
- No loss encouraging the R8 tangent/transverse result.

A smoke run may use smaller fixed sizes/steps but cannot inspect scientific outcomes or alter frozen scientific thresholds.

## Held-out task metrics

For every architecture and seed:

- `DATA_ACC`: all held-out DATA events.
- `NEW_EARLY_ACC`: first four DATA events after a first-time regime definition.
- `RETURN_EARLY_ACC`: first four DATA events after `ACTIVATE(r)`.
- `STEADY_ACC`: later DATA events after the first four in each block.
- `LONG_GAP_ACC`: DATA events whose previous useful regime/key event is separated by at least the frozen long-gap threshold and intervening distractors.
- parameter count.

Chance is 1/16 = 6.25%.

## Primary competence gate

`G_TEMPORAL_COMPETENCE` passes iff the primary GRR satisfies all of:

1. mean `DATA_ACC >= 0.85` across 8 seeds;
2. mean `RETURN_EARLY_ACC >= 0.75`;
3. at least 6/8 seeds have `DATA_ACC >= 0.80` and `RETURN_EARLY_ACC >= 0.70`.

This asks whether the richer recurrent substrate genuinely solves the temporal memory/switch/return task. It does not require beating the GRU or Transformer.

## Primary R8-signature transfer diagnostics

These diagnostics are evaluated **only on the trained GRR**, on fresh held-out streams, and only at eligible DATA positions with no regime/key/activation event in the next three updates.

### 1. One-step path-history accessibility

Fit closed-form ridge probes on disjoint probe splits:

- state: `h_t`
- one-step: `[h_t, h_t - h_{t-1}]`
- shuffled-history dimension control: `[h_t, shuffled(h_t-h_{t-1})]`

Primary quantity:

`H_GAIN = ACC(one-step) - ACC(state)`.

`G_HISTORY_TRANSFER` passes iff:

- mean `H_GAIN >= +0.01` absolute accuracy;
- bootstrap 95% CI lower bound > 0;
- positive in at least 6/8 seeds;
- one-step also beats shuffled-history on mean with CI lower bound > 0.

This is an accessibility claim only, not information beyond complete state-plus-map.

### 2. Tangent-versus-transverse recovery in the driven system

At an eligible native state `h_t`, define the local driven displacement

`d_t = F(h_{t-1}, x_t) - h_{t-1}`

and normalized tangent `u_t = d_t / ||d_t||`.

Apply equal-norm perturbations at `h_t`:

- tangent: `+/- rho ||d_t|| u_t`
- four deterministic random directions orthogonal to `u_t`.

Then feed every arm the **identical future input stream** for three recurrent updates.

At horizon 3 measure error magnitude relative to the native driven trajectory, normalized by initial perturbation norm.

`D_REC = retention_tangent - retention_transverse`.

Also compare downstream task-output damage on eligible future DATA events:

`D_FUNC = damage_tangent - damage_transverse`.

Frozen perturbation scales: `rho in {0.05, 0.10, 0.20}`.

`G_ANISOTROPY_TRANSFER` passes iff:

- mean `D_REC >= +0.05`;
- bootstrap 95% CI lower bound > 0;
- positive in at least 6/8 seeds;
- mean `D_FUNC > 0` and positive in at least 6/8 seeds.

A pass means the R8 tangent/transverse anisotropy transfers to a continuously driven recurrent system under matched future inputs. It does not establish a universal phase code.

## Secondary architecture comparisons

These are preregistered but secondary.

### GRR versus GRU

Report paired differences for all task metrics. Do not call GRR superior unless the relevant bootstrap CI excludes zero.

### Sandwich versus Transformer

Primary engineering comparison:

`S_RETURN = RETURN_EARLY_ACC(sandwich) - RETURN_EARLY_ACC(transformer)`.

Classify descriptively:

- `S+`: mean >= +0.015 and bootstrap CI lower > 0;
- `S0`: otherwise / no clear advantage;
- `S-`: mean <= -0.015 and bootstrap CI upper < 0.

Also report overall accuracy and parameter counts. A sandwich gain does not establish novelty; recurrent-attention hybrids already exist in the literature.

## Frozen overall outcomes

- `R9-A`: temporal competence passes, history transfer passes, anisotropy transfer passes.
- `R9-B`: temporal competence passes and exactly one dynamical transfer gate passes.
- `R9-C`: temporal competence passes but neither R8 dynamical transfer gate passes.
- `R9-D`: primary GRR fails temporal competence.
- `R9-X`: invalid/protocol failure.

The sandwich secondary classification is reported separately and cannot change the primary R9 outcome.

## Claim boundaries

Even `R9-A` would support only that specific R8 dynamical signatures generalize from the toy autonomous R8 organism to this richer synthetic, continuously driven gated recurrent substrate. It would not establish:

- that trajectory information exists independently of the complete Markov state and transition map;
- essential chronology;
- a universal phase/tangent code;
- superiority to Transformers;
- novelty of RNN/attention hybrids;
- LLM-scale usefulness;
- generalization to natural language, biological networks, or arbitrary tasks.

A negative result is retained. No post-hoc task redesign or training extension may rescue a failed frozen gate.
