# R8 Native-Dynamics Investigation — Exploratory Checkpoint Audit

**Status:** post-primary exploratory analysis of already-frozen ND-R1 artifacts.  
**Authoritative source run:** GitHub Actions run `33932823763`.  
**Pinned ND-R1 implementation:** `ed80a9fd60daeaf012b5384b2d1fdb0841b8246b`.  
**Seeds:** 13, 29, 53.  

This analysis does **not** change ND-R1's frozen Outcome A and is not a confirmatory result. It reuses the dense frozen ND-R1 checkpoints to investigate the R8 phenomenon without perturbing trajectories.

## Question

R8 historically reported multiphase training: contraction before learning, early readout preparation, recurrent-core dependence, and later directional organization. The current reset asks a narrower native-dynamics question:

> How does initially near-uniform recurrent destruction become relation-selective survival during ordinary training, and which learned component is responsible for that organization?

## 1. Selectivity develops gradually during training

Terminal relation-selectivity `G` was nearly zero at initialization and became large by epoch 100:

| Seed | G0 | G20 | G40 | G60 | G100 |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.018 | 0.125 | 0.270 | 0.565 | 0.828 |
| 29 | 0.025 | 0.078 | 0.435 | 0.657 | 0.811 |
| 53 | 0.016 | 0.259 | 0.539 | 0.601 | 0.618 |

The final identity of the highly preserved relation differed by seed.

## 2. The selectivity is mostly established in the first recurrent transitions

At epoch 100, relation-selectivity across the native trajectory was:

| Seed | after transition 1 | after transition 2 | terminal |
|---:|---:|---:|---:|
| 13 | 0.538 | 0.793 | 0.828 |
| 29 | 0.496 | 0.709 | 0.811 |
| 53 | 0.414 | 0.545 | 0.618 |

Thus the recurrence does not gradually discover the relation ordering over all twelve transitions. Most of the separation between preserved and suppressed relation distinctions appears immediately and then largely persists.

The individual epoch-100 natural-pair survival curves make this concrete. In seed 13, relation 2 is already amplified to about `1.65x` after the first recurrent transition and reaches `2.53x` terminal survival, while most relations fall below `0.52x` after one transition and finish around `0.28–0.41x`. Seed 29 similarly favors relations 0 and 3; seed 53 strongly favors relation 1.

## 3. Frozen component-swap analysis

A post-hoc component-swap diagnostic combined the encoder from one frozen checkpoint with the recurrent map `F` from another checkpoint. Natural valid input pairs were used exactly as in ND-R1; no latent perturbation was introduced.

Full-bank epoch-0/epoch-100 comparison:

| Seed | encoder 0 + F0 | encoder 100 + F0 | encoder 0 + F100 | encoder 100 + F100 |
|---:|---:|---:|---:|---:|
| 13 | G=0.018 | G=0.233 | G=0.014 | **G=0.828** |
| 29 | G=0.025 | G=0.219 | G=0.007 | **G=0.811** |
| 53 | G=0.016 | G=0.320 | G=0.024 | **G=0.618** |

### Interpretation boundary

The trained recurrent map alone does **not** create relation identity/selectivity when fed initialization-era encoded states: `encoder0 + F100` remains near-uniform in every seed.

The trained encoder paired with the untrained recurrent map already creates moderate relation differences, although the untrained recurrence still destroys essentially all distinctions absolutely.

The full trained encoder + trained recurrence produces much larger selectivity than either component swap.

This is consistent with **encoder–recurrence coadaptation**, with the encoder playing the dominant role in assigning relation-specific state-space placement and the trained recurrence amplifying/filtering that placement.

It does not establish a nonlinear or exotic mechanism. Component mismatches are out-of-training-distribution combinations and therefore are diagnostic, not direct causal replacements for a dedicated training ablation.

## 4. The learned recurrent map is strongly state-distribution dependent

A notable component-swap observation is that `F100` applied to `encoder0` states did not reproduce the strong contraction seen on matched trained states. For seeds 13 and 29, the median terminal separation of initialization-era distinctions was roughly `2.3x` their initial separation; seed 53 was approximately neutral/slightly expansive. Yet with the matched trained encoder, most relation distinctions contracted while selected relations were amplified.

Therefore statements such as "the learned recurrence is contractive" must be qualified:

> the trained recurrent system is strongly selective/contractive **on the trained encoded-state distribution**; its behavior is not a uniform global property of the map.

This state-distribution dependence is itself worth preserving as a mechanistic clue.

## 5. Simple initialization predictors did not explain the winner

Across the three available fresh seeds, the eventual highly preserved relation was not consistently predicted by:

- initialization h0 accuracy;
- initialization h12 accuracy;
- initial relation-embedding norm;
- initial h0-head weight norm;
- initial h12-head weight norm;
- initialization terminal survival.

With only three seeds this is weak negative evidence, not proof that initialization has no influence. It does rule out treating an obvious single initialization scalar as the established explanation.

## 6. Encoder-subspace concentration is not yet a complete explanation

Natural-pair h0 difference covariance was examined relation by relation at epoch 100. Some highly preserved relations occupied relatively concentrated low-dimensional difference structure, but this pattern was not consistent enough across all seeds to explain terminal survival by a single top-eigenvalue/effective-rank statistic.

Therefore "the winning relation is simply the lowest-dimensional encoder subspace" is not currently supported.

## Current simplest picture

The best current descriptive model is:

1. initialization recurrence is nearly indiscriminately destructive on the encoded-state distribution;
2. training reorganizes the encoder so task distinctions occupy increasingly relation-specific geometry;
3. recurrent training coadapts to that state distribution;
4. within one or two recurrent transitions, the matched system strongly amplifies a small subset of distinctions and suppresses the rest;
5. which relation becomes favored is seed-dependent in the symmetric synthetic task.

This is **training-induced native dynamical specialization**. Calling it strong theoretical emergence would be premature.

## Next experiment required

The most important simpler explanation to test is ordinary supervision structure:

> Is selective survival merely the expected consequence of optimizing the terminal recurrent readout, or does it arise without terminal supervision?

The next fresh experiment should train matched conditions with no perturbations:

1. original joint h0+h12 objective;
2. h0-only objective;
3. h12-only objective;
4. joint objective with recurrent map frozen at initialization;
5. joint objective with encoder frozen at initialization.

This will directly test whether terminal supervision is necessary/sufficient and whether encoder and recurrent plasticity are individually necessary for the full specialization effect.

No claim from this exploratory audit should be promoted before that fresh preregistered decomposition.