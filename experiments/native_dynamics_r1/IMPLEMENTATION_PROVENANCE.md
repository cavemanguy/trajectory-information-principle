# ND-R1 Implementation Provenance

**Status:** frozen before primary execution  
**Date:** 2026-09-04

## Recovery conclusion

The exact generated R8–R10 working implementation is not present on the current repository branches inspected for ND-R1. The repository's own Observer R2–R11 summary already records that the complete generated R8–R11 working bundles/raw arrays are not all checked in.

ND-R1 is therefore **not** labeled an exact R8 rerun.

It is an independent fresh-seed reproduction using the closest fully recoverable Observer-core lineage as the implementation base.

## Directly recovered source

Source branch:

`observer-r2`

Primary source file:

`experiments/observer_r2/run_observer_r2.py`

Git blob SHA:

`2d5faac398683a7b7656f91c35423308d899ad32`

Recovered dependency file:

`experiments/observer_r2/requirements.txt`

Git blob SHA:

`d9b804b3a81c3965612d14fd9ec59c95f00ccaeb`

Recovered dependency versions:

- `torch==2.10.0`
- `numpy==2.3.5`

## Recovered architecture

Constants from the R2 source:

- relations: 8
- values per relation: 16
- relation embedding: 8
- value embedding: 8
- encoder GRU hidden width: 32
- recurrent latent width: 16
- recurrent steps: 12

Core structure:

1. relation/value embeddings;
2. GRU encoder over a randomized relation ordering;
3. linear map + `tanh` to 16-dimensional `h0`;
4. recurrent map `F`:
   - Linear(16,32)
   - GELU
   - Linear(32,16)
   - Tanh
5. independent relation heads at `h0` and `h12`.

Recovered initialization:

- embeddings: Normal(0,0.02)
- linear weights: Xavier uniform
- linear biases: zero
- GRU input weights: Xavier uniform
- GRU recurrent gate blocks: orthogonal
- GRU biases: zero

## Recovered task/training constants

- train examples: 20,000
- validation examples: 2,500
- test examples: 5,000
- batch size: 256
- AdamW learning rate: 1e-3
- weight decay: 1e-4
- gradient clipping: 1.0
- original maximum epochs: 100

The recovered R2 source originally used patience-12 early stopping and selected the best validation checkpoint.

## Frozen ND-R1 training choice

ND-R1 differs deliberately at one point already frozen in its preregistration:

> train all fresh primary models for exactly 100 epochs and retain dense checkpoints at 0,1,2,5,10,20,40,60,80,100.

Reason: ND-R1 studies the *training trajectory itself*. Early stopping would censor later training states differently by seed and make the emergence curves difficult to compare.

This is a preregistered reconstruction choice, not a result-driven amendment.

The R2 optimization objective is retained: relation-wise cross-entropy from both `h0` and `h12`, summed over relations and averaged by relation count.

## Data generation

The recoverable R2 task generator draws each memory as eight independent categorical values in `[0,15]` and presents relation/value pairs to the encoder under deterministic RNG-controlled permutations.

ND-R1 retains this task family and split sizes.

Fresh seeds are fixed by the preregistration: 13, 29, 53.

All RNG streams use SHA-256-derived named namespaces beginning with `ND-R1|seed|...` so training data, validation data, test data, permutations, batching, natural-pair banks, and bootstraps are reproducible and separated.

## Natural-pair analysis reconstruction

The historical R10 intervention machinery is not used for the ND-R1 primary endpoint.

Instead, ND-R1 measures the survival of **natural controlled task contrasts**:

- draw a valid base memory;
- copy it;
- change exactly one relation value by a deterministic nonzero modulo-16 offset;
- hold the other seven values fixed;
- encode both using the same relation permutation;
- compare their native trajectories.

This removes injected latent perturbations from the primary experiment while testing the same broad scientific phenomenon: whether different task distinctions survive recurrence differently after training.

## Passive R2-style trajectory measurements

The geometry definitions are based on the recovered R2 implementation:

- `dh = h[t+1]-h[t]`
- speed = `||dh||`
- unit direction = `dh/(||dh||+eps)`
- state radius = `||h[t]||`
- consecutive-direction cosine for turning/reversal analysis

ND-R1 adds path length, endpoint/path-length efficiency, and deterministic ridge linear accessibility as passive measurements only.

## What is not claimed

ND-R1 must not say:

- the exact R8 code was recovered;
- its fresh results are a byte-identical reproduction of R8;
- R10's perturbation intervention is being reproduced;
- natural-pair survival is numerically identical to the historical R10 survival statistic.

The appropriate claim, if successful, is an **independent fresh-seed reproduction of the broader training-emergent selective-preservation phenomenon in the recovered Observer-core lineage**.
