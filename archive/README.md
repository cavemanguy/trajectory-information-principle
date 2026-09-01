# Research History Archive

This directory preserves obsolete experiments so development remains inspectable without presenting superseded claims as current evidence.

## Legacy attractor prototype

### Plain English

The first version explored hand-designed trajectories toward points called attractors. It made claims about guaranteed convergence, robustness, information preservation, scalability, applications, and novelty that the included demonstration did not justify. The code remains useful as history because it helped motivate studying responses and trajectories rather than only final latent states.

It is archived—not silently erased—because failed framings and overextended interpretations are part of an honest research record.

### Technical assessment

Files in [`legacy-attractor-prototype/`](legacy-attractor-prototype/) are retained verbatim from the prior root. They are not a validated ALI implementation.

Reasons for superseding them:

- The Lyapunov argument does not prove convergence for the implemented second-order, state-dependent, multi-force dynamics; saying one term "dominates" is not a bound.
- Stopping when velocity is small does not prove the state reached a fixed point.
- Exact recovery of values already stored in `convergence_curves` is lookup-style training-set performance, not general information preservation.
- Input is directly embedded in starting coordinates, so distinguishable trajectories do not demonstrate useful compression or recovery after information loss.
- Robustness, timing, feature importance, scaling, sensor, capacity, and comparison tables lack supporting scripts and raw outputs.
- Universal encoding, computational universality, novelty, and application claims exceed the evidence.
- The old README referenced files and directories absent from the repository.

No current result should cite these files as empirical support. They may be cited as project history or abandoned hypotheses.

## Current direction

The live project focuses on controlled, learned perturbations of frozen latent memory. It does not require attractor convergence and makes no general claim that trajectories preserve initial conditions.
