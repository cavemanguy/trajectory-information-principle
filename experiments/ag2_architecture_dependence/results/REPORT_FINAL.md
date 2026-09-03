# AG2 Final Report

## Classification

**Highest supported level: Level 2 — Cross-task discriminator.**

**Primary outcome: Outcome J — Task dependence dominates the alignability contrast, with a reproducible cross-task architecture difference in correspondence transport.**

AG2 reproduced the AG1 boundary: stage locality is robust across architectures and both controlled tasks, while strong far-stage affine rescue is not.

## Main result

The most important new result is that weak far-stage affine rescue does **not** imply incoherent local dynamics. Task-1 adjacent-stage direction prediction quality (1-NMSE) was approximately:

- Attractor reference: 0.991
- GRU: 0.981
- Leaky RNN: 0.998
- Vanilla RNN: 0.857

GRU and especially leaky RNN therefore support highly predictable local affine transport despite almost absent AG1 far-stage affine rescue.

The strongest cross-task architecture discriminator was individual correspondence retrieval after a global adjacent-stage affine map:

| Architecture | Task 1 | Task 2 |
|---|---:|---:|
| Attractor reference | 85.0% | 70.5% |
| Leaky RNN | 75.3% | 68.2% |
| GRU | 61.4% | 41.5% |
| Vanilla RNN | 49.4% | 40.6% |

On Task 1, correspondence retrieval had a strong system-level association with AG1 affine rescue (Spearman ~0.865). On Task 2 the association was much weaker (~0.430), because the attractor reference itself had only ~2.6% common-protocol affine rescue.

A compact three-predictor model selected from Task-1 associations used correspondence retrieval, transform dispersion, and Jacobian dispersion. It fit Task 1 well but failed badly on Task 2 (R² about -33.7). This is strong evidence against promoting the Task-1 architecture contrast into a task-independent explanatory law.

## Other diagnostics

Task-1 local-transform dispersion sharply separated the attractor condition (~0.31) from GRU/leaky (~1.96–1.98) and vanilla (~3.66), but the attractor rose to ~3.01 on Task 2 as its affine rescue collapsed. This is more consistent with task-conditioned transform coherence than a fixed architectural property.

Jacobian dispersion was not sufficient: leaky RNN had lower Jacobian dispersion than the attractor on both tasks while retaining weak far-stage rescue. Likewise, leaky RNN preserved neighborhoods, CKA similarity, and local correspondence very strongly without recovering the AR4-style distant-reader compatibility.

The simplest successive-re-expression interpretation is therefore too strong. The non-attractor systems, particularly leaky RNN, are not observationally behaving as though representational organization is rebuilt from scratch at every stage. They retain substantial local and relational continuity.

## Controls

C-label permutation reduced the selected C-specific geometry measure from ~0.141 to ~0.031. Breaking example identity in cross-stage training pairs reduced correspondence retrieval from ~0.854 to ~0.065. Parameter count was only weakly associated with rescue, and coarse Y accuracy was saturated.

The independent reproducibility check recomputed the selected central correspondence metric for seed 7 across all four architectures and both tasks. Maximum numerical difference from the primary computation was below floating-point tolerance.

## Narrowest defensible conclusion

Progress-conditioned locality can coexist with substantial local representational continuity. In the tested GRU and leaky RNN systems, adjacent-stage representations are highly predictable by simple affine/local maps and preserve neighborhoods/correspondence, yet distant-stage C-reader compatibility remains weak. The attractor system’s stronger Task-1 affine readability is accompanied by unusually low local-transform dispersion and high correspondence transport, but that same affine-rescue advantage disappears on Task 2. AG2 therefore identifies reproducible architecture differences in transport coherence, especially correspondence preservation, while showing that no single tested dynamical property provides a task-independent explanation of far-stage affine alignability.

No causal architectural claim is supported. Do not claim information destruction/reconstruction, Jacobian or gate causality, universal coherent transport, or that correspondence determines affine rescue.
