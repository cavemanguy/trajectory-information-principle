# R8-M7I Result — Inverted-Demand Same-Lineage Mirror

**Frozen outcome:** V0 — baseline lineage reproduction failure.

R8-M7I reused the same 12 R8-M7R family seeds, deterministic seed namespace, data generator, initialization, minibatch ordering, maturity rule, and native A/B identities, then inverted the post-maturity demand schedule from A→B→A to B→A→B.

The frozen validity rule required every family to reproduce the archived R8-M7R baseline exactly in maturity epoch M, A identity, B identity, and baseline Q within absolute tolerance 1e-5 before any I0/I1/I2 scientific classification could be promoted.

Four families failed only the baseline-Q tolerance gate:

- seed 230: same M=100, A=1, B=5; |ΔQ| ≈ 4.31e-4
- seed 263: same M=100, A=1, B=3; |ΔQ| ≈ 1.42e-5
- seed 313: same M=100, A=0, B=5; |ΔQ| ≈ 1.33e-5
- seed 346: same M=100, A=3, B=6; |ΔQ| ≈ 4.47e-5

All four reproduced the same maturity epoch and the same A/B identities. All 12 families otherwise completed the mirror schedules with finite outputs and valid fork identity.

Because the preregistered baseline-reference gate failed, the formal result remains:

> **V0 — baseline lineage reproduction failure.**

No I0/I1/I2 classification is promoted, and the tolerance is not repaired after outcome inspection.

See `R8_M7I_POSTRUN_AUDIT.md` for explicitly post-primary diagnostic analysis of the completed mirror trajectories.

## Claim boundary

R8-M7I does not establish or refute reversible specialist reassignment. Its frozen primary outcome is a protocol-validity failure caused by the preregistered exact baseline-Q reproduction criterion. Post-primary mirror patterns are preserved as exploratory evidence only and cannot replace V0.
