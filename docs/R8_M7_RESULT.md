# R8-M7 Result — Reversible Demand Tracking

**Frozen primary classification:** **V — reversible-demand training validity failure**

R8-M7 asked whether one trained recurrent lineage could reversibly reorganize native relation-selective dynamics under an A → B → A terminal-demand schedule.

## Frozen validity outcome

All 12 fresh families completed and all recorded metrics were finite, but every family failed the preregistered epoch-60 baseline competence gate.

The frozen gate required, at epoch 60:

- combined validation >= 0.38;
- h0 validation >= 0.55;
- all three post-baseline conditions complete;
- no NaN/Inf.

Observed aggregate validity:

- baseline gate: failed in 12/12 families;
- completeness: passed in 12/12;
- finite metrics: passed in 12/12.

Therefore the frozen outcome is **V** and the D0/D1/D2 reversible-tracking classifier is not promoted.

## Protocol diagnosis

The common failure pattern indicates that the fixed epoch-60 fork was too early for the competence threshold imposed on this lineage. This is a protocol limitation, not evidence for or against reversible demand-sensitive dynamics.

As a descriptive post-primary example only, seed 207 had at epoch 60:

- combined validation = 0.3150;
- h0 validation = 0.49165.

Under continued training, the same lineage reached at epoch 80:

- combined validation = 0.38005;
- h0 validation = 0.5638.

That example shows why a fixed epoch is a poor proxy for maturity here. It does **not** repair R8-M7 or permit reclassification.

## Preservation rule

R8-M7 remains **V** permanently under its frozen protocol. No threshold or fork-time adjustment is applied post hoc.

A new replication may instead use a preregistered **competence-triggered maturity fork**: train symmetrically until a frozen competence/stability rule is satisfied, then begin the same A → B → A challenge from that first qualifying checkpoint.

## Claim boundary

R8-M7 provides no confirmatory evidence either for or against reversible native-dynamics reorganization because its baseline validity gate failed before the primary test became interpretable.
