# R8-M7 Design Note — Reversible Demand Tracking

R8-M7 is the next planned hard test after R8-M5R. It is not yet a result.

The motivating question is whether native specialization merely varies across optimization paths, or whether the **same trained lineage can reorganize its internal dynamics when functional demand changes**.

Planned core manipulation:

1. Train the ordinary 16-D lineage symmetrically until an established natural specialist exists.
2. Define **A** automatically as the baseline survival winner and **B** as the baseline survival loser, using a frozen rule before any intervention outcomes are inspected.
3. Fork the identical trained checkpoint into matched continuations.
4. In the primary branch, apply terminal demand in the schedule **A → B → A**.
5. Compare with a fixed-A terminal-demand control and an A→B→A h0-demand control using identical data, minibatch order, architecture, and training duration.

The primary evidence should concern native relation-selective survival and reversibility/reassignment. No hidden-state perturbation is part of the primary observation.

The experiment must be preregistered and frozen before fresh-seed outcomes are exposed.
