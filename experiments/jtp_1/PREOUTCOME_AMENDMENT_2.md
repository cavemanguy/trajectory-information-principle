# JTP-1 Pre-outcome Amendment 2

**Status:** binding implementation clarification made before authoritative JTP-1 outcome inspection  
**Date:** 2026-09-04  
**Scope:** Outcome D bounded-regime rule only

A final pre-outcome audit identified two edge cases in the original operational definition of Outcome D.

The intended claim is a **bounded connected trajectory–perturbation regime**. Therefore:

1. the peak consensus cell itself must belong to a 4-neighbor connected consensus component of size at least 4; an unrelated connected component elsewhere on the grid cannot satisfy this condition for an isolated peak, and
2. all four boundary cells used to establish falloff — the minimum- and maximum-epsilon cells at the peak time and the `t=0` and `t=12` cells at the peak epsilon — must pass the all-seed numerical-stability screen.

The corrected Outcome D rule is therefore:

- Outcome C qualifies,
- the peak is interior in both time and epsilon,
- the connected consensus component containing that peak has size `>= 4`,
- all four comparison boundary cells are numerically stable in all three seeds, and
- the peak seed-mean corrected geometric advantage is at least `1.5×` both the larger epsilon-boundary effect and the larger time-boundary effect.

No source checkpoint, analysis sample, perturbation scale, direction, control matching rule, cell-level threshold, bootstrap procedure, Outcome A/B/C rule, or claim boundary is changed by this amendment.

Earlier automatically triggered workflow runs remain superseded for inference. The authoritative run is the first run whose head commit contains both Pre-outcome Amendment 1 and Pre-outcome Amendment 2 plus the corresponding implementation.
