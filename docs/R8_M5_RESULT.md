# R8-M5 Result — Capacity Allocation Test

**Frozen primary classification:** **C0 — simple capacity-allocation account not supported**

## Primary result

All twelve fresh families passed the cross-capacity training-validity gate.

The preregistered state-capacity account predicted that widening the autonomous state from 16D to 32D would improve terminal performance while reducing one-relation dynamical/functional concentration. The performance prediction was supported, but the redistribution predictions were reversed.

Relative to B16, S32 produced:

- mean h12 test change **+0.055544**, 95% CI **[+0.038769, +0.074646]**;
- mean survival-winner performance-gap change **+0.256655**, 95% CI **[+0.113869, +0.385850]**;
- mean dynamical selectivity `G` change **+0.098347**, 95% CI **[+0.024405, +0.180746]**;
- mean h0 test change **+0.346529**, 95% CI **[+0.337244, +0.358145]**.

Thus wider state capacity improved both encoder-side and terminal performance but made the relation-selective specialization **stronger**, not weaker.

The near-parameter-matched P16 control retained a 16D state while widening the recurrent MLP. P16 and S32 differed in total trainable parameter count by at most **2.31%**, satisfying the frozen parameter-match design. Relative to P16, S32 still had greater specialist performance concentration (`Delta D = +0.143112`, 95% CI `[+0.040988, +0.250720]`) and greater dynamical selectivity (`Delta G = +0.126627`, 95% CI `[+0.069081, +0.186493]`).

## Interpretation

R8-M5 rejects the simple hypothesis that the R8 specialization is merely a compensatory response to an overly tight 16D state bottleneck.

A defensible descriptive interpretation is instead:

> **Increasing state dimension can improve performance while allowing stronger relation-selective dynamical/functional specialization in this synthetic autonomous recurrent system.**

The experiment does not determine whether the stronger specialization is caused specifically by the larger recurrent workspace because S32 also substantially improved the encoder-side `h0` representation. That unresolved mediation motivates a new isolated follow-up rather than a reinterpretation of C0.

## Claim boundary

R8-M5 does not establish that larger states are generally better, that specialization is a universal routing mechanism, that Euclidean survival magnitude itself mediates reader usefulness, or that the result generalizes beyond this symmetric synthetic recurrent system.

A failed explanation is not the same as a failed phenomenon.
