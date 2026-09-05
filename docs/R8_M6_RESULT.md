# R8-M6 Result — Isolated Recurrent Workspace

**Frozen primary classification:** **W0 — isolated recurrent-workspace account not supported**

R8-M6 began from a common 16-D lineage model at the epoch-20 fork and compared ordinary 16-D continuation (B16), an exact function-preserving 32-D recurrent-workspace expansion (X32), and a near-parameter-matched wider-transition 16-D control (P16).

## Validity

- Fork equivalence: **True**
- Cross-condition training validity: **True**
- X32/P16 maximum relative parameter-count difference: **0.000997**

## Frozen contrasts

### X32 minus B16

- h12: **+0.057202**, 95% CI **[+0.046462, +0.070179]**
- D: **+0.010931**, 95% CI **[-0.111396, +0.143362]**
- G: **+0.026908**, 95% CI **[-0.005174, +0.058726]**
- h0: **-0.025208**, 95% CI **[-0.033465, -0.018366]**

### X32 minus P16

- h12: **+0.033742**, 95% CI **[+0.018565, +0.048579]**
- D: **-0.061914**, 95% CI **[-0.159122, +0.034578]**
- G: **+0.063998**, 95% CI **[+0.027948, +0.098449]**
- h0: **-0.026060**, 95% CI **[-0.035123, -0.018177]**

The preregistered h0-equivalence criteria failed for X32 versus both B16 and P16. Therefore the terminal h12 improvement cannot be attributed cleanly to isolated recurrent workspace under the frozen R8-M6 rule.

## Preserved interpretation

R8-M6 establishes a useful negative boundary: a function-preserving post-encoding workspace expansion can improve terminal h12 performance, including beyond the parameter-matched P16 control, but continued training changes the encoder-side representation enough that the isolated-workspace causal claim is not supported.

This does **not** overturn R8-M4, R8-M5, or later R8-M5R. It does not establish that specialization must increase or decrease with capacity.

## Claim boundary

R8-M6 is local to one symmetric synthetic autonomous recurrent architecture. It does not establish a universal trajectory-information principle, strong emergence, essential chronology, language-model generalization, or that Euclidean survival magnitude itself mediates reader usefulness.
