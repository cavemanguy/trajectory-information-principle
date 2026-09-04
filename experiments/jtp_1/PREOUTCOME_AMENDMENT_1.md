# JTP-1 Pre-outcome Amendment 1

**Status:** binding amendment made before JTP-1 outcome inspection  
**Date:** 2026-09-04  
**Supersedes:** the cell-level geometric contrast in the initial JTP-1 preregistration and first implementation commit only  
**Unchanged:** source checkpoints, seeds, analysis memories, trajectory times, epsilon grid, direction bank, temporal shuffle, finite-difference checks, secondary diagnostics, outcome labels, claim boundary

## Why this amendment was necessary

During workflow setup for the first JTP-1 Actions run (`33929010212`), before any `Execute frozen JTP-1 grid` step had begun and before any JTP-1 result was inspected, the aggregate criterion was audited conceptually.

The initial implementation defined its positive geometric advantage as

\[
R_{\rm temporal\ shuffle}-R_{\rm radius\ matched}.
\]

That direction is scientifically wrong for the purpose assigned to the radius-matched control. A temporal shuffle can move a state to a very different trajectory radius. Therefore, making temporal-shuffle separation larger than radius-matched separation a positive criterion can reward ordinary time/radius effects rather than rule them out.

The first run is therefore designated **superseded for inference**. Any outputs it produces are implementation artifacts only and must not be used to classify JTP-1.

## Corrected primary control logic

JTP-1 now uses two deterministic norm-matched controls for every native state \(h_t^{(m)}\):

1. **Same-time radius-matched control**: the nearest-latent-norm state from a different memory at the same trajectory time \(t\).
2. **Cross-time radius-matched control**: the nearest-latent-norm state from a different memory at a different trajectory time.

Both controls match gross latent radius. The same-time control estimates ordinary memory/state-identity variation while holding time fixed. The cross-time control changes trajectory time while matching radius as closely as possible.

The corrected primary geometric advantage is

\[
\Delta_R^{\rm time\mid radius}
=
R_{\rm cross\ time,\ radius\ matched}
-
R_{\rm same\ time,\ radius\ matched}.
\]

The corresponding cosine diagnostic is

\[
\Delta_C^{\rm time\mid radius}
=
C_{\rm same\ time,\ radius\ matched}
-
C_{\rm cross\ time,\ radius\ matched}.
\]

A positive value now has the intended interpretation: after controlling gross radius and generic cross-memory variation, moving to a different trajectory time changes local response geometry more than moving to a different memory at the same time.

## Corrected cell criterion

A `(t, epsilon)` cell is a consensus geometric cell only if **all three seeds** satisfy:

1. the original numerical-stability screen,
2. \(\Delta_R^{\rm time\mid radius}\ge 0.05\),
3. the deterministic 2,000-resample paired bootstrap 95% interval for \(\Delta_R^{\rm time\mid radius}\) has lower bound `> 0`, and
4. \(\Delta_C^{\rm time\mid radius}>0\).

Outcome C and Outcome D retain their original cross-cell requirements, but they now operate on this corrected cell definition. Outcome B retains the preregistered magnitude/contraction diagnostic and uses the corrected maximum normalized advantage.

## Transparency rule

The initial preregistration remains in repository history. This amendment is additive and explicitly records the design error rather than erasing it. The corrected implementation must be committed before the replacement run is interpreted.
