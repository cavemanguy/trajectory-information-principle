# R8-M7I Post-Run Audit — Exploratory Mirror Diagnostics

**Status:** POST-PRIMARY ONLY. This document does not alter the frozen R8-M7I outcome **V0 — baseline lineage reproduction failure**.

## Why V0 occurred

R8-M7I required exact paired reuse of the R8-M7R baseline lineages. Every family reproduced its maturity epoch and A/B identities, but four families missed the preregistered baseline-Q tolerance of 1e-5 by small numerical amounts. Therefore the frozen classifier correctly stopped before I0/I1/I2.

The failed absolute Q differences were approximately:

- seed 230: 4.31e-4
- seed 263: 1.42e-5
- seed 313: 1.33e-5
- seed 346: 4.47e-5

This pattern is consistent with numerical reproducibility jitter rather than a different qualitative baseline lineage, but that interpretation is post-run and cannot rescue the primary result.

## Exploratory completed-mirror pattern

All 12 B→A→B mirror schedules nevertheless completed, so the finished artifacts were summarized diagnostically while preserving V0.

Using the same Q = L_B - L_A convention, where positive Q favors B:

- immediate B phase Q_B1: mean **-1.33834**, exploratory 95% bootstrap CI **[-1.99091,-0.70473]**
- B→A shift: mean **-0.47974**, CI **[-0.60482,-0.37317]**
- middle A phase Q_A: mean **-1.81808**, CI **[-2.43975,-1.19536]**
- A→B return shift: mean **+0.32090**, CI **[+0.15444,+0.50541]**
- final B phase Q_B2: mean **-1.49718**, CI **[-2.22720,-0.75112]**
- Q_B2 minus maturity baseline Q: mean **+0.79928**, CI **[+0.16181,+1.48176]**

Exact-winner descriptors:

- B exact winner after immediate B phase: **1/12**
- A exact winner during middle A phase: **12/12**
- B exact winner after final B phase: **2/12**
- exact B→A→B winner sequence: **1/12**

Thus the mirror data do not show free specialist reassignment even descriptively. Demand moves the dynamics, but the established A-favored organization usually remains dominant.

## Exploratory functional response

Demanded-relation h12 accuracy changed substantially:

- B gain from maturity baseline to immediate B phase: mean **+15.96 pp**, exploratory 95% CI **[+6.57,+26.32] pp**
- A gain from first B phase to middle A phase: mean **+18.68 pp**, CI **[+11.37,+27.19] pp**
- B gain from middle A phase to final B phase: mean **+18.38 pp**, CI **[+7.32,+30.64] pp**

The prespecified control-style contrasts, evaluated only as post-primary diagnostics because V0 blocked formal classification, were also directionally strong:

- MIRROR versus FIXB during the middle A phase: mean Q difference **-0.55694**, exploratory 95% CI **[-0.75446,-0.39551]**
- terminal-demand mirror amplitude versus H0MIRROR: mean difference **+0.64479**, CI **[+0.48678,+0.83027]**

## Immediate versus delayed B challenge

R8-M7R first reinforced the already-established A specialist for 40 epochs before applying B demand. R8-M7I instead challenged B immediately at maturity.

The paired exploratory comparison was:

- immediate-B Q in R8-M7I minus delayed-B Q in R8-M7R: mean **+0.33955**, 95% bootstrap CI **[+0.16984,+0.51960]**

This is consistent with the hypothesis that the extra A-reinforcement phase in R8-M7R increased resistance to later displacement. It does **not** establish formal hysteresis, and it cannot alter R8-M7I V0.

## Current interpretation

The combined R8-M7R and post-primary R8-M7I pattern is most consistent with:

> **Demand-sensitive but history-dependent dynamical organization: terminal demand can bend native specialization and substantially improve the demanded relation's function, while an established dominant dynamical regime is difficult to fully overwrite.**

This remains a local synthetic-system observation. It does not establish conscious choice, universal trajectory computation, strong emergence, formal hysteresis, essential chronology, or generalization to language models or natural systems.
