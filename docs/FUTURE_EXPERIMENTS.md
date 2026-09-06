# Future Experiments

## Priority rule

Before adding a new architecture, extract as much mechanistic information as possible from the existing trained recurrent systems. New engineering experiments should be separated from the current scientific track so that new interfaces do not retroactively explain earlier results.

## Future experiment — Query-conditioned active dynamical interrogation

### Motivation

The long-term engineering idea is to treat a learned recurrent system as something that can be actively interrogated rather than only passively read.

Given latent state `h` and query `q`, a controller proposes a perturbation direction `v(q)` and measures the resulting dynamical response:

`R_h(v) = [F(h + eps v) - F(h - eps v), F^2(h + eps v) - F^2(h - eps v), ...]`.

The response trajectory, rather than only the static latent state, is then used to answer the query.

A closed-loop version may choose later probes from earlier responses:

`v_1 = C(q)`

`r_1 = R_h(v_1)`

`v_2 = C(q, r_1)`

and so on.

### Scientific constraint

Do not force a preferred latent geometry, attractor type, cycle, spectral structure, or trajectory statistic. Train only for functional success and let the system organize its dynamics however it chooses.

### Primary comparison

Compare a matched passive/static reader against the active dynamical reader under equalized capacity/compute where practical.

Required controls should include native/query-conditioned directions, wrong-query directions, shuffled directions, and matched random directions.

A particularly strong result would be active dynamical interrogation succeeding where a matched static reader fails, with causal dependence on the selected probe direction.

### Status

Future experiment only. Do not launch until the present R8/AD series has extracted as much information as possible from already trained systems.

## Near-term scientific priority

Continue with frozen-system diagnostics and causal interventions on the current learned recurrent maps. The immediate question is not whether recurrent systems can have interesting dynamics; that is established. The immediate question is whether the specific ongoing motion, phase, direction, or regime in the current systems contributes functionally useful information beyond what can be reduced to a static state readout.
