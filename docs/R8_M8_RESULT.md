# R8-M8 Result — Demand Hysteresis and Regime Persistence

**Frozen primary classification:** **Y3 — persistent history-dependent regime separation supported**

All 12 fresh families passed the preregistered maturity and execution-validity gates.

R8-M8 asked whether opposite controlled demand histories can leave the same mature recurrent lineage in different native dynamical organizations under the same current demand and matched cumulative post-maturity training duration, and whether that separation persists under prolonged identical demand.

## Primary result

All three frozen primary gates passed.

### H — matched-midpoint history effect

At `lambda = 0.50`, the A-history and B-history branches had the same current demand and each had received 120 post-maturity training epochs. The preregistered history-separation metric was:

`H_mid = Q_B-history(0.50) - Q_A-history(0.50)`

Across 12 fresh families:

- mean `H_mid = +1.331376`
- 95% bootstrap CI `[+0.643896, +2.056910]`

This passed the frozen requirement that mean `H_mid >= +0.50` with CI lower bound greater than zero.

### L — signed sweep-loop separation

Across matched demand levels `lambda = [0, 0.25, 0.50, 0.75, 1]`, the signed trapezoidal loop area was:

- mean `AREA = +1.239572`
- 95% CI `[+0.588703, +1.886889]`

This passed the frozen loop criterion.

Pointwise history separation `H(lambda)` was positive at every tested demand level:

- `lambda=0.00`: mean `+1.112606`, 95% CI `[+0.415777, +1.845017]`
- `lambda=0.25`: mean `+1.257094`, 95% CI `[+0.563852, +1.972515]`
- `lambda=0.50`: mean `+1.331376`, 95% CI `[+0.652921, +2.048968]`
- `lambda=0.75`: mean `+1.250684`, 95% CI `[+0.634838, +1.911050]`
- `lambda=1.00`: mean `+1.125668`, 95% CI `[+0.643742, +1.698221]`

### P — persistence under prolonged identical demand

The two midpoint states were cloned and then trained for an additional 120 epochs under identical `lambda=0.50` demand.

After the hold:

- mean `H_hold120 = +1.207673`
- 95% CI `[+0.499004, +1.924446]`

This passed the frozen persistence criterion requiring mean `H_hold120 >= +0.25` with CI lower bound greater than zero.

The descriptive retention fraction was:

- `H_hold120 / H_mid = 0.9071`

So about 90.7% of the original matched-midpoint separation remained after another 120 epochs under identical current demand.

The hold curve remained positive throughout:

- +30 epochs: mean `H = +1.285501`
- +60 epochs: mean `H = +1.259131`
- +90 epochs: mean `H = +1.228385`
- +120 epochs: mean `H = +1.207673`

## Secondary descriptors

- Mean maturity epoch: `98.33`, range `90–110`
- Exact opposite midpoint survival winners: `3/12`
- Mean midpoint latent distance: `h0 = 0.3452`, `h12 = 1.5200`
- Functional midpoint contrast: mean `+0.370867`, 95% CI `[+0.150931, +0.611033]`

The exact-winner result is an important boundary. Persistent history dependence does not require every family to flip into categorically opposite specialist identities. The stronger supported result is continuous separation in native organization under matched present demand.

## Interpretation

The strongest defensible interpretation is:

> **Within this symmetric synthetic autonomous recurrent system, opposite controlled demand histories can leave the same mature lineage in persistently different native dynamical organizations under the same current functional demand and matched training duration, with a direction-consistent sweep separation that survives a prolonged identical-demand hold.**

This directly strengthens the earlier R8-M7R/M7I suggestion of history-sensitive resistance. R8-M8 was designed specifically to distinguish persistent path dependence from a simple short-lived lag, and the separation retained about 90.7% of its matched-midpoint magnitude after 120 additional identical-demand epochs.

## Claim boundary

R8-M8 supports **operational persistent history dependence** in the tested learned recurrent dynamics.

It does **not** establish:

- mathematical bistability;
- formal thermodynamic or dynamical-systems hysteresis in the strict theoretical sense;
- conscious choice or intentional regime selection;
- a universal trajectory-information principle;
- information beyond the complete current state;
- essential chronology as an independent information channel;
- generalization to language models, transformers, biological systems, physical systems, or naturalistic tasks.

The result should therefore be described as **persistent history-dependent regime separation** or **hysteresis-like operational path dependence**, not as proof of formal hysteresis.

See also:

- `../experiments/r8_m8/PREREGISTRATION.md`
- `R8_M7R_RESULT.md`
- `R8_M7I_RESULT.md`
- `R8_M7I_POSTRUN_AUDIT.md`
- `CURRENT_CLAIMS.md`
- `EVIDENCE_LEDGER.md`
