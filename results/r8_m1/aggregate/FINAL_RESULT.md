# R8-M1 Final Result

**Primary classification:** M3 — encoder–recurrence coadaptation supported

- Baseline validity: **True**
- H1 terminal supervision contributes: **False**
- H2 recurrent-map plasticity contributes: **True**
- H3 encoder plasticity contributes: **True**
- H4 full encoder–recurrence coadaptation: **True**

| Seed | valid | G J | G H0 | G HT | G F-frozen | G encoder-frozen | H1 diff [CI] | H2 diff [CI] | H3 diff [CI] |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 11 | True | 0.8175 | 0.1905 | 1.1711 | 0.1914 | 0.0200 | +0.9806 [+0.9655, +0.9962] | +0.6261 [+0.6129, +0.6402] | +0.7976 [+0.7814, +0.8070] |
| 37 | True | 0.9404 | 0.2857 | 0.0674 | 0.3067 | 0.0172 | -0.2183 [-0.2428, -0.1943] | +0.6337 [+0.6078, +0.6576] | +0.9232 [+0.9058, +0.9331] |
| 71 | True | 0.7458 | 0.0618 | 0.0741 | 0.0615 | 0.0132 | +0.0123 [-0.0025, +0.0250] | +0.6843 [+0.6697, +0.6965] | +0.7326 [+0.7121, +0.7398] |

## Claim boundary

This result concerns ordinary training-induced native survival specialization in the tested synthetic recurrent architecture. No perturbation is part of the primary study. Even M3 would establish coadaptation under this training regime, not a new universal trajectory-information principle or strong theoretical emergence.
