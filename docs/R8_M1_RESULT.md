# R8-M1 Result — Objective and Plasticity Decomposition

**Authoritative workflow run:** `33934781506`  
**Pinned implementation commit:** `b9c07d6b729cdc5c30551ce1de85063161ca1d60`  
**Authoritative result commit:** `2620f2232a27e5ac15dc35141a9fa0d5b70296d0`  
**Fresh seeds:** 11, 37, 71  
**Primary classification:** **M3 — encoder–recurrence coadaptation supported**

R8-M1 was a preregistered, no-perturbation native-dynamics experiment. Within each seed, all five conditions began from the exact same initialized network and used the same data and training order. The experiment asked which ordinary training signal and which trainable component were necessary for relation-selective native survival.

## Frozen primary result

Baseline validity passed in all three seeds.

- H1 — terminal supervision contributes more specialization than h0-only learning: **not supported across seeds**.
- H2 — recurrent-map plasticity contributes beyond encoder adaptation to a frozen recurrent map: **supported in all three seeds**.
- H3 — encoder plasticity contributes beyond recurrent adaptation to a frozen encoder: **supported in all three seeds**.
- H4 — full encoder–recurrence coadaptation: **supported** because H2 and H3 both passed in all three seeds.

| Seed | G joint | G h0-only | G h12-only | G F-frozen | G encoder-frozen | H2 joint-F-frozen | H3 joint-encoder-frozen |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.8175 | 0.1905 | 1.1711 | 0.1914 | 0.0200 | +0.6261 CI [0.6129, 0.6402] | +0.7976 CI [0.7814, 0.8070] |
| 37 | 0.9404 | 0.2857 | 0.0674 | 0.3067 | 0.0172 | +0.6337 CI [0.6078, 0.6576] | +0.9232 CI [0.9058, 0.9331] |
| 71 | 0.7458 | 0.0618 | 0.0741 | 0.0615 | 0.0132 | +0.6843 CI [0.6697, 0.6965] | +0.7326 CI [0.7121, 0.7398] |

The h12-only condition was highly variable across seeds, so terminal recurrent supervision alone is not a stable explanation for specialization.

## Narrow supported statement

> In this symmetric synthetic recurrent architecture under the tested joint training regime, the full magnitude of relation-selective native survival depends on plasticity in both the encoder and recurrent map. Neither adapting the encoder to a frozen initial recurrent map nor adapting the recurrent map to a frozen initial encoder reproduced the joint specialization effect.

This supports ordinary **encoder–recurrence coadaptation** as a mechanism for the R8 specialization phenomenon.

## What this does not establish

R8-M1 does not establish:

- a universal trajectory-information principle;
- creation of new information;
- chronology as essential;
- strong theoretical emergence;
- practical advantage;
- language-model or transformer generalization;
- perturbation necessity;
- a mechanism beyond ordinary gradient-based coadaptation.

## Next question

The next R8 question is the origin of symmetry breaking:

> Why does a statistically symmetric task develop seed-dependent asymmetric dynamical specialization, and can the eventual specialized relation be predicted from the earliest stages of native training?

The next study should remain no-perturbation and should distinguish relation-specific initialization, finite-sample data asymmetry, and optimization-order noise before invoking a more exotic mechanism.
