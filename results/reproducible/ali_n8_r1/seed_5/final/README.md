# ALI-N8-R1 Seed 5 — Frozen Evidence Record

Status: **FROZEN — DO NOT MODIFY OR REPLACE**

This directory preserves the compact, human-auditable final evidence record from GitHub Actions workflow run `33540331282` (`ALI-N8-R1 Seed 5 Full`). The final artifact was `ali-n8-r1-seed5-final`, artifact ID `9814060622`, uploaded ZIP SHA-256 `c403818cc6acbb4568d1d6c577d753f28ffe6cbb175ceaf6c7305deaae796e09`.

The run checked out `reproducibility/ali-n8-r1` at commit `5f4f91d7cd5fb90f02abe2d50e1f39bc12a5f3f5`. The frozen core checkpoint SHA-256 was `6ba3d43bdf54aeaeb601a1ace949d22475246a5757943c0aca835a02d679905b` and alpha was `0.21510326862335205`.

The original GitHub Actions artifact remains the authoritative full machine output, including per-example prediction CSVs and logs. This repository record freezes the final summary, native and independent-decoder matrices/counts, query-direction cosine matrix, and runtime environment without rewriting the generated values.

## Procedural caveat

Seed 5 is **not a pristine preregistered confirmation**. As documented in `experiments/ali_n8_r1/PROTOCOL_DEVIATIONS.md`, the earlier standalone seed-5 core run evaluated temporary core heads on the test split before downstream ALI systems and diagnostics were selected. Core checkpoint selection itself used validation only, alpha used training latents only, and no downstream ALI test predictions were produced at that time. The early core-test observation was not used to change the R1 design. Nevertheless, seed 5 must retain this procedural-deviation caveat in every scientific write-up.

## Frozen headline results

- Adaptive `P(m,q)` test accuracy: `0.7226750254631042`
- Query-only `P(q)` test accuracy: `0.25565001368522644`
- Query-blind adaptive `P(m)` test accuracy: `0.5171999931335449`
- Direct `m` control: `0.7494249939918518`
- Direct `F(m)` control: `0.6572999954223633`
- Fixed direction: `0.202674999833107`
- Random direction: `0.1506499946117401`
- Zero response: `0.06404999643564224`
- Direction-only leakage `P(m,q)`: `0.7605500221252441`
- Direction-only leakage `P(m)`: `0.7351499795913696`
- Wrong-memory adaptive accuracy: `0.06382499635219574` versus native `0.7226750254631042`
- Wrong-memory prediction-change rate: `0.9239000082015991`
- `D_native`: `0.19116785714285714`
- `D_decode`: `0.10370357095130853`

These are one-seed results. They must not be promoted to the preregistered multi-seed claim until the remaining primary seeds are executed and mechanically aggregated.
