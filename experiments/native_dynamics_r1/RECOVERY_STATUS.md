# ND-R1 Source Recovery Status

ND-R1 is intentionally separated from the historical R8–R10 result record.

## What is preserved directly

The repository currently preserves:

- the Observer R2–R11 scientific summary and claim boundaries;
- compact R11 preregistration/result records;
- reproducible R2 core lineage artifacts and code paths;
- historical descriptions that R8 reproduced readout preparation from scratch with dense checkpoints and that R10 established orientation-dependent selective survival;
- the R11 reproduction of the R10 orientation→survival effect across seeds 7, 19, and 43.

## What is not complete on `main`

The repository itself already states that the complete generated R8–R11 working bundles, raw arrays, checkpoints, and figures are not all checked in.

Therefore ND-R1 must not claim to be rerunning an exact recovered R8 implementation unless that source is subsequently recovered and hash-identified.

## ND-R1 reconstruction rule

Before primary execution, create `IMPLEMENTATION_PROVENANCE.md` containing:

1. every reused source file and commit/artifact identifier;
2. exact architecture;
3. task generator and split construction;
4. optimizer, learning rate, batch size, training objective, and epoch schedule;
5. checkpoint policy;
6. all reconstruction choices that were not directly recoverable.

Those choices must be committed before any fresh-seed primary result is inspected.

If source recovery shows a material mismatch with the currently frozen 100-epoch schedule or Observer-core lineage assumptions, record a **pre-execution amendment**. The amendment may correct provenance/implementation facts but may not alter the primary scientific question or use result knowledge to favor the historical pattern.

## Historical data use

Historical R8–R10 values may be used to verify provenance and motivate diagnostics. They are not part of the fresh ND-R1 confirmatory outcome classification.
