# ALI-N8-R1 Protocol Deviations

This file records execution deviations without altering the locked scientific design in `PREREGISTRATION.md`.

## Seed 5: early core-only test evaluation

The first completed GitHub Actions core run for primary seed 5 (`ALI-N8-R1 Core`, run 33538508455) selected the core checkpoint using validation data only and calibrated alpha from training latent states only. However, after the checkpoint had been selected and frozen, the core runner evaluated the discarded temporary reconstruction heads on the seed-5 test split before downstream ALI systems, controls, and diagnostic decoders were trained.

This violates the stricter preregistered sequencing rule that the test split should remain untouched until all downstream systems and diagnostics have been selected.

The early test result was not used to choose or modify the core checkpoint, alpha, dataset generator, ALI architecture, controls, optimization settings, diagnostic architecture, or any other scientific design element. No downstream ALI test predictions were produced by that run. The completed core artifact is retained unchanged and the deviation is not repaired by changing the test split or tuning the experiment.

All downstream seed-5 training and checkpoint selection is therefore performed using train and validation data only. The staged workflow trains/selects all main systems and all 64 diagnostic readers before its final downstream test phase. Seed 5 must nevertheless be reported with this procedural-deviation caveat in any R1 write-up.
