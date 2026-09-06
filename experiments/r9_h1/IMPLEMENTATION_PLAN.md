# Frozen implementation plan

This file records the implementation order only; it does not alter the preregistered scientific criteria.

1. Reuse the R9-T1 generator and event encoding unchanged.
2. Implement the nine R9-H1 arms behind a shared training/evaluation interface.
3. Implement shared probe/control utilities.
4. Implement per-seed JSON artifact generation.
5. Implement frozen cross-seed classifier.
6. Add outcome-free smoke workflow followed by eight family jobs and aggregate classification.
