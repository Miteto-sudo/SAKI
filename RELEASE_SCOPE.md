# Release scope

## Included

- Full-vocabulary geometric trust-region bridge.
- Scalar and batched bridge solvers, including the exact-zero endpoint.
- Exact maximal coupling and positive-residual correction sampling.
- Sequential and batched first-rejection commit/invalidate semantics.
- Device-resident block verification returning compact token-aligned metadata and teacher-mode targets.
- Correction-triggered routing between accepted-position K1/RKL and correction-position teacher-mode supervision.
- Constant, linear, cosine, and piecewise epsilon schedules.
- Regression tests for bridge endpoints, output marginals, total-variation correction rates, first rejection, masking, objective gradients, valid-token normalization, teacher-mode targets, and annealing.
- A path-free reference configuration and CPU toy example.

## Not included

- Model weights or optimizer states.
- Datasets, generated responses, benchmark outputs, or evaluation/scoring infrastructure.
- Private filesystem paths, cluster launchers, GPU guards, or service addresses.
- The full VERL fork, complete SGLang integration, or Ray orchestration.
- Production cache synchronization, distributed weight synchronization, and other cluster-specific systems code.
- Unrelated ablations and experimental branches.

The omitted components are distributed and systems integrations rather than changes to the mathematical rollout distribution or coupling-routed objective represented here. In particular, this reference package is not intended by itself to reproduce the paper's end-to-end systems throughput measurement.

## Publication checklist

- Add a license only if one is selected by the authors.
- Add the anonymous paper citation or public bibliographic entry when appropriate.
- Run `scripts/run_tests.sh` in the release environment.
- Run the CPU toy example.
- Re-run the anonymity scan.
- Verify `MANIFEST.sha256` after the final edit.
