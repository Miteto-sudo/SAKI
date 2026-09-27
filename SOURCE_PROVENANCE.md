# Source provenance

This anonymous review package was extracted from the authors' development implementation. The bridge, maximal-coupling, block-verification, schedule, and coupling-routed objective code were isolated into a framework-independent reference package for review.

The direct bridge solver explicitly handles `epsilon == 0.0` and returns exactly `beta == 0.0`, `q == p`, and `KL(q||p) == 0.0`, matching the student-only endpoint used by the runtime semantics.

The SHA-256 manifest in `MANIFEST.sha256` identifies the exact prepared files.
