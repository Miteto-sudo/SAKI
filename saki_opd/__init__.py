"""Framework-independent core reference implementation of SAKI."""

from .core import (
    Bridge,
    CouplingSamples,
    QBlockVerification,
    RoutedLossOutput,
    coupling_routed_k1_teacher_mode_loss,
    maximal_coupling_samples,
    resolve_epsilon_schedule,
    trust_region_bridge,
    trust_region_bridges,
    verify_q_blocks,
)

__all__ = [
    "Bridge",
    "CouplingSamples",
    "QBlockVerification",
    "RoutedLossOutput",
    "coupling_routed_k1_teacher_mode_loss",
    "maximal_coupling_samples",
    "resolve_epsilon_schedule",
    "trust_region_bridge",
    "trust_region_bridges",
    "verify_q_blocks",
]
