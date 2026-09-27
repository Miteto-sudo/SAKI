"""Public facade for the core SAKI primitives."""

from saki_opd.engine import QBlockVerification, verify_q_blocks
from saki_opd.math import (
    Bridge,
    CouplingSamples,
    maximal_coupling_samples,
    trust_region_bridge,
    trust_region_bridges,
)
from saki_opd.objectives import (
    RoutedLossOutput,
    coupling_routed_k1_teacher_mode_loss,
)
from saki_opd.runtime import resolve_epsilon_schedule

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
