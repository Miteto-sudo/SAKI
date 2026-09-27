"""Pure, framework-independent SAKI mathematics."""

from .bridge import Bridge, trust_region_bridge, trust_region_bridges
from .coupling import (
    BehaviorSamples,
    CouplingSamples,
    SpeculativeBlockVerification,
    direct_q_samples,
    maximal_coupling_from_bridge,
    maximal_coupling_from_proposals,
    maximal_coupling_sample,
    maximal_coupling_samples,
    verify_batched_speculative_block,
    verify_speculative_block,
)

__all__ = [
    "BehaviorSamples",
    "Bridge",
    "CouplingSamples",
    "SpeculativeBlockVerification",
    "direct_q_samples",
    "maximal_coupling_from_bridge",
    "maximal_coupling_from_proposals",
    "maximal_coupling_sample",
    "maximal_coupling_samples",
    "trust_region_bridge",
    "trust_region_bridges",
    "verify_batched_speculative_block",
    "verify_speculative_block",
]
