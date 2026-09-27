"""Coupling-routed distillation objectives."""

from .coupling_routed import (
    RoutedLossOutput,
    coupling_routed_k1_teacher_mode_loss,
)

__all__ = [
    "RoutedLossOutput",
    "coupling_routed_k1_teacher_mode_loss",
]

