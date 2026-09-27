"""Positive residual used by exact maximal coupling, not as a supervision target."""

import torch


def positive_coupling_residual(p: torch.Tensor, q: torch.Tensor):
    positive = (q - p).clamp_min(0.0)
    residual_mass = positive.sum()
    if float(residual_mass) > torch.finfo(torch.float32).eps:
        return positive / residual_mass, residual_mass
    return torch.zeros_like(positive), residual_mass
