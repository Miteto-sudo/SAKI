"""Correction-triggered K1/RKL and teacher-mode supervision.

Accepted positions use the detached sampled-token K1 signal with the one-epoch
clipped importance-ratio optimization used in training. Correction positions
remove the accepted-token term and replace it with NLL on the teacher's highest-
probability token. Both branches share the valid-token denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoutedLossOutput:
    total: torch.Tensor
    accepted_rkl: torch.Tensor
    teacher_mode: torch.Tensor
    valid_count: torch.Tensor
    correction_count: torch.Tensor


def _require_shape(name: str, value: torch.Tensor, shape: torch.Size) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")


def coupling_routed_k1_teacher_mode_loss(
    actor_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    old_student_logprobs: torch.Tensor,
    teacher_sample_logprobs: torch.Tensor,
    teacher_mode_ids: torch.Tensor,
    correction_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
    dual_clip_ratio: float = 3.0,
    k1_value_clamp: float = 10.0,
) -> RoutedLossOutput:
    """Compute the main coupling-routed objective.

    Args:
        actor_logits: Current student logits with shape ``[B, L, V]``.
        sampled_token_ids: Final rollout token IDs, shape ``[B, L]``.
        old_student_logprobs: Frozen pre-update student log-probability of the
            sampled token, shape ``[B, L]``.
        teacher_sample_logprobs: Teacher log-probability of that sampled token,
            shape ``[B, L]``.
        teacher_mode_ids: Teacher argmax token at each realized prefix, shape
            ``[B, L]``. Values at non-correction positions are ignored.
        correction_mask: Realized maximal-coupling correction events.
        valid_mask: Valid response-token mask. The sum of this mask is the only
            normalization denominator.

    The correction token controls the next rollout prefix. It is intentionally
    not used as the direct correction target; ``teacher_mode_ids`` supplies that
    target.
    """
    if actor_logits.ndim != 3:
        raise ValueError("actor_logits must have shape [batch, length, vocab]")
    token_shape = actor_logits.shape[:2]
    for name, tensor in (
        ("sampled_token_ids", sampled_token_ids),
        ("old_student_logprobs", old_student_logprobs),
        ("teacher_sample_logprobs", teacher_sample_logprobs),
        ("teacher_mode_ids", teacher_mode_ids),
        ("correction_mask", correction_mask),
        ("valid_mask", valid_mask),
    ):
        _require_shape(name, tensor, token_shape)
    if clip_ratio < 0.0:
        raise ValueError("clip_ratio must be non-negative")
    if dual_clip_ratio <= 1.0:
        raise ValueError("dual_clip_ratio must exceed 1")
    if k1_value_clamp <= 0.0:
        raise ValueError("k1_value_clamp must be positive")

    valid = valid_mask.bool()
    correction = correction_mask.bool() & valid
    accepted = valid & ~correction
    valid_count = valid.sum()
    correction_count = correction.sum()
    if int(valid_count) == 0:
        zero = actor_logits.sum() * 0.0
        return RoutedLossOutput(zero, zero, zero, valid_count, correction_count)

    log_probs = F.log_softmax(actor_logits.float(), dim=-1)
    current_sample_logprobs = log_probs.gather(
        -1, sampled_token_ids.long().unsqueeze(-1)
    ).squeeze(-1)

    # Same sampled-token K1 value and detached policy-gradient semantics used
    # by the training implementation.
    k1 = (current_sample_logprobs - teacher_sample_logprobs).clamp(
        min=-k1_value_clamp, max=k1_value_clamp
    )
    advantages = -k1.detach()
    log_ratio = (current_sample_logprobs - old_student_logprobs).clamp(
        min=-20.0, max=20.0
    )
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    clipped_surrogate = torch.maximum(unclipped, clipped)
    dual_clip = -advantages * dual_clip_ratio
    k1_surrogate = torch.where(
        advantages < 0.0,
        torch.minimum(dual_clip, clipped_surrogate),
        clipped_surrogate,
    )

    # No K1 term is retained at a correction position.
    accepted_sum = k1_surrogate.masked_select(accepted).sum()

    # q-block verification marks the invalid speculative suffix with -1. Those
    # rows never receive teacher-mode supervision, but gather still requires a
    # valid vocabulary index before masking. Substituting token 0 outside the
    # correction mask leaves all valid loss values and gradients unchanged.
    safe_teacher_mode_ids = torch.where(
        correction,
        teacher_mode_ids,
        torch.zeros_like(teacher_mode_ids),
    )
    teacher_mode_nll = -log_probs.gather(
        -1, safe_teacher_mode_ids.long().unsqueeze(-1)
    ).squeeze(-1)
    teacher_mode_sum = teacher_mode_nll.masked_select(correction).sum()

    denominator = valid_count.to(dtype=actor_logits.dtype)
    accepted_loss = accepted_sum / denominator
    teacher_mode_loss = teacher_mode_sum / denominator
    return RoutedLossOutput(
        total=accepted_loss + teacher_mode_loss,
        accepted_rkl=accepted_loss,
        teacher_mode=teacher_mode_loss,
        valid_count=valid_count,
        correction_count=correction_count,
    )
