"""GPU-resident batched verification for exact SAKI q blocks."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from saki_opd.math.bridge import trust_region_bridges


_DEBUG_Q = os.environ.get("SAKI_Q_DEBUG_MATH", "0").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class QBlockVerification:
    """Compact token-aligned results for a wave of speculative blocks."""

    output_tokens: torch.Tensor
    commit_lengths: torch.Tensor
    valid_mask: torch.Tensor
    correction_mask: torch.Tensor
    teacher_mode_ids: torch.Tensor
    p_final_logprob: torch.Tensor
    q_final_logprob: torch.Tensor
    beta: torch.Tensor
    kl_q_p: torch.Tensor
    residual_mass: torch.Tensor
    acceptance_probability: torch.Tensor


def verify_q_blocks(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    proposal_tokens: torch.Tensor,
    *,
    epsilon: float | torch.Tensor,
    bisection_steps: int = 16,
    beta_tolerance: float = 1e-5,
    kl_tolerance: float = 1e-6,
    generator: torch.Generator | None = None,
) -> QBlockVerification:
    """Verify ``[batch, block, vocab]`` proposals entirely on their device."""
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student and teacher logits must be aligned [batch, block, vocab]")
    batch_size, block_size, vocab_size = student_logits.shape
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("q block batch and block dimensions must be non-empty")
    if proposal_tokens.shape != (batch_size, block_size):
        raise ValueError("proposal_tokens must have shape [batch, block]")
    proposals = proposal_tokens.to(device=student_logits.device, dtype=torch.long)
    if _DEBUG_Q and bool(((proposals < 0) | (proposals >= vocab_size)).any()):
        raise ValueError("proposal token is outside the bridge vocabulary")

    epsilon_rows = torch.as_tensor(
        epsilon, dtype=torch.float32, device=student_logits.device
    ).reshape(-1)
    if epsilon_rows.numel() == 1:
        flat_epsilon = epsilon_rows
    elif epsilon_rows.numel() == batch_size:
        flat_epsilon = epsilon_rows.repeat_interleave(block_size)
    elif epsilon_rows.numel() == batch_size * block_size:
        flat_epsilon = epsilon_rows
    else:
        raise ValueError(
            "epsilon must be scalar, [batch], or [batch, block]"
        )
    flat_bridge = trust_region_bridges(
        student_logits.reshape(-1, vocab_size),
        teacher_logits.reshape(-1, vocab_size),
        epsilon=flat_epsilon,
        bisection_steps=bisection_steps,
        beta_tolerance=beta_tolerance,
        kl_tolerance=kl_tolerance,
    )
    p = flat_bridge.p.view(batch_size, block_size, vocab_size)
    q = flat_bridge.q.view(batch_size, block_size, vocab_size)
    beta = flat_bridge.beta.view(batch_size, block_size)
    kl_q_p = flat_bridge.kl_q_p.view(batch_size, block_size)

    proposal_index = proposals.unsqueeze(-1)
    tiny = torch.finfo(torch.float32).tiny
    proposal_p = p.gather(-1, proposal_index).squeeze(-1).clamp_min(tiny)
    proposal_q = q.gather(-1, proposal_index).squeeze(-1)
    acceptance_probability = (proposal_q / proposal_p).clamp(min=0.0, max=1.0)
    uniforms = torch.rand(
        (batch_size, block_size),
        device=student_logits.device,
        dtype=torch.float32,
        generator=generator,
    )
    accepted = uniforms < acceptance_probability
    accepted_prefix = accepted.to(torch.int32).cumprod(dim=1).to(torch.bool)
    rejected = ~accepted_prefix
    has_rejection = rejected.any(dim=1)
    first_rejection = rejected.to(torch.int64).argmax(dim=1)
    first_rejection = torch.where(
        has_rejection,
        first_rejection,
        torch.full_like(first_rejection, block_size),
    )
    commit_lengths = torch.where(has_rejection, first_rejection + 1, first_rejection)
    positions = torch.arange(block_size, device=student_logits.device).unsqueeze(0)
    valid_mask = positions < commit_lengths.unsqueeze(1)
    correction_mask = (positions == first_rejection.unsqueeze(1)) & has_rejection.unsqueeze(1)
    teacher_mode_ids = teacher_logits.argmax(dim=-1).to(dtype=torch.long)

    output_tokens = proposals.clone()
    # Keep this token-aligned for every committed row.  It is useful both for
    # training diagnostics and for proving that the GPU path matches the
    # existing exact bridge implementation; it is not only a rejection-time
    # quantity.
    residual_mass = (q - p).clamp_min(0.0).sum(dim=-1)
    # Indexing with the GPU-computed row list avoids a Python-side any() sync.
    # torch.nonzero and the empty-row multinomial stay on device, so a wave
    # with no rejections no longer blocks the scheduler on the host.
    rejected_rows = torch.nonzero(has_rejection, as_tuple=False).squeeze(-1)
    rejected_pos = first_rejection[rejected_rows]
    p_rejected = p[rejected_rows, rejected_pos]
    q_rejected = q[rejected_rows, rejected_pos]
    positive_residual = (q_rejected - p_rejected).clamp_min(0.0)
    mass = positive_residual.sum(dim=-1)
    use_q = mass <= torch.finfo(torch.float32).eps
    correction_distribution = positive_residual / mass.clamp_min(tiny).unsqueeze(-1)
    correction_distribution = torch.where(
        use_q.unsqueeze(-1), q_rejected, correction_distribution
    )
    correction_tokens = torch.multinomial(
        correction_distribution, 1, generator=generator
    ).squeeze(-1)
    output_tokens[rejected_rows, rejected_pos] = correction_tokens

    final_index = output_tokens.unsqueeze(-1)
    p_final_logprob = p.gather(-1, final_index).squeeze(-1).clamp_min(tiny).log()
    q_final_logprob = q.gather(-1, final_index).squeeze(-1).clamp_min(tiny).log()
    zero = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    one = torch.ones((), device=student_logits.device, dtype=torch.float32)
    p_final_logprob = torch.where(valid_mask, p_final_logprob, zero)
    q_final_logprob = torch.where(valid_mask, q_final_logprob, zero)
    beta = torch.where(valid_mask, beta, zero)
    kl_q_p = torch.where(valid_mask, kl_q_p, zero)
    residual_mass = torch.where(valid_mask, residual_mass, zero)
    acceptance_probability = torch.where(valid_mask, acceptance_probability, one)
    teacher_mode_ids = torch.where(
        valid_mask,
        teacher_mode_ids,
        torch.full_like(teacher_mode_ids, -1),
    )

    return QBlockVerification(
        output_tokens=output_tokens,
        commit_lengths=commit_lengths,
        valid_mask=valid_mask,
        correction_mask=correction_mask,
        teacher_mode_ids=teacher_mode_ids,
        p_final_logprob=p_final_logprob,
        q_final_logprob=q_final_logprob,
        beta=beta,
        kl_q_p=kl_q_p,
        residual_mass=residual_mass,
        acceptance_probability=acceptance_probability,
    )
