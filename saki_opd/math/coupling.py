"""Exact K=1 maximal coupling and direct-q sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .bridge import Bridge, trust_region_bridge
from .residual import positive_coupling_residual


@dataclass(frozen=True)
class CouplingSamples:
    proposal_tokens: torch.Tensor
    final_tokens: torch.Tensor
    accepted: torch.Tensor
    beta: torch.Tensor
    kl_q_p: torch.Tensor
    acceptance_probability: torch.Tensor
    residual_mass: torch.Tensor
    correction_distribution: torch.Tensor


@dataclass(frozen=True)
class BehaviorSamples:
    proposal_tokens: torch.Tensor
    final_tokens: torch.Tensor
    accepted: torch.Tensor
    bridge: Bridge
    acceptance_probability: torch.Tensor
    residual_mass: torch.Tensor


@dataclass(frozen=True)
class SpeculativeBlockVerification:
    emitted_tokens: torch.Tensor
    accepted: torch.Tensor
    acceptance_probability: torch.Tensor


def maximal_coupling_samples(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_samples: int,
    epsilon: float = 0.02,
    generator: torch.Generator | None = None,
) -> CouplingSamples:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    bridge = trust_region_bridge(student_logits, teacher_logits, epsilon)
    p, q = bridge.p, bridge.q
    proposals = torch.multinomial(p, num_samples, replacement=True, generator=generator)
    proposal_p = p.gather(0, proposals).clamp_min(torch.finfo(torch.float32).tiny)
    accept_prob = (q.gather(0, proposals) / proposal_p).clamp(max=1.0)
    uniforms = torch.rand(num_samples, device=p.device, generator=generator)
    accepted = uniforms < accept_prob
    residual, residual_mass = positive_coupling_residual(p, q)
    if float(residual_mass) > torch.finfo(torch.float32).eps:
        corrections = torch.multinomial(residual, num_samples, replacement=True, generator=generator)
    else:
        corrections = proposals
    final = torch.where(accepted, proposals, corrections)
    return CouplingSamples(
        proposal_tokens=proposals,
        final_tokens=final,
        accepted=accepted,
        beta=bridge.beta,
        kl_q_p=bridge.kl_q_p,
        acceptance_probability=accept_prob,
        residual_mass=residual_mass,
        correction_distribution=residual,
    )


def direct_q_samples(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_samples: int,
    epsilon: float = 0.02,
    generator: torch.Generator | None = None,
) -> BehaviorSamples:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    bridge = trust_region_bridge(student_logits, teacher_logits, epsilon)
    proposals = torch.multinomial(bridge.p, num_samples, replacement=True, generator=generator)
    final = torch.multinomial(bridge.q, num_samples, replacement=True, generator=generator)
    residual_mass = (bridge.q - bridge.p).clamp_min(0.0).sum()
    return BehaviorSamples(
        proposal_tokens=proposals,
        final_tokens=final,
        accepted=torch.ones(num_samples, dtype=torch.bool, device=bridge.p.device),
        bridge=bridge,
        acceptance_probability=torch.ones(num_samples, dtype=torch.float32, device=bridge.p.device),
        residual_mass=residual_mass,
    )


def maximal_coupling_from_bridge(
    bridge: Bridge,
    proposal_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> BehaviorSamples:
    """Complete maximal coupling from a bridge that has already been solved.

    The online agent loop needs the bridge for logging p(final), q(final), beta,
    and KL.  Accepting a precomputed bridge avoids solving the same full-vocab
    trust-region problem a second time inside coupling.
    """
    if proposal_tokens.ndim != 1 or proposal_tokens.numel() == 0:
        raise ValueError("proposal_tokens must be a non-empty rank-1 tensor")
    p, q = bridge.p, bridge.q
    proposals = proposal_tokens.long()
    proposal_p = p.gather(0, proposals).clamp_min(torch.finfo(torch.float32).tiny)
    accept_prob = (q.gather(0, proposals) / proposal_p).clamp(max=1.0)
    uniforms = torch.rand(proposals.numel(), device=p.device, generator=generator)
    accepted = uniforms < accept_prob

    # Residual mass is retained for exact diagnostics.  Normalization and the
    # comparatively costly multinomial are only needed for rejected proposals.
    positive_residual = (q - p).clamp_min(0.0)
    residual_mass = positive_residual.sum()
    final = proposals.clone()
    rejected = ~accepted
    if bool(rejected.any()) and float(residual_mass) > torch.finfo(torch.float32).eps:
        residual = positive_residual / residual_mass
        final[rejected] = torch.multinomial(
            residual,
            int(rejected.sum()),
            replacement=True,
            generator=generator,
        )

    return BehaviorSamples(
        proposal_tokens=proposals,
        final_tokens=final,
        accepted=accepted,
        bridge=bridge,
        acceptance_probability=accept_prob,
        residual_mass=residual_mass,
    )


def verify_speculative_block(
    bridges: Sequence[Bridge],
    proposal_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> SpeculativeBlockVerification:
    """Verify an autoregressive proposal block with exact maximal coupling.

    Verification stops after the first rejection because later bridge rows were
    computed under the rejected proposal prefix. Accepted proposals and the
    single correction token are exactly distributed according to q.
    """
    if proposal_tokens.ndim != 1 or proposal_tokens.numel() == 0:
        raise ValueError("proposal_tokens must be a non-empty rank-1 tensor")
    if len(bridges) != proposal_tokens.numel():
        raise ValueError("one bridge is required for every proposal token")

    device = bridges[0].p.device
    emitted_tokens: list[int] = []
    accepted_values: list[bool] = []
    acceptance_probabilities: list[float] = []
    tiny = torch.finfo(torch.float32).tiny

    for bridge, proposal_value in zip(bridges, proposal_tokens.tolist()):
        p, q = bridge.p, bridge.q
        if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
            raise ValueError("every bridge must contain aligned rank-1 distributions")
        if p.device != device or q.device != device:
            raise ValueError("all bridge distributions must share one device")
        proposal = int(proposal_value)
        if proposal < 0 or proposal >= p.numel():
            raise ValueError("proposal token is outside the bridge vocabulary")

        proposal_p = p[proposal].clamp_min(tiny)
        accept_probability = (q[proposal] / proposal_p).clamp(max=1.0)
        accepted = bool(torch.rand((), device=device, generator=generator) < accept_probability)
        acceptance_probabilities.append(float(accept_probability))

        if accepted:
            emitted_tokens.append(proposal)
            accepted_values.append(True)
            continue

        positive_residual = (q - p).clamp_min(0.0)
        residual_mass = positive_residual.sum()
        if float(residual_mass) <= torch.finfo(torch.float32).eps:
            correction = int(torch.multinomial(q, 1, generator=generator).item())
        else:
            correction = int(
                torch.multinomial(
                    positive_residual / residual_mass,
                    1,
                    generator=generator,
                ).item()
            )
        emitted_tokens.append(correction)
        accepted_values.append(False)
        break

    return SpeculativeBlockVerification(
        emitted_tokens=torch.tensor(emitted_tokens, dtype=torch.long, device=device),
        accepted=torch.tensor(accepted_values, dtype=torch.bool, device=device),
        acceptance_probability=torch.tensor(
            acceptance_probabilities,
            dtype=torch.float32,
            device=device,
        ),
    )


def verify_batched_speculative_block(
    bridge: Bridge,
    proposal_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> SpeculativeBlockVerification:
    """Vectorized exact verification for a batched bridge ``[K, vocab]``.

    Acceptance probabilities and random decisions are computed in one tensor
    operation. Only the first rejected row needs a residual distribution and
    multinomial draw; later rows are discarded because their prefix is stale.
    """
    p, q = bridge.p, bridge.q
    if p.ndim != 2 or q.ndim != 2 or p.shape != q.shape:
        raise ValueError("batched verification expects aligned [rows, vocab] distributions")
    if proposal_tokens.ndim != 1 or proposal_tokens.numel() != p.shape[0]:
        raise ValueError("one proposal token is required for every bridge row")
    proposals = proposal_tokens.to(device=p.device, dtype=torch.long)
    if bool(((proposals < 0) | (proposals >= p.shape[1])).any()):
        raise ValueError("proposal token is outside the bridge vocabulary")

    row_ids = torch.arange(p.shape[0], device=p.device)
    tiny = torch.finfo(torch.float32).tiny
    proposal_p = p[row_ids, proposals].clamp_min(tiny)
    acceptance_probability = (q[row_ids, proposals] / proposal_p).clamp(max=1.0)
    # Preserve the reference verifier's RNG order: draw one uniform at a time
    # and stop immediately at the first rejection, before drawing a correction.
    accepted_values = []
    first_rejection = None
    for row in range(p.shape[0]):
        accepted = bool(
            torch.rand((), device=p.device, generator=generator) < acceptance_probability[row]
        )
        accepted_values.append(accepted)
        if not accepted:
            first_rejection = row
            break
    emitted_count = len(accepted_values)
    accepted_all = torch.tensor(accepted_values, dtype=torch.bool, device=p.device)
    emitted = proposals.clone()
    if first_rejection is not None:
        positive_residual = (q[first_rejection] - p[first_rejection]).clamp_min(0.0)
        residual_mass = positive_residual.sum()
        if float(residual_mass) <= torch.finfo(torch.float32).eps:
            correction_distribution = q[first_rejection]
        else:
            correction_distribution = positive_residual / residual_mass
        emitted[first_rejection] = torch.multinomial(
            correction_distribution,
            1,
            generator=generator,
        )

    return SpeculativeBlockVerification(
        emitted_tokens=emitted[:emitted_count],
        accepted=accepted_all,
        acceptance_probability=acceptance_probability[:emitted_count],
    )


def maximal_coupling_from_proposals(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    proposal_tokens: torch.Tensor,
    epsilon: float = 0.02,
    generator: torch.Generator | None = None,
) -> BehaviorSamples:
    if proposal_tokens.ndim != 1 or proposal_tokens.numel() == 0:
        raise ValueError("proposal_tokens must be a non-empty rank-1 tensor")
    bridge = trust_region_bridge(student_logits, teacher_logits, epsilon)
    return maximal_coupling_from_bridge(bridge, proposal_tokens, generator=generator)


def maximal_coupling_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    epsilon: float = 0.02,
    generator: torch.Generator | None = None,
) -> CouplingSamples:
    return maximal_coupling_samples(student_logits, teacher_logits, 1, epsilon, generator)
