"""Trust-region geometric bridge over full-vocabulary logits."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Bridge:
    p: torch.Tensor
    q: torch.Tensor
    beta: torch.Tensor
    kl_q_p: torch.Tensor


_DEBUG_MATH = os.environ.get("SAKI_Q_DEBUG_MATH", "0").lower() in {"1", "true", "yes"}


def _bridge_at_beta(log_p: torch.Tensor, log_t: torch.Tensor, beta: torch.Tensor):
    beta_value = float(beta)
    if beta_value == 0.0:
        return log_p.exp(), torch.zeros((), dtype=torch.float32, device=log_p.device)
    if beta_value == 1.0:
        log_q = log_t
    else:
        log_q = (1.0 - beta) * log_p + beta * log_t
        log_q = log_q - torch.logsumexp(log_q, dim=-1)
    q = log_q.exp()
    kl_terms = torch.where(q > 0, q * (log_q - log_p), torch.zeros_like(q))
    return q, torch.sum(kl_terms, dim=-1)


def _bridges_at_beta(log_p: torch.Tensor, log_t: torch.Tensor, beta: torch.Tensor):
    """Vectorized bridge evaluation for ``[rows, vocab]`` log-probabilities."""
    zero_endpoint = beta == 0.0
    teacher_endpoint = beta == 1.0
    # Do not evaluate 0 * -inf at either exact endpoint. Apart from producing
    # NaNs, that would disagree with the scalar reference implementation where
    # beta=0 is exactly p and beta=1 is exactly the teacher distribution.
    interior_beta = torch.where(
        zero_endpoint | teacher_endpoint,
        torch.full_like(beta, 0.5),
        beta,
    )
    beta_column = interior_beta.unsqueeze(-1)
    log_q = (1.0 - beta_column) * log_p + beta_column * log_t
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
    log_q = torch.where(zero_endpoint.unsqueeze(-1), log_p, log_q)
    log_q = torch.where(teacher_endpoint.unsqueeze(-1), log_t, log_q)
    q = log_q.exp()
    kl_terms = torch.where(q > 0, q * (log_q - log_p), torch.zeros_like(q))
    return q, torch.sum(kl_terms, dim=-1)


def trust_region_bridges(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    epsilon: float | torch.Tensor = 0.02,
    bisection_steps: int = 50,
    beta_tolerance: float = 0.0,
    kl_tolerance: float = 0.0,
) -> Bridge:
    """Solve a wave of independent exact geometric bridges together.

    The returned ``Bridge`` stores ``p`` and ``q`` as ``[rows, vocab]`` and
    beta/KL as ``[rows]``. Rows stop independently under the same tolerances as
    :func:`trust_region_bridge`.
    """
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("batched bridge expects [rows, vocab] tensors")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[0] == 0:
        raise ValueError("student and teacher batch shapes must match and be non-empty")
    epsilon_rows = torch.as_tensor(
        epsilon, dtype=torch.float32, device=student_logits.device
    ).reshape(-1)
    if epsilon_rows.numel() == 1:
        epsilon_rows = epsilon_rows.expand(student_logits.shape[0])
    elif epsilon_rows.numel() != student_logits.shape[0]:
        raise ValueError("epsilon must be scalar or have one value per bridge row")
    if bool((epsilon_rows < 0).any()):
        raise ValueError("epsilon must be non-negative")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    if beta_tolerance < 0 or kl_tolerance < 0:
        raise ValueError("bridge tolerances must be non-negative")
    # These checks force a GPU->CPU synchronization. Keep them for debugging,
    # but leave them off in the rollout hot path where logits are expected to be finite.
    if _DEBUG_MATH:
        if torch.isnan(student_logits).any() or torch.isposinf(student_logits).any():
            raise ValueError("pi_old logits contain NaN or +inf")
        if torch.isnan(teacher_logits).any() or torch.isposinf(teacher_logits).any():
            raise ValueError("Teacher logits contain NaN or +inf")

    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_t = F.log_softmax(teacher_logits.float(), dim=-1)
    if _DEBUG_MATH:
        if not bool(torch.isfinite(torch.logsumexp(log_p, dim=-1)).all()):
            raise ValueError("pi_old distribution has no finite support")
        if not bool(torch.isfinite(torch.logsumexp(log_t, dim=-1)).all()):
            raise ValueError("Teacher distribution has no finite support")

    p = log_p.exp()
    rows = log_p.shape[0]
    one = torch.ones(rows, dtype=torch.float32, device=log_p.device)
    q_one, kl_one = _bridges_at_beta(log_p, log_t, one)
    zero_endpoint = epsilon_rows == 0.0
    needs_search = (kl_one > epsilon_rows) & ~zero_endpoint
    # Always execute a fixed number of bisection waves. The former early return
    # and active.any() checks synchronized the GPU batch with Python per block.
    lo = torch.zeros_like(one)
    hi = one.clone()
    active = needs_search.clone()
    for _ in range(bisection_steps):
        mid = (lo + hi) * 0.5
        _, kl_mid = _bridges_at_beta(log_p, log_t, mid)
        feasible = kl_mid <= epsilon_rows
        lo = torch.where(active & feasible, mid, lo)
        hi = torch.where(active & ~feasible, mid, hi)
        converged = ((hi - lo) <= beta_tolerance) | (
            (kl_mid - epsilon_rows).abs() <= kl_tolerance
        )
        active = active & ~converged

    beta = torch.where(needs_search, lo, one)
    beta = torch.where(zero_endpoint, torch.zeros_like(beta), beta)
    q, kl = _bridges_at_beta(log_p, log_t, beta)
    return Bridge(p=p, q=q, beta=beta, kl_q_p=kl)


def trust_region_bridge(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    epsilon: float = 0.02,
    bisection_steps: int = 50,
    beta_tolerance: float = 0.0,
    kl_tolerance: float = 0.0,
) -> Bridge:
    """Return the largest beta whose geometric bridge obeys KL(q||p)<=epsilon."""
    if student_logits.ndim != 1 or teacher_logits.ndim != 1:
        raise ValueError("K=1 reference implementation expects one full-vocabulary vector")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher vocabulary shapes must match")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    if beta_tolerance < 0 or kl_tolerance < 0:
        raise ValueError("bridge tolerances must be non-negative")
    if torch.isnan(student_logits).any() or torch.isposinf(student_logits).any():
        raise ValueError("pi_old logits contain NaN or +inf")
    if torch.isnan(teacher_logits).any() or torch.isposinf(teacher_logits).any():
        raise ValueError("Teacher logits contain NaN or +inf")
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_t = F.log_softmax(teacher_logits.float(), dim=-1)
    if not torch.isfinite(torch.logsumexp(log_p, dim=-1)):
        raise ValueError("pi_old distribution has no finite support")
    if not torch.isfinite(torch.logsumexp(log_t, dim=-1)):
        raise ValueError("Teacher distribution has no finite support")
    p = log_p.exp()
    if epsilon == 0.0:
        zero = torch.zeros((), dtype=torch.float32, device=log_p.device)
        return Bridge(p=p, q=p, beta=zero, kl_q_p=zero)
    one = torch.ones((), dtype=torch.float32, device=log_p.device)
    q_one, kl_one = _bridge_at_beta(log_p, log_t, one)
    if float(kl_one) <= epsilon:
        return Bridge(p=p, q=q_one, beta=one, kl_q_p=kl_one)
    lo = torch.zeros((), dtype=torch.float32, device=log_p.device)
    hi = one
    for _ in range(bisection_steps):
        mid = (lo + hi) * 0.5
        _, kl_mid = _bridge_at_beta(log_p, log_t, mid)
        kl_value = float(kl_mid)
        if kl_value <= epsilon:
            lo = mid
        else:
            hi = mid
        if float(hi - lo) <= beta_tolerance or abs(kl_value - epsilon) <= kl_tolerance:
            break
    q, kl = _bridge_at_beta(log_p, log_t, lo)
    return Bridge(p=p, q=q, beta=lo, kl_q_p=kl)
