"""Rollout configuration, epsilon scheduling, and lifecycle guards."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


def _epsilon(value: Any, *, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number, got {value!r}")
    return value


def _piecewise_points(raw_points: Sequence[Any]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for item in raw_points:
        if isinstance(item, Mapping) or hasattr(item, "get"):
            step = int(_get(item, "step"))
            value = _epsilon(_get(item, "epsilon"), field="saki.epsilon_schedule.points[].epsilon")
        else:
            if len(item) != 2:
                raise ValueError("Each piecewise epsilon point must be [step, epsilon] or a mapping")
            step, value = int(item[0]), _epsilon(item[1], field="saki.epsilon_schedule.points[][1]")
        if step < 0:
            raise ValueError("Piecewise epsilon steps must be non-negative")
        points.append((step, value))
    points.sort(key=lambda point: point[0])
    if not points:
        raise ValueError("saki.epsilon_schedule.points must not be empty for a piecewise schedule")
    if len({step for step, _ in points}) != len(points):
        raise ValueError("saki.epsilon_schedule.points must have unique steps")
    return points


def resolve_epsilon_schedule(saki_config: Any, global_step: int) -> float:
    """Resolve the rollout epsilon for one batch boundary.

    The exact endpoint value is returned outside the interpolation interval, so
    an ``end_epsilon`` of zero is represented as exactly ``0.0`` and can safely
    trigger the student-only rollout path.
    """

    step = int(global_step)
    if step < 0:
        raise ValueError(f"global_step must be non-negative, got {step}")
    base = _epsilon(_get(saki_config, "epsilon", 0.02), field="saki.epsilon")
    schedule = _get(saki_config, "epsilon_schedule")
    if schedule is None:
        return base

    schedule_type = str(_get(schedule, "type", "constant")).lower()
    if schedule_type == "constant":
        value = _get(schedule, "value")
        return base if value is None else _epsilon(value, field="saki.epsilon_schedule.value")

    if schedule_type == "piecewise":
        points = _piecewise_points(_get(schedule, "points", []))
        interpolation = str(_get(schedule, "interpolation", "constant")).lower()
        if step <= points[0][0]:
            return points[0][1]
        if step >= points[-1][0]:
            return points[-1][1]
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:]):
            if left_step <= step < right_step:
                if interpolation == "constant":
                    return left_value
                if interpolation == "linear":
                    progress = (step - left_step) / (right_step - left_step)
                    return left_value + (right_value - left_value) * progress
                raise ValueError("piecewise interpolation must be 'constant' or 'linear'")
        raise AssertionError("unreachable piecewise epsilon interval")

    if schedule_type not in {"linear", "cosine"}:
        raise ValueError("saki.epsilon_schedule.type must be constant, linear, cosine, or piecewise")
    start_step = int(_get(schedule, "start_step", 0))
    end_step_raw = _get(schedule, "end_step")
    if end_step_raw is None:
        raise ValueError(f"saki.epsilon_schedule.end_step is required for {schedule_type}")
    end_step = int(end_step_raw)
    if start_step < 0 or end_step <= start_step:
        raise ValueError("epsilon schedule requires 0 <= start_step < end_step")
    start_raw = _get(schedule, "start_epsilon")
    end_raw = _get(schedule, "end_epsilon")
    start_value = base if start_raw is None else _epsilon(start_raw, field="saki.epsilon_schedule.start_epsilon")
    end_value = base if end_raw is None else _epsilon(end_raw, field="saki.epsilon_schedule.end_epsilon")
    if step <= start_step:
        return start_value
    if step >= end_step:
        return end_value
    progress = (step - start_step) / (end_step - start_step)
    if schedule_type == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_value + (end_value - start_value) * progress


def validate_rollout_mode(mode: str) -> None:
    allowed = {"vanilla", "direct_q", "coupling_q"}
    if mode not in allowed:
        raise ValueError(f"saki.rollout_mode must be one of {sorted(allowed)}, got {mode!r}")


def uses_q_rollout(mode: str) -> bool:
    validate_rollout_mode(mode)
    return mode in {"direct_q", "coupling_q"}


def validate_frozen_pi_old_steps(global_steps: list[int]) -> None:
    if global_steps and min(global_steps) != max(global_steps):
        raise RuntimeError(
            "pi_old changed during one response; rollout weights must stay frozen until the update boundary."
        )


def validate_block_size(block_size: int) -> None:
    if block_size not in {1, 2, 4, 8, 16}:
        raise ValueError("saki.block_size must be one of 1, 2, 4, 8, or 16")
