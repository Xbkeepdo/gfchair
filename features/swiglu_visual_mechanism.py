"""Local and finite internal decompositions for SwiGLU visual directions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SwiGLULocalComponents:
    normalized_directions: torch.Tensor
    gate_preactivation_directions: torch.Tensor
    up_preactivation_directions: torch.Tensor
    gate_path: torch.Tensor
    up_path: torch.Tensor
    activation_directions: torch.Tensor
    output_directions: torch.Tensor
    product_rule_relative_error: float


@dataclass(frozen=True)
class SwiGLUFiniteComponents:
    gate_path: torch.Tensor
    up_path: torch.Tensor
    interaction: torch.Tensor
    total_activation_change: torch.Tensor
    output_change: torch.Tensor
    completeness_relative_error: float


def silu_derivative(value: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(value)
    return sigmoid * (1.0 + value * (1.0 - sigmoid))


def local_swiglu_components(
    *,
    norm: Callable[[torch.Tensor], torch.Tensor],
    gate_projection: torch.nn.Module,
    up_projection: torch.nn.Module,
    down_projection: torch.nn.Module,
    z: torch.Tensor,
    writes: torch.Tensor,
    eps: float = 1e-12,
) -> SwiGLULocalComponents:
    if z.ndim != 1 or writes.ndim != 2 or writes.shape[1:] != z.shape:
        raise ValueError("Expected z[D] and writes[M,D]")
    normalized = norm(z)
    gate = gate_projection(normalized)
    up = up_projection(normalized)

    def norm_jvp(direction):
        return torch.func.jvp(norm, (z,), (direction,))[1]

    normalized_directions = torch.vmap(norm_jvp)(writes)
    gate_weight = gate_projection.weight
    up_weight = up_projection.weight
    delta_gate = F.linear(normalized_directions, gate_weight, bias=None)
    delta_up = F.linear(normalized_directions, up_weight, bias=None)
    gate_path = silu_derivative(gate).unsqueeze(0) * delta_gate * up.unsqueeze(0)
    up_path = F.silu(gate).unsqueeze(0) * delta_up
    activation_directions = gate_path + up_path
    output_directions = F.linear(
        activation_directions, down_projection.weight, bias=None
    )

    def full_map(value):
        normalized_value = norm(value)
        activation = F.silu(gate_projection(normalized_value)) * up_projection(
            normalized_value
        )
        return down_projection(activation)

    def full_jvp(direction):
        return torch.func.jvp(full_map, (z,), (direction,))[1]

    exact = torch.vmap(full_jvp)(writes)
    relative_error = float(
        (exact - output_directions).norm()
        / exact.norm().clamp_min(float(eps))
    )
    return SwiGLULocalComponents(
        normalized_directions=normalized_directions,
        gate_preactivation_directions=delta_gate,
        up_preactivation_directions=delta_up,
        gate_path=gate_path,
        up_path=up_path,
        activation_directions=activation_directions,
        output_directions=output_directions,
        product_rule_relative_error=relative_error,
    )


def finite_swiglu_components(
    *,
    norm: Callable[[torch.Tensor], torch.Tensor],
    gate_projection: torch.nn.Module,
    up_projection: torch.nn.Module,
    down_projection: torch.nn.Module,
    z: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-12,
) -> SwiGLUFiniteComponents:
    if z.shape != direction.shape or z.ndim != 1:
        raise ValueError("z and direction must be matching [D] vectors")
    normalized0 = norm(z)
    normalized1 = norm(z + direction)
    gate0 = gate_projection(normalized0)
    gate1 = gate_projection(normalized1)
    up0 = up_projection(normalized0)
    up1 = up_projection(normalized1)
    activated_gate0 = F.silu(gate0)
    activated_gate_change = F.silu(gate1) - activated_gate0
    up_change = up1 - up0
    gate_path = activated_gate_change * up0
    up_path = activated_gate0 * up_change
    interaction = activated_gate_change * up_change
    activation_change = F.silu(gate1) * up1 - activated_gate0 * up0
    reconstructed = gate_path + up_path + interaction
    output_change = F.linear(
        activation_change, down_projection.weight, bias=None
    )
    relative_error = float(
        (reconstructed - activation_change).norm()
        / activation_change.norm().clamp_min(float(eps))
    )
    return SwiGLUFiniteComponents(
        gate_path=gate_path,
        up_path=up_path,
        interaction=interaction,
        total_activation_change=activation_change,
        output_change=output_change,
        completeness_relative_error=relative_error,
    )


def neuron_target_contributions(
    activation_directions: torch.Tensor,
    down_projection: torch.nn.Module,
    target_residual_gradient: torch.Tensor,
) -> torch.Tensor:
    """Signed per-neuron target contribution for every visual direction."""
    if activation_directions.ndim != 2:
        raise ValueError("activation_directions must have shape [M,Dff]")
    if target_residual_gradient.ndim != 1:
        raise ValueError("target residual gradient must be one-dimensional")
    neuron_covector = down_projection.weight.T @ target_residual_gradient
    if int(neuron_covector.numel()) != int(activation_directions.shape[-1]):
        raise ValueError("Down projection and activation widths disagree")
    return activation_directions * neuron_covector.unsqueeze(0)
