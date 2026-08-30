"""Riesz and finite-path attribution for current-block visual FFN writes.

The functions in this module deliberately know nothing about a particular
vision-language model.  Callers provide

* ``ffn_map(z) = FFN(Norm(z))`` (the residual identity is excluded), and
* a branch-isolated downstream scalar ``score_from_ffn_output(m)`` whose
  residual skip is already fixed by the caller.

This separation makes the mathematical identities testable in FP32/FP64
without loading a model and prevents an attribution helper from silently
changing the causal estimand.  A visual write is always conditional on the
observed clean attention pattern; removing it is not a complete image-patch
intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import torch


TensorFunction = Callable[[torch.Tensor], torch.Tensor]
QuadratureMethod = Literal["trapezoid", "gauss_legendre"]


@dataclass(frozen=True)
class QuadratureRule:
    """Nodes and weights on ``[0, 1]``.

    ``K=1`` is intentionally the clean-endpoint local Jacobian requested by
    the experiment protocol, rather than a one-node approximation to the
    integral.  It is therefore tagged ``is_local=True``.
    """

    method: QuadratureMethod
    integration_points: int
    nodes: torch.Tensor
    weights: torch.Tensor
    is_local: bool


@dataclass(frozen=True)
class LocalRieszResult:
    ffn_output: torch.Tensor
    target_gradient: torch.Tensor
    pullback: torch.Tensor
    response_vectors: torch.Tensor | None
    token_scores: torch.Tensor
    total_score: torch.Tensor
    direct_pairing_scores: torch.Tensor | None
    duality_max_abs_error: float | None
    additivity_abs_error: float


@dataclass(frozen=True)
class VectorPathResult:
    components: torch.Tensor
    total_finite_effect: torch.Tensor
    component_sum: torch.Tensor
    completeness_relative_error: float
    quadrature: QuadratureRule


@dataclass(frozen=True)
class ScalarPathResult:
    token_scores: torch.Tensor
    total_finite_effect: torch.Tensor
    score_sum: torch.Tensor
    completeness_absolute_error: float
    completeness_relative_error: float
    quadrature: QuadratureRule
    pullbacks: torch.Tensor | None
    target_gradients: torch.Tensor | None


@dataclass(frozen=True)
class SymmetricDifferenceResult:
    eta: float
    predicted_derivative: float
    positive_one_sided: float
    negative_one_sided: float
    symmetric_derivative: float
    curvature: float
    relative_linearization_error: float
    center: float
    plus: float
    minus: float
    zero_observed: bool


def _require_case_shapes(z: torch.Tensor, writes: torch.Tensor) -> None:
    if z.ndim != 1:
        raise ValueError(f"z must be one residual vector [D], got {tuple(z.shape)}")
    if writes.ndim != 2 or int(writes.shape[-1]) != int(z.numel()):
        raise ValueError(
            f"writes must have shape [M,{z.numel()}], got {tuple(writes.shape)}"
        )
    if int(writes.shape[0]) <= 0:
        raise ValueError("writes must contain at least one visual token")
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(writes).all()):
        raise ValueError("z and writes must be finite")


def quadrature_rule(
    method: QuadratureMethod,
    integration_points: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> QuadratureRule:
    """Construct a deterministic quadrature rule on ``[0,1]``."""
    count = int(integration_points)
    if count <= 0:
        raise ValueError("integration_points must be positive")
    if method not in {"trapezoid", "gauss_legendre"}:
        raise ValueError(f"Unsupported quadrature method {method!r}")
    if count == 1:
        values = torch.ones(1, device=device, dtype=dtype)
        return QuadratureRule(method, count, values, values.clone(), True)
    if method == "trapezoid":
        nodes = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
        weights = torch.full(
            (count,), 1.0 / float(count - 1), device=device, dtype=dtype
        )
        weights[0] *= 0.5
        weights[-1] *= 0.5
    else:
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
        nodes = torch.as_tensor(
            (raw_nodes + 1.0) * 0.5, device=device, dtype=dtype
        )
        weights = torch.as_tensor(
            raw_weights * 0.5, device=device, dtype=dtype
        )
    return QuadratureRule(method, count, nodes, weights, False)


def target_scalar_from_logits(
    logits: torch.Tensor,
    *,
    target_token_id: int,
    scalar: Literal["logit", "margin", "log_probability"],
    competitor_token_id: int | None = None,
) -> torch.Tensor:
    """Evaluate one of the three preregistered target functionals."""
    if logits.ndim != 1:
        raise ValueError("target logits must be a one-dimensional vocabulary row")
    target = int(target_token_id)
    if target < 0 or target >= int(logits.numel()):
        raise IndexError(f"target token {target} outside vocabulary")
    if scalar == "logit":
        return logits[target]
    if scalar == "log_probability":
        return torch.log_softmax(logits, dim=-1)[target]
    if scalar == "margin":
        if competitor_token_id is None:
            raise ValueError("margin requires a fixed clean competitor token")
        competitor = int(competitor_token_id)
        if competitor == target:
            raise ValueError("fixed competitor must differ from target")
        if competitor < 0 or competitor >= int(logits.numel()):
            raise IndexError(f"competitor token {competitor} outside vocabulary")
        return logits[target] - logits[competitor]
    raise ValueError(f"Unknown target scalar {scalar!r}")


def batched_scalar_gradients(
    scalar_values: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiate several scalar families in one batched VJP.

    Each mapping value may be a scalar or a batch of independent per-example
    scalars.  Summing over examples preserves one gradient row per example
    because the downstream graph is batch-separable, while ``is_grads_batched``
    evaluates the different scalar families together.
    """
    names = list(scalar_values)
    if not names:
        raise ValueError("scalar_values must not be empty")
    # PyTorch's batched-VJP path introduces a three-wide vmap dimension through
    # the full decoder.  That is fast and stable in FP32/FP64, but on native
    # BF16/FP16 model graphs it both raises peak memory substantially and can
    # change low-precision accumulation order.  Preserve the original serial
    # VJP semantics for low-precision leaves; path *nodes* can still be batched.
    if inputs.dtype in {torch.bfloat16, torch.float16}:
        return {
            name: torch.autograd.grad(
                scalar_values[name].sum(),
                inputs,
                retain_graph=index + 1 < len(names),
            )[0].detach()
            for index, name in enumerate(names)
        }

    totals = torch.stack([scalar_values[name].sum() for name in names])
    basis = torch.eye(
        len(names), device=totals.device, dtype=totals.dtype
    )
    stacked = torch.autograd.grad(
        totals,
        inputs,
        grad_outputs=basis,
        is_grads_batched=True,
    )[0]
    return {name: stacked[index].detach() for index, name in enumerate(names)}


def fixed_clean_competitor(logits: torch.Tensor, target_token_id: int) -> int:
    """Return the strongest non-target token from the unperturbed logits."""
    if logits.ndim != 1:
        raise ValueError("clean logits must be one-dimensional")
    target = int(target_token_id)
    masked = logits.detach().clone()
    masked[target] = -torch.inf
    return int(masked.argmax().item())


def batched_ffn_jvps(
    ffn_map: TensorFunction,
    z: torch.Tensor,
    directions: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Return ``J_G(z) directions[m]`` for all directions."""
    _require_case_shapes(z, directions)
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive or None")

    def one(direction: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(ffn_map, (z,), (direction,))[1]

    with torch.enable_grad():
        return torch.vmap(one, chunk_size=chunk_size)(directions)


def ffn_pullback(
    ffn_map: TensorFunction,
    z: torch.Tensor,
    target_gradient: torch.Tensor,
) -> torch.Tensor:
    """Return ``J_G(z)^T target_gradient`` with one VJP."""
    if z.ndim != 1 or target_gradient.shape != z.shape:
        raise ValueError("z and target_gradient must be matching [D] vectors")
    with torch.enable_grad():
        _output, vjp = torch.func.vjp(ffn_map, z)
        return vjp(target_gradient)[0]


def scalar_gradient(function: TensorFunction, value: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar-valued function at one vector."""
    with torch.enable_grad():
        output, vjp = torch.func.vjp(function, value)
        if output.numel() != 1:
            raise ValueError(
                "score_from_ffn_output must return exactly one scalar, got "
                f"shape {tuple(output.shape)}"
            )
        return vjp(torch.ones_like(output))[0]


def local_riesz_attribution(
    *,
    ffn_map: TensorFunction,
    score_from_ffn_output: TensorFunction,
    z: torch.Tensor,
    writes: torch.Tensor,
    save_response_vectors: bool = False,
    validate_duality: bool = True,
    chunk_size: int | None = None,
) -> LocalRieszResult:
    """Compute local target attribution with one downstream VJP and pullback.

    The token score is evaluated efficiently as ``writes @ (J_G^T g)``.
    Complete response vectors are materialized only when explicitly requested
    or when duality validation requires them.
    """
    _require_case_shapes(z, writes)
    ffn_output = ffn_map(z)
    if ffn_output.shape != z.shape:
        raise ValueError("ffn_map must preserve residual width")
    gradient = scalar_gradient(score_from_ffn_output, ffn_output)
    pullback = ffn_pullback(ffn_map, z, gradient)
    token_scores = writes @ pullback
    total = token_scores.sum()
    direct = None
    responses = None
    duality_error = None
    if save_response_vectors or validate_duality:
        responses = batched_ffn_jvps(
            ffn_map, z, writes, chunk_size=chunk_size
        )
        direct = responses @ gradient
        duality_error = float((direct - token_scores).abs().max().detach().cpu())
        if not save_response_vectors:
            responses = None
    aggregate_direct = torch.dot(pullback, writes.sum(dim=0))
    additivity_error = float((total - aggregate_direct).abs().detach().cpu())
    return LocalRieszResult(
        ffn_output=ffn_output,
        target_gradient=gradient,
        pullback=pullback,
        response_vectors=responses,
        token_scores=token_scores,
        total_score=total,
        direct_pairing_scores=direct,
        duality_max_abs_error=duality_error,
        additivity_abs_error=additivity_error,
    )


def source_scaling_autograd_scores(
    *,
    ffn_map: TensorFunction,
    score_from_ffn_output: TensorFunction,
    z: torch.Tensor,
    writes: torch.Tensor,
) -> torch.Tensor:
    """Differentiate explicit clean-source scaling at ``lambda=1``.

    ``z(lambda) = z + sum_m (lambda_m - 1) a_m``.
    """
    _require_case_shapes(z, writes)
    lambdas = torch.ones(
        int(writes.shape[0]), device=z.device, dtype=z.dtype, requires_grad=True
    )
    scaled_z = z + torch.einsum("m,md->d", lambdas - 1.0, writes)
    score = score_from_ffn_output(ffn_map(scaled_z))
    return torch.autograd.grad(score, lambdas)[0]


def vector_path_components(
    *,
    ffn_map: TensorFunction,
    z: torch.Tensor,
    writes: torch.Tensor,
    method: QuadratureMethod,
    integration_points: int,
    baseline_scale: float = 0.0,
    chunk_size: int | None = None,
    eps: float = 1e-12,
) -> VectorPathResult:
    """Integrate every vector component along the aggregate-write path.

    ``baseline_scale=0`` is the current-block zero-write baseline
    ``z0=z-sum(writes)``.  A value in ``[0,1)`` supplies a scaled-write
    baseline while keeping the endpoint equal to clean ``z``.
    """
    _require_case_shapes(z, writes)
    scale = float(baseline_scale)
    if not math.isfinite(scale) or scale < 0.0 or scale >= 1.0:
        raise ValueError("baseline_scale must be finite in [0,1)")
    aggregate = writes.sum(dim=0)
    z0 = z - (1.0 - scale) * aggregate
    path_writes = (1.0 - scale) * writes
    rule = quadrature_rule(
        method, integration_points, device=z.device, dtype=z.dtype
    )
    components = torch.zeros_like(writes)
    for alpha, weight in zip(rule.nodes, rule.weights):
        point = z0 + alpha * (1.0 - scale) * aggregate
        components = components + weight * batched_ffn_jvps(
            ffn_map, point, path_writes, chunk_size=chunk_size
        )
    finite_effect = ffn_map(z) - ffn_map(z0)
    component_sum = components.sum(dim=0)
    relative_error = float(
        (
            (component_sum - finite_effect).norm()
            / finite_effect.norm().clamp_min(float(eps))
        )
        .detach()
        .cpu()
    )
    return VectorPathResult(
        components=components,
        total_finite_effect=finite_effect,
        component_sum=component_sum,
        completeness_relative_error=relative_error,
        quadrature=rule,
    )


def scalar_path_attribution(
    *,
    ffn_map: TensorFunction,
    score_from_ffn_output: TensorFunction,
    z: torch.Tensor,
    writes: torch.Tensor,
    method: QuadratureMethod,
    integration_points: int,
    baseline_scale: float = 0.0,
    save_point_vectors: bool = False,
    eps: float = 1e-12,
) -> ScalarPathResult:
    """Efficient target-conditioned path attribution.

    At each integration point this performs one downstream scalar VJP and one
    FFN pullback, then scores every token with a matrix-vector product.  It
    never performs one backward pass per visual token.
    """
    _require_case_shapes(z, writes)
    scale = float(baseline_scale)
    if not math.isfinite(scale) or scale < 0.0 or scale >= 1.0:
        raise ValueError("baseline_scale must be finite in [0,1)")
    aggregate = writes.sum(dim=0)
    z0 = z - (1.0 - scale) * aggregate
    path_writes = (1.0 - scale) * writes
    rule = quadrature_rule(
        method, integration_points, device=z.device, dtype=z.dtype
    )
    scores = torch.zeros(
        int(writes.shape[0]), device=writes.device, dtype=writes.dtype
    )
    pullbacks: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    for alpha, weight in zip(rule.nodes, rule.weights):
        point = z0 + alpha * (1.0 - scale) * aggregate
        output = ffn_map(point)
        gradient = scalar_gradient(score_from_ffn_output, output)
        pullback = ffn_pullback(ffn_map, point, gradient)
        scores = scores + weight * (path_writes @ pullback)
        if save_point_vectors:
            pullbacks.append(pullback)
            gradients.append(gradient)
    clean_score = score_from_ffn_output(ffn_map(z))
    baseline_score = score_from_ffn_output(ffn_map(z0))
    finite_effect = clean_score - baseline_score
    score_sum = scores.sum()
    absolute_error = float((score_sum - finite_effect).abs().detach().cpu())
    relative_error = float(
        (
            (score_sum - finite_effect).abs()
            / finite_effect.abs().clamp_min(float(eps))
        )
        .detach()
        .cpu()
    )
    return ScalarPathResult(
        token_scores=scores,
        total_finite_effect=finite_effect,
        score_sum=score_sum,
        completeness_absolute_error=absolute_error,
        completeness_relative_error=relative_error,
        quadrature=rule,
        pullbacks=torch.stack(pullbacks) if pullbacks else None,
        target_gradients=torch.stack(gradients) if gradients else None,
    )


def frozen_write_leave_one_out(
    *,
    ffn_map: TensorFunction,
    score_from_ffn_output: TensorFunction,
    z: torch.Tensor,
    writes: torch.Tensor,
    regions: Sequence[Sequence[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite FFN-vector and scalar effects for frozen-write regions."""
    _require_case_shapes(z, writes)
    selected_regions = (
        [[index] for index in range(int(writes.shape[0]))]
        if regions is None
        else [list(map(int, region)) for region in regions]
    )
    clean_output = ffn_map(z)
    clean_score = score_from_ffn_output(clean_output)
    vector_effects = []
    scalar_effects = []
    for region in selected_regions:
        if not region:
            raise ValueError("leave-one-out regions must not be empty")
        if min(region) < 0 or max(region) >= int(writes.shape[0]):
            raise IndexError(f"region contains invalid token index: {region}")
        region_write = writes[region].sum(dim=0)
        without = ffn_map(z - region_write)
        vector_effects.append(clean_output - without)
        scalar_effects.append(clean_score - score_from_ffn_output(without))
    return torch.stack(vector_effects), torch.stack(scalar_effects)


def direct_unembedding_scores(
    response_vectors: torch.Tensor, unembedding_direction: torch.Tensor
) -> torch.Tensor:
    """Supplementary ``w_y^T delta_m`` baseline."""
    if response_vectors.ndim != 2:
        raise ValueError("response_vectors must have shape [M,D]")
    if unembedding_direction.shape != response_vectors.shape[1:]:
        raise ValueError("unembedding direction has the wrong residual width")
    return response_vectors @ unembedding_direction


def symmetric_directional_differences(
    *,
    scalar_function: TensorFunction,
    center: torch.Tensor,
    direction: torch.Tensor,
    predicted_derivative: float | torch.Tensor,
    etas: Sequence[float],
    resolution_floor: float = 0.0,
    eps: float = 1e-12,
) -> list[SymmetricDifferenceResult]:
    """Evaluate one-/two-sided derivatives and curvature in native precision."""
    if center.shape != direction.shape:
        raise ValueError("center and direction must have matching shapes")
    predicted = float(torch.as_tensor(predicted_derivative).detach().cpu())
    with torch.no_grad():
        center_score = float(scalar_function(center).detach().cpu())
    rows = []
    for raw_eta in etas:
        eta = float(raw_eta)
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("all eta values must be finite and positive")
        with torch.no_grad():
            plus = float(scalar_function(center + eta * direction).detach().cpu())
            minus = float(scalar_function(center - eta * direction).detach().cpu())
        positive = (plus - center_score) / eta
        negative = (center_score - minus) / eta
        symmetric = (plus - minus) / (2.0 * eta)
        curvature = abs(plus - 2.0 * center_score + minus) / (eta * eta)
        relative = abs(symmetric - predicted) / max(abs(predicted), float(eps))
        observed_change = max(abs(plus - center_score), abs(minus - center_score))
        rows.append(
            SymmetricDifferenceResult(
                eta=eta,
                predicted_derivative=predicted,
                positive_one_sided=positive,
                negative_one_sided=negative,
                symmetric_derivative=symmetric,
                curvature=curvature,
                relative_linearization_error=relative,
                center=center_score,
                plus=plus,
                minus=minus,
                zero_observed=observed_change <= float(resolution_floor),
            )
        )
    return rows


def signed_mass_statistics(scores: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """Persist both signs and cancellation; never hide negative attribution."""
    flat = scores.float().reshape(-1)
    positive = flat.clamp_min(0.0).sum()
    negative = (-flat).clamp_min(0.0).sum()
    net = flat.sum()
    absolute = flat.abs().sum()
    return {
        "positive_mass": float(positive),
        "negative_mass": float(negative),
        "net_effect": float(net),
        "negative_token_fraction": float((flat < 0).float().mean()),
        "signed_cancellation_ratio": float(net.abs() / absolute.clamp_min(eps)),
    }
