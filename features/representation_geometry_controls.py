"""Reference-calibrated cosine and coordinate-geometry controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WhiteningCalibration:
    mean: torch.Tensor
    whitening: torch.Tensor
    eigenvalue_floor: float


def row_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left/right must have matching [N,D] shapes")
    return F.cosine_similarity(left, right, dim=-1, eps=float(eps))


def centered_cosine(
    left: torch.Tensor, right: torch.Tensor, reference_mean: torch.Tensor
) -> torch.Tensor:
    if reference_mean.shape != left.shape[-1:]:
        raise ValueError("reference_mean must have residual width D")
    return row_cosine(left - reference_mean, right - reference_mean)


def residualize_direction(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or direction.shape != values.shape[-1:]:
        raise ValueError("Expected values[N,D] and direction[D]")
    unit = direction / direction.norm().clamp_min(1e-12)
    return values - (values @ unit).unsqueeze(-1) * unit.unsqueeze(0)


def fit_whitening(
    train_values: torch.Tensor, eigenvalue_floor: float = 1e-5
) -> WhiteningCalibration:
    if train_values.ndim != 2 or int(train_values.shape[0]) < 2:
        raise ValueError("Whitening requires at least two train rows")
    floor = float(eigenvalue_floor)
    if floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    values = train_values.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = eigenvalues.clamp_min(floor).rsqrt()
    whitening = eigenvectors @ torch.diag(scale) @ eigenvectors.T
    return WhiteningCalibration(
        mean=mean.to(train_values),
        whitening=whitening.to(train_values),
        eigenvalue_floor=floor,
    )


def whiten(values: torch.Tensor, calibration: WhiteningCalibration) -> torch.Tensor:
    return (values - calibration.mean) @ calibration.whitening


def mahalanobis_cosine(
    left: torch.Tensor, right: torch.Tensor, calibration: WhiteningCalibration
) -> torch.Tensor:
    return row_cosine(whiten(left, calibration), whiten(right, calibration))


def empirical_cdf_scores(
    values: torch.Tensor, train_reference: torch.Tensor
) -> torch.Tensor:
    """Train-only empirical CDF with mid-rank handling of ties."""
    reference = train_reference.reshape(-1).sort().values
    query = values.reshape(-1)
    lower = torch.searchsorted(reference, query, right=False)
    upper = torch.searchsorted(reference, query, right=True)
    return ((lower + upper).to(torch.float64) * 0.5 / max(reference.numel(), 1)).to(values)


def matched_null_zscore(
    values: torch.Tensor, null_values: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    if null_values.ndim < 1:
        raise ValueError("null_values must have a sample dimension")
    mean = null_values.mean(dim=0)
    std = null_values.std(dim=0, unbiased=True).clamp_min(float(eps))
    return (values - mean) / std


def transform_covector(covector: torch.Tensor, coordinate_map: torch.Tensor) -> torch.Tensor:
    """For x'=Mx, return the covector coordinates M^{-T}g."""
    return torch.linalg.solve(coordinate_map.T, covector)
