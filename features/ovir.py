"""Operator innovation of FFN responses on visual attention WRITE subspaces."""

from __future__ import annotations

from typing import Any

import torch


def visual_write_energy_basis(
    writes: torch.Tensor,
    *,
    energy_threshold: float = 0.95,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """Return an FP64 basis for the smallest raw-WRITE energy subspace.

    ``writes`` is ``[hidden, visual_tokens]``. Columns are deliberately not
    normalized: the SVD energy is the energy of the actual attention WRITEs.
    A Gram eigensolve avoids a full ``hidden x visual_tokens`` SVD. The final
    thin QR makes projection diagnostics insensitive to Gram roundoff.
    """
    if writes.ndim != 2:
        raise ValueError("Expected [hidden, visual_tokens] WRITEs")
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1]")
    a = writes.double()
    if not torch.isfinite(a).all():
        raise ValueError("Nonfinite visual WRITEs")
    hidden, tokens = map(int, a.shape)
    total = a.square().sum()
    if tokens == 0 or not bool(total > 0):
        empty = a.new_empty((hidden, 0))
        audit = dict(
            rank95=0,
            visual_tokens=tokens,
            source_energy=float(total),
            retained_energy=float("nan"),
            previous_retained_energy=float("nan"),
            projection_residual_energy=float("nan"),
            basis_orthogonality_error=0.0,
            basis_energy_identity_error=0.0,
            max_singular_value=0.0,
            min_kept_singular_value=float("nan"),
            zero_input=True,
        )
        factors = dict(
            right=a.new_empty((tokens, 0)),
            singular_values=a.new_empty((0,)),
            qr_r=a.new_empty((0, 0)),
        )
        return empty, audit, factors

    gram = a.T @ a
    eigenvalues, right = torch.linalg.eigh(gram)
    largest = eigenvalues[-1]
    if bool(eigenvalues[0] < -1e-10 * largest):
        raise ValueError("Invalid visual-WRITE Gram spectrum")
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    right = right.flip(1)
    cumulative = eigenvalues.cumsum(0) / eigenvalues.sum()
    rank = int(torch.searchsorted(
        cumulative,
        cumulative.new_tensor(float(energy_threshold)),
        right=False,
    ).item()) + 1
    kept_values = eigenvalues[:rank]
    singular = kept_values.sqrt()
    if not bool((singular > 0).all()):
        raise ValueError("Selected visual-WRITE subspace contains a zero singular value")
    kept_right = right[:, :rank]
    raw_basis = (a @ kept_right) / singular[None, :]
    basis, qr_r = torch.linalg.qr(raw_basis, mode="reduced")

    eye = torch.eye(rank, device=a.device, dtype=a.dtype)
    orthogonality = (basis.T @ basis - eye).norm()
    projected = basis.T @ a
    retained = projected.square().sum() / total
    residual = a - basis @ projected
    residual_fraction = residual.square().sum() / total
    identity_error = (retained + residual_fraction - 1).abs()
    previous = eigenvalues[: rank - 1].sum() / eigenvalues.sum() if rank > 1 else total.new_zeros(())
    if float(orthogonality) > 1e-9:
        raise ValueError(f"Nonorthogonal visual-WRITE basis: {float(orthogonality)}")
    if float(retained) + 1e-10 < float(energy_threshold):
        raise ValueError("Visual-WRITE basis misses the requested energy")
    if rank > 1 and float(previous) >= float(energy_threshold) + 1e-10:
        raise ValueError("Visual-WRITE basis rank is not minimal")
    if float(identity_error) > 1e-9:
        raise ValueError("Visual-WRITE projection energy identity failed")

    audit = dict(
        rank95=rank,
        visual_tokens=tokens,
        source_energy=float(total),
        retained_energy=float(retained),
        previous_retained_energy=float(previous),
        projection_residual_energy=float(residual_fraction),
        basis_orthogonality_error=float(orthogonality),
        basis_energy_identity_error=float(identity_error),
        max_singular_value=float(singular[0]),
        min_kept_singular_value=float(singular[-1]),
        zero_input=False,
    )
    factors = dict(right=kept_right, singular_values=singular, qr_r=qr_r)
    return basis, audit, factors


def operator_visual_write_innovation(
    basis: torch.Tensor,
    response: torch.Tensor,
    *,
    zero_tolerance: float = 1e-30,
) -> dict[str, Any]:
    """Compute OVIR from an orthonormal basis and its operator responses."""
    q, y = basis.double(), response.double()
    if q.ndim != 2 or y.shape != q.shape or not torch.isfinite(q).all() or not torch.isfinite(y).all():
        raise ValueError("Expected matching finite [hidden, rank] basis/responses")
    rank = int(q.shape[1])
    if rank == 0:
        return dict(
            ovir=float("nan"),
            operator_total_energy=0.0,
            operator_inside_energy=0.0,
            operator_outside_energy=0.0,
            operator_energy_identity_error=0.0,
            zero_operator=True,
        )
    eye = torch.eye(rank, device=q.device, dtype=q.dtype)
    if float((q.T @ q - eye).norm()) > 1e-8:
        raise ValueError("OVIR basis is not orthonormal")
    coordinates = q.T @ y
    inside = coordinates.square().sum()
    outside_matrix = y - q @ coordinates
    outside = outside_matrix.square().sum()
    total = y.square().sum()
    identity_error = (total - inside - outside).abs() / total.clamp_min(float(zero_tolerance))
    if float(identity_error) > 1e-9:
        raise ValueError("OVIR orthogonal energy identity failed")
    if not bool(total > float(zero_tolerance)):
        value = float("nan")
        zero = True
    else:
        value = float((outside / total).clamp(0, 1))
        zero = False
    return dict(
        ovir=value,
        operator_total_energy=float(total),
        operator_inside_energy=float(inside),
        operator_outside_energy=float(outside),
        operator_energy_identity_error=float(identity_error),
        zero_operator=zero,
    )


def response_from_token_matrix(
    token_responses: torch.Tensor,
    factors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute ``J Q`` as ``E V Sigma^-1 R^-1`` for audit parity.

    The QR relation is ``A V Sigma^-1 = Q R``. Therefore token responses
    ``E=J A`` imply ``J Q = E V Sigma^-1 R^-1``.
    """
    e = token_responses.double()
    right = factors["right"].double()
    singular = factors["singular_values"].double()
    qr_r = factors["qr_r"].double()
    if e.ndim != 2 or e.shape[1] != right.shape[0]:
        raise ValueError("Token response matrix is incompatible with the visual basis")
    if right.shape[1] == 0:
        return e.new_empty((e.shape[0], 0))
    raw_response = (e @ right) / singular[None, :]
    return torch.linalg.solve_triangular(
        qr_r.T,
        raw_response.T,
        upper=False,
    ).T
