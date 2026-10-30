"""
Probabilistic I-spline head with closed-form Kalman conditioning.

Replaces the recurrent rollout of the gru_per_var head by a functional basis:
for each target variable v, a dedicated MLP maps the variable's pooled context
to a Gaussian over K spline coefficients

    a ~ N(m, S),  m = softplus(raw_m) >= 0,
                  S = diag(d) + U U^T,  d = softplus(raw_d) + var_min,
                                        U = softplus(raw_U) >= 0 (K x rank)

and the native-space trajectory (log(y + shift) for log-space losses, y
otherwise) is z(t) = z0 + phi(t)^T a + e, e ~ N(0, r), with phi(t) the
I-spline basis (non-decreasing, 0 at t = 0, 1 at t = 1) and z0 the
native-space value of y = 0. Prior guarantees:
    - mean z0 + phi^T m is non-decreasing (phi non-decreasing, m >= 0)
    - variance phi^T S phi + r is non-decreasing (d >= 0, U >= 0)

Online measurements (given positions) condition the coefficients exactly
(linear-Gaussian model, information form, K x K algebra only):
    P = S^-1 + Phi_o^T Phi_o / r,  S' = P^-1,
    m' = S' (S^-1 m + Phi_o^T (z_o - z0) / r)
The low-rank factor U correlates the coefficients so that early measurements
move the late part of the curve; it is trained by the masked-conditioning
curriculum (loss on hidden positions after conditioning). After conditioning
the mean is no longer guaranteed monotone.

Time is t = position / t_max with a global t_max (shared physical axis). The
basis is tabulated on a fine grid in [0, 1] and linearly interpolated.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bspline_basis(t: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Cox-de Boor B-spline basis (G, n_basis) on a clamped knot vector."""
    n_basis = len(knots) - degree - 1
    B = np.zeros((len(t), len(knots) - 1))
    for i in range(len(knots) - 1):
        if knots[i + 1] > knots[i]:
            B[:, i] = (t >= knots[i]) & (t < knots[i + 1])
    # close the last non-empty span at the right end
    last = np.max(np.nonzero(np.diff(knots) > 0)[0])
    B[t >= knots[-1], last] = 1.0
    for p in range(1, degree + 1):
        B_new = np.zeros((len(t), len(knots) - 1 - p))
        for i in range(len(knots) - 1 - p):
            den1 = knots[i + p] - knots[i]
            den2 = knots[i + p + 1] - knots[i + 1]
            left = (t - knots[i]) / den1 * B[:, i] if den1 > 0 else 0.0
            right = (knots[i + p + 1] - t) / den2 * B[:, i + 1] if den2 > 0 else 0.0
            B_new[:, i] = left + right
        B = B_new
    return B[:, :n_basis]


def ispline_basis(n_basis: int, n_grid: int = 1024, degree: int = 3,
                  knot_power: float = 1.0) -> np.ndarray:
    """Tabulated I-spline basis (n_grid, n_basis) on a uniform grid in [0, 1].

    Each I-spline is the normalized running integral of a B-spline:
    non-decreasing, 0 at t = 0, 1 at t = 1.

    Args:
        n_basis: number of basis functions K (>= degree + 1)
        n_grid: number of grid points
        degree: degree of the integrated B-splines
        knot_power: interior knot warp u -> 1 - (1 - u)^p; p > 1 places more
            knots near t = 1 (late failures), p = 1 is uniform
    """
    assert n_basis >= degree + 1, f"n_basis must be >= {degree + 1}"
    n_int = n_basis - degree - 1
    u = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
    interior = 1.0 - (1.0 - u) ** knot_power
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    # trapezoidal integration on an 8x finer grid, then subsample
    t = np.linspace(0.0, 1.0, 8 * (n_grid - 1) + 1)
    B = bspline_basis(t, knots, degree)
    dt = t[1] - t[0]
    I = np.concatenate([np.zeros((1, n_basis)),
                        np.cumsum(0.5 * (B[1:] + B[:-1]) * dt, axis=0)], axis=0)
    I = I / I[-1:].clip(min=1e-12)
    return I[::8]


class SplineHead(nn.Module):
    """Per-variable probabilistic I-spline head (see module docstring).

    Args:
        n_vars: number of target variables
        ctx_dim: per-variable context dimension
        n_basis: number of spline coefficients K
        rank: rank of the covariance factor U (0 = diagonal: early
            measurements barely inform distant coefficients)
        hidden: MLP hidden width
        var_min: floor on the diagonal coefficient variance
        noise_min: floor on the observation-noise variance r
        knot_power: interior knot warp (see ispline_basis)
        t_max: global horizon (max position); None = set by the owner before
            the first forward (inferred from the target shape); persisted
        n_grid: resolution of the tabulated basis
    """

    def __init__(self, n_vars: int, ctx_dim: int, n_basis: int = 16,
                 rank: int = 2, hidden: int = 128, var_min: float = 1e-4,
                 noise_min: float = 1e-4, knot_power: float = 1.0,
                 t_max: Optional[float] = None, n_grid: int = 1024):
        super().__init__()
        self.n_vars = n_vars
        self.K = n_basis
        self.rank = rank
        self.var_min = var_min
        self.noise_min = noise_min
        n_out = 2 * n_basis + n_basis * rank + 1
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(ctx_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, n_out)) for _ in range(n_vars)])
        for mlp in self.mlp:
            # small init: near-flat curve, moderate variance, small noise
            last = mlp[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
            last.bias.data[:n_basis] = -2.0
            last.bias.data[n_basis:2 * n_basis] = -4.0
            last.bias.data[2 * n_basis:-1] = -3.0
            last.bias.data[-1] = -6.0
        grid = torch.tensor(ispline_basis(n_basis, n_grid, knot_power=knot_power),
                            dtype=torch.float32)
        self.register_buffer("grid", grid, persistent=False)   # (G, K)
        self.register_buffer("t_max", torch.tensor(float(t_max or 0.0)))

    def basis(self, pos: torch.Tensor) -> torch.Tensor:
        """Interpolated basis rows (..., K); NaN positions give zero rows."""
        t = (torch.nan_to_num(pos, nan=0.0) / self.t_max).clamp(0.0, 1.0)
        G = self.grid.size(0)
        x = t * (G - 1)
        i0 = x.floor().long().clamp(0, G - 2)
        w = (x - i0.to(x.dtype)).unsqueeze(-1)
        phi = (1 - w) * self.grid[i0] + w * self.grid[i0 + 1]
        return phi * (~torch.isnan(pos)).unsqueeze(-1).to(phi.dtype)

    def params(self, context: torch.Tensor):
        """Coefficient prior and noise from context (B, V, ctx_dim).

        Returns m (B, V, K), S (B, V, K, K), r (B, V).
        """
        raw = torch.stack([mlp(context[:, v]) for v, mlp in enumerate(self.mlp)],
                          dim=1)
        K, R = self.K, self.rank
        m = F.softplus(raw[..., :K])
        S = torch.diag_embed(F.softplus(raw[..., K:2 * K]) + self.var_min)
        if R > 0:
            U = F.softplus(raw[..., 2 * K:2 * K + K * R]).unflatten(-1, (K, R))
            S = S + U @ U.transpose(-1, -2)
        r = F.softplus(raw[..., -1]) + self.noise_min
        return m, S, r

    @staticmethod
    def condition(m, S, r, phi, dz, w):
        """Exact Gaussian conditioning of the coefficients (information form).

        Args:
            m: (B, K) prior mean; S: (B, K, K) prior covariance; r: (B,) noise
            phi: (B, L, K) basis rows; dz: (B, L) observed z - z0
            w: (B, L) 1.0 where the observation is used, 0.0 elsewhere

        Returns:
            posterior mean (B, K), posterior covariance (B, K, K)
        """
        dt = m.dtype
        m, S, r, phi, dz, w = (x.double() for x in (m, S, r, phi, dz, w))
        eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        S_inv = torch.cholesky_inverse(torch.linalg.cholesky(S + 1e-10 * eye))
        phi_w = phi * (w / r.unsqueeze(-1)).unsqueeze(-1)
        P = S_inv + phi_w.transpose(1, 2) @ phi
        S_post = torch.cholesky_inverse(torch.linalg.cholesky(P))
        b = (S_inv @ m.unsqueeze(-1)).squeeze(-1) \
            + (phi_w * torch.nan_to_num(dz).unsqueeze(-1)).sum(1)
        m_post = (S_post @ b.unsqueeze(-1)).squeeze(-1)
        return m_post.to(dt), S_post.to(dt)

    def forward(self, context, pos, var_ids, z_obs, given_obs, z0: float):
        """Predictive marginals at every target token.

        Args:
            context: (B, V, ctx_dim) per-variable pooled context
            pos: (B, L) target positions (NaN = padding)
            var_ids: (B, L) variable ids 1..V (0 = padding)
            z_obs: (B, L) native-space values (used where given_obs)
            given_obs: (B, L) bool, value given AND observed (conditioning)
            z0: native-space value of y = 0

        Returns:
            mu (B, L), var (B, L): native-space predictive mean / variance
            (padding entries: mu = 0, var = 1)
        """
        if float(self.t_max) <= 0:
            raise RuntimeError("SplineHead.t_max is not set")
        m, S, r = self.params(context)
        phi = self.basis(pos)                                    # (B, L, K)
        mu = torch.zeros_like(phi[..., 0])
        var = torch.ones_like(mu)
        for v in range(self.n_vars):
            mask_v = var_ids == v + 1
            m_v, S_v, r_v = m[:, v], S[:, v], r[:, v]
            w = (given_obs & mask_v).to(phi.dtype)
            if bool(w.any()):
                m_v, S_v = self.condition(m_v, S_v, r_v, phi, z_obs - z0, w)
            mu_v = z0 + (phi @ m_v.unsqueeze(-1)).squeeze(-1)
            var_v = ((phi @ S_v) * phi).sum(-1) + r_v.unsqueeze(-1)
            mu = torch.where(mask_v, mu_v, mu)
            var = torch.where(mask_v, var_v, var)
        return mu, var
