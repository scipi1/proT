"""
Position-weighted, NaN-aware MSE loss for proT forecasters.

Rationale:
- Uniform averaging over sequence positions under-weights the tail of the
  target trajectory ("predictions start nicely but do not follow the target
  later on"). Position weights let final values contribute more.
- NaN targets (missing values) must not contribute to the loss. Previously
  they were zero-imputed via nan_to_num and included, which biases
  predictions towards zero at missing positions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_position_weights(
    seq_len: int,
    scheme: str = "none",
    alpha: float = 2.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Build per-position loss weights, normalized to mean 1 so the loss scale
    (and the tuned learning rate) stays comparable to the unweighted case.

    Args:
        seq_len: Sequence length L.
        scheme: "none" (uniform), "linear" (w_i ~ (i+1)/L) or
            "exp" (w_i ~ exp(alpha * i / (L-1))).
        alpha: Sharpness of the exponential ramp (only for scheme="exp").
        device: Target device.
        dtype: Target dtype.

    Returns:
        torch.Tensor: Weights of shape (L,) normalized to mean 1.
    """
    if scheme is None:
        scheme = "none"

    if scheme == "none":
        w = torch.ones(seq_len, dtype=dtype or torch.float32, device=device)
    elif scheme == "linear":
        w = torch.arange(1, seq_len + 1, dtype=dtype or torch.float32, device=device) / seq_len
    elif scheme == "exp":
        if seq_len > 1:
            t = torch.arange(seq_len, dtype=dtype or torch.float32, device=device) / (seq_len - 1)
        else:
            t = torch.zeros(1, dtype=dtype or torch.float32, device=device)
        w = torch.exp(alpha * t)
    else:
        raise ValueError(
            f"Unknown position weighting scheme: {scheme}. Choose 'none', 'linear' or 'exp'."
        )

    # normalize to mean 1 to keep the loss scale comparable
    return w / w.mean()


class PositionWeightedMSE(nn.Module):
    """
    MSE loss with per-position weighting and NaN-target masking.

    Only valid (non-NaN) targets contribute to the loss. Position weights
    are cached per (sequence length, device, dtype) to avoid rebuilding
    them at every step.

    Args:
        scheme: Weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
    """

    def __init__(self, scheme: str = "none", alpha: float = 2.0):
        super().__init__()
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: Predictions of shape (B, L).
            target: Raw targets of shape (B, L), may contain NaN (missing).

        Returns:
            torch.Tensor: Scalar loss, weighted mean over valid targets.
        """
        valid = ~torch.isnan(target)
        mse = self.mse(predicted, torch.nan_to_num(target))
        w = self._get_weights(predicted.shape[1], predicted.device, predicted.dtype)
        w_valid = w.view(1, -1) * valid
        denom = w_valid.sum().clamp_min(1e-8)
        return (mse * w_valid).sum() / denom


class GaussianNLL(nn.Module):
    """
    Standard Gaussian negative log-likelihood for scalar (non-censored) targets.

    Model: y ~ N(mu, sigma) per target. All targets are observed (no NaN
    censoring). The loss is a weighted mean over all targets.

    Args:
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        eps: Minimum sigma (added after softplus) for numerical stability.
        var_reg: Weight for the variance regularization term (penalizes
            raw_sigma^2). Prevents the model from inflating the variance to
            reduce the NLL. Default 0.0 (disabled).
    """

    def __init__(
        self,
        scheme: str = "none",
        alpha: float = 2.0,
        eps: float = 1e-3,
        var_reg: float = 0.0,
    ):
        super().__init__()
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.eps = eps
        self.var_reg = var_reg
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean, shape (B, D).
            raw_sigma: Raw scale parameter, shape (B, D); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, D); NaN marks missing values (excluded).

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all targets.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)

        # Gaussian NLL of the exact value
        nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
            + 0.5 * ((torch.nan_to_num(target) - mu) / sigma) ** 2

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        denom = (w * valid.float()).sum().clamp_min(1e-8)
        loss = (nll * w * valid.float()).sum() / denom

        # variance regularization: penalize large raw_sigma to prevent
        # the model from inflating the variance to reduce the NLL
        if self.var_reg > 0:
            loss = loss + self.var_reg * torch.mean(raw_sigma ** 2)

        return loss


class CensoredGaussianNLL(nn.Module):
    """
    Censored Gaussian negative log-likelihood (Tobit) for right-censored targets.

    Model: y ~ N(mu, sigma) per position. NaN targets are informative
    right-censoring events ("the value reached the failure threshold"), not
    missing-at-random data. Each position contributes its exact likelihood:

    - observed position:  log N(y; mu, sigma)
    - censored position:  log P(y >= threshold) = log_ndtr((mu - threshold) / sigma)

    With bce_on_observed=True, observed positions additionally contribute the
    threshold-side label -log P(y < threshold) = -log_ndtr(-(mu-threshold)/sigma),
    which recovers the "derived mask + BCE on all positions" variant.

    The failure mask is derived from the value distribution by construction:
    p_fail = ndtr((mu - threshold) / sigma), so the two heads cannot disagree.

    The loss is a weighted mean over ALL positions (observed and censored),
    so both terms share one denominator and no balancing weight is needed.

    Args:
        threshold: Censoring threshold (1.0 for normalized IST data).
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        bce_on_observed: Add the threshold-side term on observed positions.
        eps: Minimum sigma (added after softplus) for numerical stability.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        scheme: str = "none",
        alpha: float = 2.0,
        bce_on_observed: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.threshold = threshold
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.bce_on_observed = bce_on_observed
        self.eps = eps
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def fail_probability(mu: torch.Tensor, sigma: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """Derived failure mask: P(y >= threshold) under N(mu, sigma)."""
        return torch.special.ndtr((mu - threshold) / sigma)

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean, shape (B, L).
            raw_sigma: Raw scale parameter, shape (B, L); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, L); NaN marks right-censored positions.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all positions.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)
        z = (mu - self.threshold) / sigma

        # observed: Gaussian NLL of the exact value
        nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
            + 0.5 * ((torch.nan_to_num(target) - mu) / sigma) ** 2

        # optional threshold-side label on observed positions (Option 2 flavor)
        if self.bce_on_observed:
            nll = nll - torch.special.log_ndtr(-z)

        # censored: -log P(y >= threshold)
        nll = torch.where(valid, nll, -torch.special.log_ndtr(z))

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        denom = (w * torch.ones_like(valid, dtype=nll.dtype)).sum().clamp_min(1e-8)
        return (nll * w).sum() / denom
