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

    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                sample_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            predicted: Predictions of shape (B, L).
            target: Raw targets of shape (B, L), may contain NaN (missing).
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all positions of each sample.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over valid targets.
        """
        valid = ~torch.isnan(target)
        mse = self.mse(predicted, torch.nan_to_num(target))
        w = self._get_weights(predicted.shape[1], predicted.device, predicted.dtype)
        w_valid = w.view(1, -1) * valid
        if sample_weight is not None:
            w_valid = w_valid * sample_weight
        denom = w_valid.sum().clamp_min(1e-8)
        return (mse * w_valid).sum() / denom

    def per_sample(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over valid positions."""
        with torch.no_grad():
            valid = ~torch.isnan(target)
            mse = self.mse(predicted, torch.nan_to_num(target))
            w = self._get_weights(predicted.shape[1], predicted.device, predicted.dtype)
            w_valid = w.view(1, -1) * valid
            denom = w_valid.sum(dim=1).clamp_min(1e-8)
            return (mse * w_valid).sum(dim=1) / denom


class GaussianNLL(nn.Module):
    """
    Standard Gaussian negative log-likelihood for scalar (non-censored) targets.

    Model: y ~ N(mu, sigma) per target. All targets are observed (no NaN
    censoring). The loss is a weighted mean over all targets.

    Args:
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        eps: Minimum sigma (added after softplus) for numerical stability.
        var_reg: Weight for the variance regularization term. Prevents the
            model from inflating the variance to reduce the NLL.
            Default 0.0 (disabled).
        var_reg_form: "raw_sigma" penalizes raw_sigma^2 (minimum at
            raw_sigma=0, i.e. sigma ~ softplus(0)+eps); "sigma" penalizes
            sigma^2 (minimum at sigma=eps, i.e. variance is genuinely
            expensive). Default "raw_sigma" (backward compatible).
    """

    def __init__(
        self,
        scheme: str = "none",
        alpha: float = 2.0,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
    ):
        super().__init__()
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
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
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean, shape (B, D).
            raw_sigma: Raw scale parameter, shape (B, D); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, D); NaN marks missing values (excluded).
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all targets of each sample.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all targets.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)

        # Gaussian NLL of the exact value
        nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
            + 0.5 * ((torch.nan_to_num(target) - mu) / sigma) ** 2

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        w_valid = w * valid.float()
        if sample_weight is not None:
            w_valid = w_valid * sample_weight
        denom = w_valid.sum().clamp_min(1e-8)
        loss = (nll * w_valid).sum() / denom

        # variance regularization: penalize large sigma to prevent
        # the model from inflating the variance to reduce the NLL
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else sigma ** 2
            loss = loss + self.var_reg * torch.mean(pen)

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over valid targets."""
        with torch.no_grad():
            sigma = F.softplus(raw_sigma) + self.eps
            valid = ~torch.isnan(target)
            nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
                + 0.5 * ((torch.nan_to_num(target) - mu) / sigma) ** 2
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            w_valid = w * valid.float()
            denom = w_valid.sum(dim=1).clamp_min(1e-8)
            return (nll * w_valid).sum(dim=1) / denom


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
        var_reg: Weight for the variance regularization term. Default 0.0
            (disabled). Note: keep it gentle - for censored positions some
            uncertainty is real, and squeezing sigma biases the derived
            failure probability downward.
        var_reg_form: "raw_sigma" penalizes raw_sigma^2; "sigma" penalizes
            sigma^2 (minimum at sigma=eps). Default "raw_sigma".
    """

    def __init__(
        self,
        threshold: float = 1.0,
        scheme: str = "none",
        alpha: float = 2.0,
        bce_on_observed: bool = False,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
    ):
        super().__init__()
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.threshold = threshold
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.bce_on_observed = bce_on_observed
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
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
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean, shape (B, L).
            raw_sigma: Raw scale parameter, shape (B, L); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, L); NaN marks right-censored positions.
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all positions of each sample.

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
        w = w * torch.ones_like(valid, dtype=nll.dtype)
        if sample_weight is not None:
            w = w * sample_weight
        denom = w.sum().clamp_min(1e-8)
        loss = (nll * w).sum() / denom

        # variance regularization: penalize large sigma to prevent
        # the model from inflating the variance to reduce the NLL
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else sigma ** 2
            loss = loss + self.var_reg * torch.mean(pen)

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over ALL positions."""
        with torch.no_grad():
            sigma = F.softplus(raw_sigma) + self.eps
            valid = ~torch.isnan(target)
            z = (mu - self.threshold) / sigma
            nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
                + 0.5 * ((torch.nan_to_num(target) - mu) / sigma) ** 2
            if self.bce_on_observed:
                nll = nll - torch.special.log_ndtr(-z)
            nll = torch.where(valid, nll, -torch.special.log_ndtr(z))
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            denom = w.sum(dim=1).clamp_min(1e-8)
            return (nll * w).sum(dim=1) / denom


class StudentTNLL(nn.Module):
    """
    Log-space Student-t negative log-likelihood for scalar (non-censored) targets.

    Model: log(y + shift) ~ t_df(mu, sigma) per target. Same log-space
    parameterization as LogNormalNLL, but with polynomial (heavy) tails:
    extreme targets are absorbed by the tail instead of forcing sigma up.
    df -> inf recovers LogNormalNLL. The + log(y + shift) Jacobian keeps the
    loss a proper density in y-space, comparable to the other NLL losses.

    Args:
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        eps: Minimum sigma (added after softplus) for numerical stability.
        var_reg: Weight for the variance regularization term.
            Default 0.0 (disabled).
        var_reg_form: "raw_sigma" penalizes raw_sigma^2; "sigma" penalizes
            sigma^2 (minimum at sigma=eps, i.e. variance is genuinely
            expensive). Default "raw_sigma" (backward compatible).
        shift: Shift s so that log(y + s) is defined; must satisfy
            min(target) + shift > 0 over train AND test data.
        df: Degrees of freedom nu (fixed, must be > 0). Lower = heavier
            tails; nu = 4 is a common robust choice.
    """

    def __init__(
        self,
        scheme: str = "none",
        alpha: float = 2.0,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
        shift: float = 1e-3,
        df: float = 4.0,
    ):
        super().__init__()
        assert df > 0, f"df must be > 0, got {df}"
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
        self.shift = shift
        self.df = df
        # normalizing constant of the t log-density (log Gamma terms);
        # float64: the lgamma difference cancels catastrophically in float32
        self._log_c = (torch.lgamma(torch.tensor((df + 1.0) / 2.0, dtype=torch.float64))
                       - torch.lgamma(torch.tensor(df / 2.0, dtype=torch.float64))
                       - 0.5 * math.log(df * math.pi)).item()
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def point_prediction(mu: torch.Tensor, shift: float = 1e-3) -> torch.Tensor:
        """Median of the log-t in y-space (MAE-optimal point forecast)."""
        return torch.exp(mu) - shift

    def _nll(self, mu, raw_sigma, target):
        """Per-position NLL (B, L), with the log(y + shift) Jacobian."""
        sigma = F.softplus(raw_sigma) + self.eps
        log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))
        z = (log_y - mu) / sigma
        return (-self._log_c + torch.log(sigma)
                + 0.5 * (self.df + 1.0) * torch.log1p(z ** 2 / self.df) + log_y)

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted location in log-space, shape (B, D).
            raw_sigma: Raw scale parameter, shape (B, D); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, D); NaN marks missing values (excluded).
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all targets of each sample.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all targets.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)
        nll = self._nll(mu, raw_sigma, target)

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        w_valid = w * valid.float()
        if sample_weight is not None:
            w_valid = w_valid * sample_weight
        denom = w_valid.sum().clamp_min(1e-8)
        loss = (nll * w_valid).sum() / denom

        # variance regularization: penalize large sigma to prevent
        # the model from inflating the scale to reduce the NLL
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else sigma ** 2
            loss = loss + self.var_reg * torch.mean(pen)

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over valid targets."""
        with torch.no_grad():
            valid = ~torch.isnan(target)
            nll = self._nll(mu, raw_sigma, target)
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            w_valid = w * valid.float()
            denom = w_valid.sum(dim=1).clamp_min(1e-8)
            return (nll * w_valid).sum(dim=1) / denom


class CensoredLaplaceNLL(nn.Module):
    """
    Censored log-space Laplace negative log-likelihood (Laplace Tobit) for
    right-censored targets.

    Model: log(y + shift) ~ Laplace(mu, b) per position, b = softplus(raw)
    + eps. Same censoring semantics as CensoredLogNormalNLL (NaN =
    informative right-censoring at the threshold), but with exponential
    tails in log-space: heavier than the log-normal, and the Laplace CDF is
    closed-form, so the Tobit terms are exact. Each position contributes:

    - observed position:  log Laplace(log(y+s); mu, b) + log(y+s)  (Jacobian)
    - censored position:  log P(y >= threshold) = log SF(log(threshold+s))

    The failure mask is derived from the value distribution by construction:
    p_fail = SF(log(threshold+s)), so the two heads cannot disagree.

    Args:
        threshold: Censoring threshold (1.0 for normalized IST data).
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        bce_on_observed: Add the threshold-side term on observed positions.
        eps: Minimum scale b (added after softplus) for numerical stability.
        var_reg: Weight for the scale regularization term. Default 0.0.
        var_reg_form: "raw_sigma" penalizes raw_sigma^2; "sigma" penalizes
            b^2 (minimum at b=eps). Default "raw_sigma".
        shift: Shift s so that log(y + s) is defined; must satisfy
            min(target) + shift > 0 over train AND test data.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        scheme: str = "none",
        alpha: float = 2.0,
        bce_on_observed: bool = False,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
        shift: float = 1e-3,
    ):
        super().__init__()
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.threshold = threshold
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.bce_on_observed = bce_on_observed
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
        self.shift = shift
        # censoring threshold in log-space
        self.log_threshold = math.log(threshold + shift)
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def _log_sf(u: torch.Tensor) -> torch.Tensor:
        """log P(X >= x) for Laplace(0, 1) with u = (x - mu) / b, stable."""
        # u >= 0: SF = 0.5 * exp(-u)  ->  log SF = -log(2) - u
        # u <  0: SF = 1 - 0.5 * exp(u) -> log SF = log1p(-0.5 * exp(u))
        upper = -math.log(2.0) - u.clamp_min(0.0)
        lower = torch.log1p(-0.5 * torch.exp(u.clamp_max(0.0)))
        return torch.where(u >= 0, upper, lower)

    @staticmethod
    def _log_cdf(u: torch.Tensor) -> torch.Tensor:
        """log P(X < x) for Laplace(0, 1) with u = (x - mu) / b, stable."""
        # u <  0: F = 0.5 * exp(u)   ->  log F = u - log(2)
        # u >= 0: F = 1 - 0.5 * exp(-u) -> log F = log1p(-0.5 * exp(-u))
        lower = u.clamp_max(0.0) - math.log(2.0)
        upper = torch.log1p(-0.5 * torch.exp((-u).clamp_max(0.0)))
        return torch.where(u < 0, lower, upper)

    @staticmethod
    def fail_probability(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        threshold: float = 1.0,
        shift: float = 1e-3,
    ) -> torch.Tensor:
        """Derived failure mask: P(y >= threshold) under log-Laplace(mu, b)."""
        u = (math.log(threshold + shift) - mu) / sigma
        return torch.exp(CensoredLaplaceNLL._log_sf(u))

    @staticmethod
    def point_prediction(mu: torch.Tensor, shift: float = 1e-3) -> torch.Tensor:
        """Median of the log-Laplace in y-space (MAE-optimal point forecast)."""
        return torch.exp(mu) - shift

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted location in log-space, shape (B, L).
            raw_sigma: Raw scale parameter, shape (B, L); b = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, L); NaN marks right-censored positions.
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all positions of each sample.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all positions.
        """
        b = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)
        u = (self.log_threshold - mu) / b

        # log transform with shift; clamp only as a numerical last resort
        log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))

        # observed: Laplace NLL of the exact value (with Jacobian)
        nll = math.log(2.0) + torch.log(b) + torch.abs(log_y - mu) / b + log_y

        # optional threshold-side label on observed positions
        if self.bce_on_observed:
            nll = nll - self._log_cdf(u)

        # censored: -log P(y >= threshold)
        nll = torch.where(valid, nll, -self._log_sf(u))

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        w = w * torch.ones_like(valid, dtype=nll.dtype)
        if sample_weight is not None:
            w = w * sample_weight
        denom = w.sum().clamp_min(1e-8)
        loss = (nll * w).sum() / denom

        # scale regularization: penalize large b to prevent the model from
        # inflating the scale to reduce the NLL
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else b ** 2
            loss = loss + self.var_reg * torch.mean(pen)

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over ALL positions."""
        with torch.no_grad():
            b = F.softplus(raw_sigma) + self.eps
            valid = ~torch.isnan(target)
            u = (self.log_threshold - mu) / b
            log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))
            nll = math.log(2.0) + torch.log(b) + torch.abs(log_y - mu) / b + log_y
            if self.bce_on_observed:
                nll = nll - self._log_cdf(u)
            nll = torch.where(valid, nll, -self._log_sf(u))
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            denom = w.sum(dim=1).clamp_min(1e-8)
            return (nll * w).sum(dim=1) / denom


class LogNormalNLL(nn.Module):
    """
    Log-normal negative log-likelihood for scalar (non-censored) targets.

    Model: log(y + shift) ~ N(mu, sigma) per target. Unlike the Gaussian NLL,
    the variance is asymmetric in y-space (right-skewed, growing with the
    value), which matches multiplicative noise. All targets are observed (NaN
    targets are excluded as missing). The loss is a weighted mean over all
    targets.

    The + log(y + shift) term is the Jacobian of the log transform: it makes
    the loss a proper density in y-space, so the NLL is directly comparable
    to GaussianNLL / CensoredGaussianNLL.

    Args:
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        eps: Minimum sigma (added after softplus) for numerical stability.
        var_reg: Weight for the variance regularization term.
            Default 0.0 (disabled).
        var_reg_form: "raw_sigma" penalizes raw_sigma^2 (minimum at
            raw_sigma=0, i.e. sigma ~ softplus(0)+eps); "sigma" penalizes
            sigma^2 (minimum at sigma=eps, i.e. variance is genuinely
            expensive). Default "raw_sigma" (backward compatible).
        shift: Shift s so that log(y + s) is defined; must satisfy
            min(target) + shift > 0 over train AND test data.
        nll_beta: Exponent of the beta-NLL weighting (Seitzer et al. 2022):
            each element's NLL is multiplied by sigma**nll_beta. The weight
            is DETACHED (stop-gradient), matching the paper's formulation:
            it is a per-element weight, not part of the optimized objective.
            The plain NLL mu-gradient scales as resid/sigma^2 (vanishing
            where sigma is large) while the sigma-gradient is strongest
            where sigma is small; nll_beta > 0 rebalances this coupling
            (mu-gradient ~ resid/sigma^(2-beta)) while the sigma-gradient
            keeps the plain-NLL optimum sigma^2 = resid^2. Letting gradients
            flow through the weight is unstable here: the log-space Jacobian
            log(y + shift) is a large negative constant, so sigma**beta *
            log_y rewards sigma inflation without bound (variance explosion,
            frozen constant mean). 0.0 = plain NLL (default, backward
            compatible), 0.5 = recommended general setting,
            1.0 ~ MSE-like (sigma-independent) mu-gradient.
        var_reg_weighted: If True, the variance penalty is added to the
            per-element NLL BEFORE the position/sample weighting and
            normalization, so its strength relative to the NLL is stable
            under sample weighting and hard-example mining. If False
            (default, backward compatible), the penalty is added to the
            normalized loss as var_reg * mean(pen), which makes its
            relative weight fluctuate with the batch weight sum.
        sw_mean_only: If True, sample weights (adaptive sample weighting /
            hard-example mining) are applied ONLY to the mean-fit term
            0.5 * ((log_y - mu) / sigma)^2; the variance side of the NLL
            (log(sigma) + constants) and the variance penalty are always
            unweighted means. Rationale: the class-imbalance signal should
            concentrate the mu-gradient where fitting matters, while sigma
            is minimized uniformly everywhere it can be - a well-fitted
            mean needs little variance, and only where the mean cannot fit
            does the (weighted) residual term push sigma up to |resid|
            ("pay the price with variance"). This also removes the
            pathological coupling where a large sample weight multiplies
            both the sigma-gradient and the variance penalty on the same
            hard samples. Position weights still apply to both terms.
            Default False (backward compatible).
    """

    def __init__(
        self,
        scheme: str = "none",
        alpha: float = 2.0,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
        shift: float = 1e-3,
        nll_beta: float = 0.0,
        var_reg_weighted: bool = False,
        sw_mean_only: bool = False,
    ):
        super().__init__()
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
        self.shift = shift
        self.nll_beta = nll_beta
        self.var_reg_weighted = var_reg_weighted
        self.sw_mean_only = sw_mean_only
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def point_prediction(mu: torch.Tensor, shift: float = 1e-3) -> torch.Tensor:
        """Median of the log-normal in y-space (MAE-optimal point forecast)."""
        return torch.exp(mu) - shift

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean in log-space, shape (B, D).
            raw_sigma: Raw scale parameter, shape (B, D); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, D); NaN marks missing values (excluded).
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all targets of each sample.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all targets.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)

        # log transform with shift; clamp only as a numerical last resort
        # (shift must be chosen so that target + shift > 0)
        log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))

        # log-normal NLL of the exact value (Gaussian NLL in log-space plus
        # the log(y + shift) Jacobian)
        nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
            + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y

        # beta-NLL (Seitzer et al. 2022): weight each element's NLL by
        # sigma**beta so the mu-gradient no longer vanishes at large sigma.
        # The weight is detached (as in the paper): with gradients flowing
        # through the weight, the constant log-space Jacobian log_y makes
        # d(loss)/d(sigma) negative everywhere -> unbounded sigma inflation.
        if self.nll_beta > 0:
            nll = nll * (sigma ** self.nll_beta).detach()

        # variance regularization: penalize large sigma to prevent
        # the model from inflating the variance to reduce the NLL.
        # var_reg_weighted: the penalty joins the per-element NLL before
        # weighting/normalization (stable relative strength); legacy mode
        # adds var_reg * mean(pen) to the normalized loss instead.
        pen = None
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else sigma ** 2
            if self.var_reg_weighted and not self.sw_mean_only:
                nll = nll + self.var_reg * pen

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        w_valid = w * valid.float()

        if self.sw_mean_only:
            # Sample weights act ONLY on the mean-fit (residual) term; the
            # variance side (log(sigma) + constants) and the variance
            # penalty are unweighted means over valid elements, so sigma is
            # minimized uniformly everywhere and only the weighted residual
            # term can push it up where the mean cannot fit.
            # NOTE: with nll_beta > 0 the detached sigma**beta weight was
            # applied to the whole nll above; in sw_mean_only mode beta
            # should be 0 (the mean-term reweighting is the sample
            # weighter's job), so the split below uses the unweighted terms.
            res_term = 0.5 * ((log_y - mu) / sigma) ** 2
            var_term = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + log_y

            w_sw = w_valid
            if sample_weight is not None:
                w_sw = w_sw * sample_weight
            denom_sw = w_sw.sum().clamp_min(1e-8)
            denom = w_valid.sum().clamp_min(1e-8)

            loss = (res_term * w_sw).sum() / denom_sw \
                + (var_term * w_valid).sum() / denom
            if self.var_reg > 0:
                loss = loss + self.var_reg * (pen * w_valid).sum() / denom
            return loss

        if sample_weight is not None:
            w_valid = w_valid * sample_weight
        denom = w_valid.sum().clamp_min(1e-8)
        loss = (nll * w_valid).sum() / denom

        if self.var_reg > 0 and not self.var_reg_weighted:
            loss = loss + self.var_reg * torch.mean(pen)

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over valid targets.

        NOTE: intentionally NOT beta-weighted even when nll_beta > 0, so the
        sample-weighter z-scores and hard-pool ranking keep their plain-NLL
        semantics (single-lever attribution of nll_beta)."""
        with torch.no_grad():
            sigma = F.softplus(raw_sigma) + self.eps
            valid = ~torch.isnan(target)
            log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))
            nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
                + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            w_valid = w * valid.float()
            denom = w_valid.sum(dim=1).clamp_min(1e-8)
            return (nll * w_valid).sum(dim=1) / denom


def lognormal_crps(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    shift: float = 1e-3,
) -> torch.Tensor:
    """
    Closed-form CRPS of a log-normal predictive distribution (mean over valid).

    Motivation (checkpoint selection): the NLL is a strictly proper scoring
    rule, but it is unbounded as sigma -> 0 for a fixed residual. With a
    variance penalty active (loss_var_reg > 0, var_reg_form "sigma") the
    training objective is a *penalized* likelihood while the logged val_nll is
    the *unpenalized* one, so val_nll stops being monotone in fit quality and
    degrades as sigma shrinks even while MAE/R2 keep improving. That makes it
    unusable for best-checkpoint selection.

    CRPS is also strictly proper (minimized only by the true predictive
    distribution, so it stays a genuine probabilistic score and is compatible
    with probabilistic training), but it is measured in units of y and
    degrades gracefully to the MAE as sigma -> 0. It is therefore monotone in
    fit quality and robust to the var_reg-induced sigma shrinkage.

    For Y ~ logN(mu, sigma) on the shifted variable y + shift (Gneiting &
    Raftery 2007):

        CRPS = (y+s) * (2*Phi(w) - 1)
               - 2*exp(mu + sigma^2/2) * (Phi(w - sigma) + Phi(sigma/sqrt(2)) - 1)

    with w = (log(y+s) - mu) / sigma.

    Args:
        mu: Predicted log-space mean, shape (B, D).
        sigma: Predicted log-space std (already positive), shape (B, D).
        target: Raw targets in y-space, shape (B, D); NaN marks missing.
        shift: Shift s used in log(y + s).

    Returns:
        torch.Tensor: Scalar mean CRPS over valid targets (lower is better).
    """
    valid = ~torch.isnan(target)
    y_s = (torch.nan_to_num(target) + shift).clamp_min(1e-12)
    log_y = torch.log(y_s)
    w = (log_y - mu) / sigma

    crps = y_s * (2.0 * torch.special.ndtr(w) - 1.0) - 2.0 * torch.exp(
        mu + 0.5 * sigma ** 2
    ) * (
        torch.special.ndtr(w - sigma)
        + torch.special.ndtr(sigma / math.sqrt(2.0))
        - 1.0
    )

    denom = valid.float().sum().clamp_min(1.0)
    return (crps * valid.float()).sum() / denom


def gaussian_crps(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form CRPS of a Gaussian predictive distribution (mean over valid).

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)),
    with z = (y - mu) / sigma. See lognormal_crps for the rationale.
    """
    valid = ~torch.isnan(target)
    z = (torch.nan_to_num(target) - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    crps = sigma * (
        z * (2.0 * torch.special.ndtr(z) - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )
    denom = valid.float().sum().clamp_min(1.0)
    return (crps * valid.float()).sum() / denom


class CensoredLogNormalNLL(nn.Module):
    """
    Censored log-normal negative log-likelihood (lognormal Tobit) for
    right-censored targets.

    Model: log(y + shift) ~ N(mu, sigma) per position. Same censoring
    semantics as CensoredGaussianNLL (NaN = informative right-censoring at
    the threshold), but with a right-skewed variance in y-space. Each
    position contributes its exact likelihood:

    - observed position:  log N(log(y+s); mu, sigma) - log(y+s)  (Jacobian)
    - censored position:  log P(y >= threshold) = log_ndtr((mu - log(t+s)) / sigma)

    The Jacobian makes the observed term a proper density in y-space, so the
    NLL is directly comparable to the Gaussian Tobit NLL.

    The failure mask is derived from the value distribution by construction:
    p_fail = ndtr((mu - log(threshold+s)) / sigma), so the two heads cannot
    disagree.

    Args:
        threshold: Censoring threshold (1.0 for normalized IST data).
        scheme: Position weighting scheme ("none", "linear", "exp").
        alpha: Sharpness for scheme="exp".
        bce_on_observed: Add the threshold-side term on observed positions.
        eps: Minimum sigma (added after softplus) for numerical stability.
        var_reg: Weight for the variance regularization term. Default 0.0
            (disabled). Note: keep it gentle - for censored positions some
            uncertainty is real, and squeezing sigma biases the derived
            failure probability downward.
        var_reg_form: "raw_sigma" penalizes raw_sigma^2; "sigma" penalizes
            sigma^2 (minimum at sigma=eps). Default "raw_sigma".
        shift: Shift s so that log(y + s) is defined; must satisfy
            min(target) + shift > 0 over train AND test data.
        expected_len: Expected per-variable sequence length (e.g. 200 cycles).
            When set, the termination-aware "fail mode" activates: see
            classify_termination. Default None (legacy behavior).
        fail_margin: A terminated sequence counts as an honest failure only if
            the trigger variable's last observed value >= fail_margin *
            threshold. Below that, the termination is treated as an exogenous
            interruption (instrument glitch, open circuit) and its tail is
            fully masked.
        fail_anchor: Weight of the boundary-anchor term: at the last observed
            position of the trigger variable, mu is pulled to
            log(threshold + shift) (the last IST prediction must be 1.0).
        bce_weight: Weight of the bce_on_observed "alive" term
            -log P(y < threshold) on observed positions. It penalizes false /
            early predicted failures (the Tobit survival term covers missed /
            late ones). In fail mode the anchor positions are excluded (the
            anchor pulls them to the threshold, p_fail = 0.5). Default 1.0
            (legacy behavior).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        scheme: str = "none",
        alpha: float = 2.0,
        bce_on_observed: bool = False,
        eps: float = 1e-3,
        var_reg: float = 0.0,
        var_reg_form: str = "raw_sigma",
        shift: float = 1e-3,
        var_reg_weighted: bool = False,
        expected_len: int = None,
        fail_margin: float = 0.8,
        fail_anchor: float = 0.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()
        assert var_reg_form in ("raw_sigma", "sigma"), \
            f"Unknown var_reg_form: {var_reg_form}"
        self.threshold = threshold
        self.scheme = scheme if scheme is not None else "none"
        self.alpha = alpha
        self.bce_on_observed = bce_on_observed
        # weight of the "alive" term -log P(y < threshold) on observed
        # positions (only with bce_on_observed); 1.0 = legacy behavior
        self.bce_weight = bce_weight
        self.eps = eps
        self.var_reg = var_reg
        self.var_reg_form = var_reg_form
        self.shift = shift
        self.var_reg_weighted = var_reg_weighted
        # termination-aware "fail mode" (see classify_termination)
        self.expected_len = expected_len
        self.fail_margin = fail_margin
        self.fail_anchor = fail_anchor
        self.fail_mode = expected_len is not None
        # censoring threshold in log-space
        self.log_threshold = math.log(threshold + shift)
        self._weight_cache = {}

    def _get_weights(self, seq_len: int, device, dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = build_position_weights(
                seq_len, scheme=self.scheme, alpha=self.alpha, device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def _neg_log_sf(z: torch.Tensor, z_max: float = 30.0) -> torch.Tensor:
        """-log P(Z > z) = -log_ndtr(-z) with finite gradients for large z.

        torch's log_ndtr backward overflows for very negative arguments
        (z ~ 1e4 when sigma -> eps). Above z_max the Mills-ratio asymptote
        0.5 z^2 + log(z) + 0.5 log(2 pi) (relative error < 1e-3 at z = 30) is
        used instead; both branches see clamped inputs so torch.where never
        mixes in an inf/NaN gradient.
        """
        z_lo = z.clamp(max=z_max)
        z_hi = z.clamp(min=z_max)
        exact = -torch.special.log_ndtr(-z_lo)
        asym = 0.5 * z_hi ** 2 + torch.log(z_hi) + 0.5 * math.log(2.0 * math.pi)
        return torch.where(z > z_max, asym, exact)

    def _bce_add(self, z: torch.Tensor, masks: dict = None) -> torch.Tensor:
        """Weighted "alive" term -bce_weight * log P(y < threshold), (B, L).

        Zero when bce_on_observed is off. Only observed positions keep it
        downstream (censored positions are overwritten by the survival term);
        in fail mode the anchor positions are zeroed here, because the anchor
        pulls them to the threshold (p_fail = 0.5) and the two terms would
        conflict.
        """
        if not self.bce_on_observed or self.bce_weight == 0:
            return torch.zeros_like(z)
        bce = self.bce_weight * self._neg_log_sf(z)
        if masks is not None:
            bce = torch.where(masks["anchor"], torch.zeros_like(bce), bce)
        return bce

    def bce_term(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        masks: dict = None,
    ) -> torch.Tensor:
        """Detached diagnostic: mean UNWEIGHTED -log P(y < threshold) over the
        observed positions (anchor positions excluded in fail mode). Logged as
        {stage}_bce; independent of bce_on_observed / bce_weight."""
        with torch.no_grad():
            sigma = F.softplus(raw_sigma) + self.eps
            z = (mu - self.log_threshold) / sigma
            m = ~torch.isnan(target)
            if masks is not None:
                m = masks["observed"] & ~masks["anchor"]
            bce = torch.nan_to_num(-torch.special.log_ndtr(-z), posinf=1e6)
            return bce[m].mean() if m.any() else torch.zeros((), device=mu.device)

    @staticmethod
    def fail_probability(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        threshold: float = 1.0,
        shift: float = 1e-3,
    ) -> torch.Tensor:
        """Derived failure mask: P(y >= threshold) under logN(mu, sigma)."""
        return torch.special.ndtr((mu - math.log(threshold + shift)) / sigma)

    @staticmethod
    def point_prediction(mu: torch.Tensor, shift: float = 1e-3) -> torch.Tensor:
        """Median of the log-normal in y-space (MAE-optimal point forecast)."""
        return torch.exp(mu) - shift

    @staticmethod
    def classify_termination(
        target: torch.Tensor,
        var_ids: torch.Tensor,
        pos: torch.Tensor,
        expected_len: int,
        threshold: float,
        fail_margin: float,
    ) -> dict:
        """
        Termination-aware position masks for the censored loss ("fail mode").

        Classification is PER VARIABLE: each target variable of a sample is
        judged on its own, because one variable failing does not imply the
        sibling's tail is informative (and vice versa: a failed variable must
        be supervised even when the sibling survives to full length).

        Per (sample, variable):

        - The variable is *terminated* iff its last OBSERVED position is
          before expected_len (length is the solid anchor; a full-length
          sequence "survived", regardless of its final value; interior NaN
          gaps never count as termination).
        - A terminated variable is an *honest failure* iff its last observed
          value >= fail_margin * threshold. Then ALL its censored tail
          positions after the last observation get the Tobit survival term
          (failure is absorbing: y >= threshold holds at every later cycle,
          so each is a genuine censored observation).
        - A terminated variable with last value < fail_margin * threshold is
          an *early interruption* (exogenous: instrument glitch, open
          circuit): its tail stays masked.

        Non-terminated variables keep their observed positions only; any
        censored positions (interior gaps) stay masked.

        Args:
            target: Raw targets (B, L); NaN marks censored/missing values.
            var_ids: Variable id per position (B, L); 0 or NaN = padding.
            pos: Cycle/position index per position (B, L); NaN = padding.
            expected_len: Expected per-variable sequence length.
            threshold: Censoring (failure) threshold in y-space.
            fail_margin: Honest-failure margin on the last observed value.

        Returns:
            dict with bool (B, L) masks:
              observed: exact value known - standard observed NLL term.
              survival: censored tail positions of honestly-failed variables
                  - Tobit survival term (y >= threshold).
              anchor: last observed position of each honestly-failed
                  variable - the boundary anchor (mu must equal
                  log(threshold + shift)).
              active: positions entering the NLL sum (observed | survival).
              plus float (B, n_vars) per-variable labels (column j = j-th
              sorted variable id; NaN = variable absent / never observed;
              survived = 0 in both; the labels are mutually exclusive):
              honest: 1 = this variable honestly failed.
              interrupted: 1 = this variable terminated exogenously
                  (tail masked, no failure supervision).
        """
        B, L = target.shape
        device = target.device
        var_ids = torch.nan_to_num(var_ids, nan=0.0).long()
        valid = ~torch.isnan(pos) & (var_ids > 0)
        observed = valid & ~torch.isnan(target)
        censored = valid & torch.isnan(target)

        active = observed.clone()
        survival = torch.zeros(B, L, dtype=torch.bool, device=device)
        anchor = torch.zeros(B, L, dtype=torch.bool, device=device)
        all_vars = sorted(v for v in torch.unique(var_ids[valid]).tolist() if v > 0)
        # per-variable labels (column j <-> all_vars[j]); NaN = variable absent
        honest = torch.full((B, len(all_vars)), float("nan"), device=device)
        interrupted = torch.full((B, len(all_vars)), float("nan"), device=device)

        for b in range(B):
            for j, v in enumerate(all_vars):
                m_v = valid[b] & (var_ids[b] == v)
                if not m_v.any():
                    continue
                o_v = m_v & observed[b]
                if not o_v.any():
                    continue
                honest[b, j] = 0.0
                interrupted[b, j] = 0.0
                # terminated = the last OBSERVED position precedes
                # expected_len (censored tail positions carry pos/var but no
                # value; interior NaN gaps do not count as termination)
                if pos[b][o_v].max() >= expected_len:
                    continue  # survived
                # value at the last observed position of variable v
                last_val = target[b][o_v][pos[b][o_v].argmax()].item()
                if last_val < fail_margin * threshold:
                    interrupted[b, j] = 1.0  # exogenous stop: tail masked
                    continue
                # honest failure: supervise ALL censored tail positions of
                # this variable after the last observation
                honest[b, j] = 1.0
                idx_last = torch.nonzero(o_v, as_tuple=True)[0][
                    pos[b][o_v].argmax()]
                anchor[b, idx_last] = True
                c_v = censored[b] & (var_ids[b] == v) \
                    & (pos[b] > pos[b, idx_last])
                survival[b] |= c_v
                active[b] |= c_v

        return {"observed": observed, "survival": survival, "anchor": anchor,
                "active": active, "honest": honest,
                "interrupted": interrupted}

    def forward(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor = None,
        masks: dict = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean in log-space, shape (B, L).
            raw_sigma: Raw scale parameter, shape (B, L); sigma = softplus(raw_sigma) + eps.
            target: Raw targets of shape (B, L); NaN marks right-censored positions.
            sample_weight: Optional per-sample weights of shape (B, 1),
                broadcast over all positions of each sample.
            masks: Optional classify_termination output (fail mode). When
                given, the NLL sums over active positions only (observed term
                on observed, survival term on the boundary survival position,
                censored tails masked) and the anchor term
                fail_anchor * mean((mu[anchor] - log_threshold)^2) is added.

        Returns:
            torch.Tensor: Scalar loss, weighted mean over all positions.
        """
        sigma = F.softplus(raw_sigma) + self.eps
        valid = ~torch.isnan(target)
        z = (mu - self.log_threshold) / sigma

        # log transform with shift; clamp only as a numerical last resort
        log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))

        # observed: log-normal NLL of the exact value (with Jacobian)
        nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
            + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y

        # optional weighted "alive" label on observed positions
        # (anchor positions excluded in fail mode, see _bce_add)
        nll = nll + self._bce_add(z, masks)

        # censored: -log P(y >= threshold)
        nll = torch.where(valid, nll, -torch.special.log_ndtr(z))

        # fail mode: keep only active positions (observed + boundary survival)
        if masks is not None:
            nll = torch.where(masks["observed"], nll,
                              -torch.special.log_ndtr(z))

        # variance regularization: penalize large sigma to prevent
        # the model from inflating the variance to reduce the NLL.
        # var_reg_weighted: penalty joins the per-element NLL before
        # weighting/normalization; legacy mode adds it afterwards.
        pen = None
        if self.var_reg > 0:
            pen = raw_sigma ** 2 if self.var_reg_form == "raw_sigma" else sigma ** 2
            if self.var_reg_weighted:
                nll = nll + self.var_reg * pen

        # Active positions: fail mode restricts to observed + boundary
        # survival; legacy mode keeps ALL positions (censored positions
        # contribute their Tobit survival term).
        # Select them with torch.where BEFORE weighting: mu/raw_sigma at inactive
        # (masked) positions receive no gradient and may drift to extremes; there
        # -log_ndtr(z) can be +inf, and inf * 0 (zero weight) = NaN would poison
        # the sum. where() selects values, it never multiplies by zero.
        active = masks["active"] if masks is not None else torch.ones_like(valid)
        nll = torch.where(active, nll, torch.zeros_like(nll))
        # pathological-case guard: even an inf/NaN at an ACTIVE position stays
        # finite, so a single bad batch cannot NaN the optimizer state
        nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)

        w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
        w = w * active.to(nll.dtype)
        if sample_weight is not None:
            w = w * sample_weight
        denom = w.sum().clamp_min(1e-8)
        loss = (nll * w).sum() / denom

        if self.var_reg > 0 and not self.var_reg_weighted:
            loss = loss + self.var_reg * torch.mean(pen)

        # boundary anchor: the last observed IST prediction must be the threshold
        if masks is not None and self.fail_anchor > 0 and masks["anchor"].any():
            anch = (mu[masks["anchor"]] - self.log_threshold) ** 2
            if sample_weight is not None:
                sw = sample_weight.expand_as(mu)[masks["anchor"]]
                anch = anch * sw / sw.sum().clamp_min(1e-8)
                loss = loss + self.fail_anchor * anch.sum()
            else:
                loss = loss + self.fail_anchor * anch.mean()

        return loss

    def per_sample(
        self,
        mu: torch.Tensor,
        raw_sigma: torch.Tensor,
        target: torch.Tensor,
        masks: dict = None,
    ) -> torch.Tensor:
        """Detached per-sample loss (B,): weighted mean over ALL positions
        (over the active positions only when fail-mode masks are given)."""
        with torch.no_grad():
            sigma = F.softplus(raw_sigma) + self.eps
            valid = ~torch.isnan(target)
            z = (mu - self.log_threshold) / sigma
            log_y = torch.log((torch.nan_to_num(target) + self.shift).clamp_min(1e-8))
            nll = 0.5 * math.log(2.0 * math.pi) + torch.log(sigma) \
                + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
            nll = nll + self._bce_add(z, masks)
            nll = torch.where(valid, nll, -torch.special.log_ndtr(z))
            # same NaN-safety as forward(): select active positions before
            # weighting (inf * 0 = NaN at drifting inactive positions), then
            # clamp pathological values (this loss feeds the adaptive sample
            # weighter and the hard-mining memory, so a NaN here re-enters
            # the training loss as NaN sample weights). Legacy mode keeps ALL
            # positions active (censored positions contribute the Tobit term).
            active = masks["active"] if masks is not None else torch.ones_like(valid)
            nll = torch.where(active, nll, torch.zeros_like(nll))
            nll = torch.nan_to_num(nll, nan=0.0, posinf=1e6, neginf=-1e6)
            w = self._get_weights(mu.shape[1], mu.device, mu.dtype).view(1, -1)
            w = w * active.to(nll.dtype)
            denom = w.sum(dim=1).clamp_min(1e-8)
            return (nll * w).sum(dim=1) / denom
