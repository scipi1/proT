"""
Sample-level loss weighting for imbalanced regression.

Rationale:
- Position weights (build_position_weights in losses.py) reweight positions
  *within* each sample; all samples contribute equally to the batch mean, so
  rare high-target samples ("minority" samples, e.g. genuinely defective
  parts with steep IST slopes) are swamped by the majority.
- AdaptiveLossWeighter weights entire samples (one scalar per sample,
  broadcast over all of its positions/targets) as a function of the current
  per-sample loss, z-scored against EMA-tracked statistics. This is the
  regression analog of focal loss (Lin et al., 2017) and, in its exponential
  form, TERM (tilted empirical risk minimization, Li et al., 2021).
- The scheme is self-regulating: w -> 1 as the sample's loss approaches the
  mean loss, so samples are not boosted merely for sitting above a fixed
  percentile, and the weighting automatically tracks a *shifting* minority
  as training progresses.

Note on sign safety: NLL losses can be negative, so naive ratios like
l / mean(l) would invert the ordering (a worse sample at -2 vs. a mean of
-5 would get ratio < 1 and be *downweighted*). The z-score form is
well-defined for any loss sign.
"""

import torch


def summarize_target(target: torch.Tensor, summary: str = "max") -> torch.Tensor:
    """
    Per-sample scalar summary of (B, L) targets (NaN = censored/missing).

    Used for diagnostics (e.g. the tail-MAE metric), not for the adaptive
    weights themselves (which need no target summary: the per-sample loss is
    already a scalar).

    Args:
        target: Raw targets of shape (B, L), may contain NaN.
        summary: "max" (largest observed value), "last" (last observed value)
            or "mean" (mean over observed positions).

    Returns:
        torch.Tensor: Summary of shape (B,); NaN for all-NaN rows.
    """
    valid = ~torch.isnan(target)
    if summary == "max":
        s = torch.where(valid, target, torch.full_like(target, float("-inf"))).max(dim=1).values
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, float("nan")))
    elif summary == "mean":
        denom = valid.float().sum(dim=1)
        s = torch.nan_to_num(target, nan=0.0).sum(dim=1) / denom.clamp_min(1.0)
        s = torch.where(denom > 0, s, torch.full_like(s, float("nan")))
    elif summary == "last":
        L = target.shape[1]
        # index of the last valid position per row
        rev = valid.flip(dims=[1]).float().argmax(dim=1)
        last_idx = (L - 1 - rev).clamp_min(0)
        s = torch.nan_to_num(target, nan=0.0).gather(1, last_idx.unsqueeze(1)).squeeze(1)
        s = torch.where(valid.any(dim=1), s, torch.full_like(s, float("nan")))
    else:
        raise ValueError(f"Unknown summary: {summary}. Choose 'max', 'last' or 'mean'.")
    return s


class AdaptiveLossWeighter:
    """
    Loss-based adaptive sample weights (focal/TERM-style, sign-safe).

    The per-sample loss l_i is z-scored against EMA estimates of the loss
    mean and std (detached, no gradient through the weights):

        z_i = (l_i - mu) / sqrt(var + eps)

    Weights (capped, then normalized to mean 1 per batch):

        linear: w = clamp(1 + gamma * z, min=w_min, max=w_max)
        exp:    w = clamp(exp(gamma * z), min=w_min, max=w_max)

    Self-regulating: a sample at the mean loss gets w = 1, a sample 1 std
    above gets 1 + gamma (linear) or exp(gamma) (exp), and below-mean samples
    are smoothly downweighted towards w_min. As a sample's loss approaches
    the mean, its weight returns to 1.

    State (EMA statistics) is not checkpointed; on resume it re-estimates
    within a few batches.

    Note on w_min: when the loss drops fast (early training), the EMA mean
    lags behind and all z-scores of a batch can fall below -1/gamma. With
    w_min = 0 the whole batch would clamp to 0 and contribute nothing to the
    loss (a degenerate step). A small positive floor (default 0.1) turns such
    batches into uniform weighting instead.

    Args:
        gamma: Boost per standard deviation above the mean loss.
        form: "linear" or "exp".
        momentum: EMA momentum for the loss statistics.
        w_min: Weight floor (before normalization); must be > 0 to avoid
            degenerate all-zero batches under EMA lag.
        w_max: Weight cap (before normalization).
        start_epoch: First epoch at which weighting is active (uniform
            weights before); use a warmup to let the representation form.
        eps: Numerical floor for the std.
    """

    def __init__(self, gamma: float = 1.0, form: str = "linear", momentum: float = 0.9,
                 w_min: float = 0.1, w_max: float = 5.0, start_epoch: int = 0,
                 eps: float = 1e-8):
        assert w_min > 0, "w_min must be > 0 (degenerate all-zero batches otherwise)"
        assert form in ("linear", "exp"), f"Unknown form: {form}"
        self.gamma = gamma
        self.form = form
        self.momentum = momentum
        self.w_min = w_min
        self.w_max = w_max
        self.start_epoch = start_epoch
        self.eps = eps
        self.mu = None
        self.var = None

    def update(self, per_sample_loss: torch.Tensor) -> None:
        """Update the EMA loss statistics from a detached (B,) loss tensor."""
        l = per_sample_loss.detach()
        m, v = l.mean(), l.var(unbiased=False)
        if self.mu is None:
            self.mu, self.var = m, v
        else:
            self.mu = self.momentum * self.mu + (1 - self.momentum) * m
            self.var = self.momentum * self.var + (1 - self.momentum) * v

    def __call__(self, per_sample_loss: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Args:
            per_sample_loss: Per-sample losses of shape (B,) (detached inside).
            epoch: Current epoch (for the warmup period).

        Returns:
            torch.Tensor: Weights of shape (B, 1), normalized to mean 1.
        """
        l = per_sample_loss.detach()
        if epoch < self.start_epoch or self.mu is None:
            w = torch.ones_like(l)
        else:
            z = (l - self.mu) / torch.sqrt(self.var + self.eps)
            if self.form == "linear":
                w = 1.0 + self.gamma * z
            else:
                w = torch.exp(self.gamma * z)
            w = w.clamp(min=self.w_min, max=self.w_max)
        w = w / w.mean().clamp_min(self.eps)
        return w.unsqueeze(-1)  # (B, 1) for broadcasting over positions
