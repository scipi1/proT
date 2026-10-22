"""
Unit tests for AdaptiveLossWeighter and the sample_weight argument of the
losses (sample-level weighting for imbalanced regression).

Covers: warmup/uniform behavior, the self-regulating fixed point (w = 1 at
the mean loss), sign safety for negative NLL losses, monotonicity, capping,
mean-1 normalization, EMA updates, and backwards compatibility of the losses
(sample_weight=None must be identical to the original behavior).
"""

import pytest
import torch

from proT.training.sample_weights import AdaptiveLossWeighter, summarize_target
from proT.training.losses import (
    PositionWeightedMSE,
    GaussianNLL,
    LogNormalNLL,
    CensoredGaussianNLL,
    CensoredLogNormalNLL,
)


class TestAdaptiveLossWeighter:
    def test_uniform_before_stats(self):
        """Without EMA statistics, weights must be uniform."""
        w = AdaptiveLossWeighter()
        l = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(w(l, epoch=0), torch.ones(3, 1))

    def test_uniform_during_warmup(self):
        """Before start_epoch, weights must be uniform even with stats."""
        w = AdaptiveLossWeighter(start_epoch=5)
        l = torch.tensor([1.0, 2.0, 3.0])
        w.update(l)
        assert torch.allclose(w(l, epoch=3), torch.ones(3, 1))

    def test_fixed_point_at_mean(self):
        """A sample at the mean loss must get weight ~1 (self-regulation).

        Exact fixed point before the mean-1 normalization; the normalization
        shifts it slightly (the clamped below-mean weights pull the mean
        below 1), so compare against the exact expected values.
        """
        w = AdaptiveLossWeighter(gamma=1.0, form="linear")  # default w_min=0.1
        l = torch.tensor([1.0, 2.0, 3.0])
        w.update(l)  # mu = 2, var = 2/3
        weights = w(l, epoch=0).squeeze(-1)
        z = (l - 2.0) / (2.0 / 3.0) ** 0.5
        expected = (1.0 + z).clamp(min=0.1)
        expected = expected / expected.mean()
        assert torch.allclose(weights, expected, atol=1e-5)
        # sample at the mean (l = 2) -> z = 0 -> w ~ 1
        assert weights[1] == pytest.approx(1.0, abs=0.1)

    def test_monotonic_in_loss(self):
        w = AdaptiveLossWeighter(gamma=1.0, form="linear")
        l = torch.tensor([1.0, 2.0, 3.0])
        w.update(l)
        weights = w(l, epoch=0).squeeze(-1)
        assert weights[2] > weights[1] > weights[0]

    def test_negative_losses_sign_safe(self):
        """For negative NLL losses, a *higher* (less negative) loss = harder
        sample and must get a *higher* weight."""
        w = AdaptiveLossWeighter(gamma=1.0, form="linear")
        l = torch.tensor([-6.0, -5.0, -4.0])  # -4 is the worst sample
        w.update(l)
        weights = w(l, epoch=0).squeeze(-1)
        assert weights[2] > weights[1] > weights[0]
        assert weights[1] == pytest.approx(1.0, abs=0.1)  # mean -> ~1

    def test_exp_form(self):
        w = AdaptiveLossWeighter(gamma=1.0, form="exp")
        l = torch.tensor([1.0, 2.0, 3.0])
        w.update(l)
        weights = w(l, epoch=0).squeeze(-1)
        assert weights[2] > weights[1] > weights[0]
        assert torch.all(weights > 0)

    def test_cap_at_w_max(self):
        w = AdaptiveLossWeighter(gamma=1.0, form="linear", w_max=2.0)
        l = torch.tensor([1.0, 1.0, 1.0, 100.0])  # one extreme outlier
        w.update(l)
        weights = w(l, epoch=0).squeeze(-1)
        # capped at w_max before normalization; after mean-1 normalization
        # the outlier weight must stay below w_max / mean(w) <= w_max / (1/4)
        assert weights.max() <= 2.0 / 0.25 + 1e-5

    def test_mean_one_normalization(self):
        w = AdaptiveLossWeighter(gamma=0.5, form="linear")
        l = torch.randn(64)
        w.update(l)
        weights = w(l, epoch=0)
        assert weights.mean() == pytest.approx(1.0, rel=1e-5)

    def test_ema_update_changes_stats(self):
        w = AdaptiveLossWeighter(momentum=0.5)
        w.update(torch.tensor([0.0, 2.0]))
        mu1 = w.mu.item()
        w.update(torch.tensor([10.0, 12.0]))
        assert w.mu.item() > mu1

    def test_equal_losses_give_uniform_weights(self):
        """Zero variance -> z = 0 -> uniform weights (no division blow-up)."""
        w = AdaptiveLossWeighter()
        l = torch.full((8,), 3.0)
        w.update(l)
        weights = w(l, epoch=0)
        assert torch.allclose(weights, torch.ones(8, 1))

    def test_no_gradient_through_weights(self):
        l = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        w = AdaptiveLossWeighter()
        w.update(l)
        weights = w(l, epoch=0)
        assert not weights.requires_grad


class TestSummarizeTarget:
    def test_max_summary(self):
        t = torch.tensor([[0.1, 0.5, 0.3], [0.2, float("nan"), float("nan")]])
        s = summarize_target(t, "max")
        assert s[0] == pytest.approx(0.5)
        assert s[1] == pytest.approx(0.2)

    def test_last_summary(self):
        t = torch.tensor([[0.1, 0.5, 0.3], [0.2, 0.4, float("nan")]])
        s = summarize_target(t, "last")
        assert s[0] == pytest.approx(0.3)
        assert s[1] == pytest.approx(0.4)

    def test_all_nan_row_is_nan(self):
        t = torch.tensor([[0.1, 0.5], [float("nan"), float("nan")]])
        for summary in ("max", "last", "mean"):
            s = summarize_target(t, summary)
            assert torch.isfinite(s[0])
            assert torch.isnan(s[1])


class TestLossSampleWeightArg:
    """sample_weight=None must be backwards compatible; non-uniform weights
    must reweight entire samples (not positions)."""

    def _all_losses(self):
        return [
            GaussianNLL(),
            LogNormalNLL(),
            CensoredGaussianNLL(threshold=1.0),
            CensoredLogNormalNLL(threshold=1.0),
        ]

    def test_none_equals_uniform(self):
        mu = torch.tensor([[0.2, 0.5], [0.7, 0.1]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.3, 0.4], [0.6, float("nan")]])
        for loss_fn in self._all_losses():
            plain = loss_fn(mu, raw_sigma, target)
            uniform = loss_fn(mu, raw_sigma, target,
                              sample_weight=torch.ones(2, 1))
            assert plain == pytest.approx(uniform.item(), rel=1e-6)

    def test_single_sample_weight_recovers_its_loss(self):
        """Zeroing all samples but one must recover that sample's per-sample loss."""
        mu = torch.tensor([[0.2, 0.5], [0.7, 0.1]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.3, 0.4], [0.6, 0.2]])
        sw = torch.tensor([[0.0], [1.0]])
        for loss_fn in self._all_losses():
            weighted = loss_fn(mu, raw_sigma, target, sample_weight=sw)
            per_sample = loss_fn.per_sample(mu, raw_sigma, target)
            assert weighted == pytest.approx(per_sample[1].item(), rel=1e-5)

    def test_per_sample_is_detached(self):
        mu = torch.tensor([[0.2, 0.5]], requires_grad=True)
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.3, 0.4]])
        for loss_fn in self._all_losses():
            assert not loss_fn.per_sample(mu, raw_sigma, target).requires_grad

    def test_mse_sample_weight(self):
        loss_fn = PositionWeightedMSE()
        pred = torch.tensor([[0.1, 0.2], [0.5, 0.5]])
        target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        plain = loss_fn(pred, target)
        uniform = loss_fn(pred, target, sample_weight=torch.ones(2, 1))
        assert plain == pytest.approx(uniform.item(), rel=1e-6)
        # upweighting the high-error sample must increase the loss
        boosted = loss_fn(pred, target, sample_weight=torch.tensor([[0.5], [2.0]]))
        assert boosted > plain

    def test_per_sample_matches_forward_with_uniform_weights(self):
        """mean(per_sample) must equal the unweighted forward loss."""
        mu = torch.tensor([[0.2, 0.5], [0.7, 0.1], [0.3, 0.9]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.3, 0.4], [0.6, float("nan")], [0.2, 0.8]])
        for loss_fn in [GaussianNLL(), LogNormalNLL()]:
            per_sample = loss_fn.per_sample(mu, raw_sigma, target)
            # forward normalizes by the total valid count; with uniform
            # position weights mean(per_sample) equals forward only if all
            # rows have the same valid count -> row 1 has one NaN, so compare
            # against the manual weighted mean instead
            valid = ~torch.isnan(target)
            manual = (per_sample * valid.float().sum(dim=1)).sum() / valid.float().sum()
            assert loss_fn(mu, raw_sigma, target) == pytest.approx(manual.item(), rel=1e-5)
