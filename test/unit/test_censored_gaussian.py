"""
Unit tests for CensoredGaussianNLL (Tobit loss for right-censored targets).

Covers: correctness of observed/censored terms, the derived failure mask,
the bce_on_observed variant, gradient flow to both head channels, numerical
stability at extreme z, and position weighting.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from proT.training.losses import CensoredGaussianNLL

EPS = 1e-3  # must match the loss default


def _sigma(raw):
    return F.softplus(raw) + EPS


class TestObservedTerm:
    def test_matches_gaussian_nll(self):
        """Observed-only input must reduce to the standard Gaussian NLL."""
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        mu = torch.tensor([[0.2, 0.5, 0.8]])
        raw_sigma = torch.zeros_like(mu)  # sigma = softplus(0) + eps
        target = torch.tensor([[0.3, 0.4, 0.7]])

        sigma = _sigma(raw_sigma)
        expected = (0.5 * math.log(2 * math.pi) + torch.log(sigma)
                    + 0.5 * ((target - mu) / sigma) ** 2).mean()

        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_zero_at_perfect_fit_tight_sigma(self):
        """With mu == target and small sigma, the NLL approaches its minimum."""
        loss_fn = CensoredGaussianNLL(threshold=1.0, eps=1e-6)
        mu = torch.tensor([[0.3, 0.4]])
        raw_sigma = torch.full_like(mu, -10.0)  # sigma ~ 4.5e-5
        target = mu.clone()
        loss = loss_fn(mu, raw_sigma, target)
        # minimum of Gaussian NLL at y=mu is 0.5*log(2*pi*sigma^2), negative for small sigma
        assert loss.item() < 0.0


class TestCensoredTerm:
    def test_matches_log_ndtr(self):
        """Censored-only input must reduce to -log_ndtr((mu - threshold)/sigma)."""
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        mu = torch.tensor([[0.5, 1.5]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.full_like(mu, float("nan"))

        sigma = _sigma(raw_sigma)
        z = (mu - 1.0) / sigma
        expected = -torch.special.log_ndtr(z).mean()

        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_censored_loss_decreases_as_mu_rises(self):
        """The censored term must push mu above the threshold."""
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        raw_sigma = torch.zeros(1, 1)
        target = torch.full((1, 1), float("nan"))

        loss_low = loss_fn(torch.tensor([[0.0]]), raw_sigma, target)
        loss_high = loss_fn(torch.tensor([[2.0]]), raw_sigma, target)
        assert loss_high < loss_low

    def test_numerical_stability_extreme_z(self):
        """log_ndtr must keep the loss finite even for very negative z."""
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        mu = torch.tensor([[-1e4]])          # z ~ -1e4 / sigma -> deeply negative
        raw_sigma = torch.zeros_like(mu)
        target = torch.full_like(mu, float("nan"))

        loss = loss_fn(mu, raw_sigma, target)
        assert torch.isfinite(loss)


class TestDerivedMask:
    def test_fail_probability_monotonic(self):
        mu = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        sigma = torch.full_like(mu, 0.2)
        p = CensoredGaussianNLL.fail_probability(mu, sigma, threshold=1.0)
        assert torch.all(p[1:] > p[:-1]), "p_fail must increase with mu"
        assert p[2] == pytest.approx(0.5)     # mu == threshold -> 0.5
        assert p[0].item() < 1e-6             # far below -> ~0
        assert p[-1].item() > 1 - 1e-6        # far above -> ~1


class TestBceOnObserved:
    def test_adds_threshold_side_term(self):
        """bce_on_observed=True must add -log_ndtr(-z) on observed positions."""
        mu = torch.tensor([[0.4, 0.6]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.5, 0.55]])  # all observed

        loss_plain = CensoredGaussianNLL(threshold=1.0)(mu, raw_sigma, target)
        loss_bce = CensoredGaussianNLL(threshold=1.0, bce_on_observed=True)(mu, raw_sigma, target)

        sigma = _sigma(raw_sigma)
        z = (mu - 1.0) / sigma
        extra = -torch.special.log_ndtr(-z).mean()
        assert loss_bce - loss_plain == pytest.approx(extra.item(), rel=1e-5)


class TestGradientsAndWeighting:
    def test_gradient_flows_to_both_channels(self):
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        mu = torch.tensor([[0.3, 0.9]], requires_grad=True)
        raw_sigma = torch.zeros(1, 2, requires_grad=True)
        target = torch.tensor([[0.4, float("nan")]])  # one observed, one censored

        loss_fn(mu, raw_sigma, target).backward()
        assert mu.grad is not None and torch.all(torch.isfinite(mu.grad))
        assert raw_sigma.grad is not None and torch.all(torch.isfinite(raw_sigma.grad))
        assert mu.grad.abs().sum() > 0
        assert raw_sigma.grad.abs().sum() > 0

    def test_position_weighting_changes_loss(self):
        mu = torch.tensor([[0.0, 0.2]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.5, 0.1]])

        plain = CensoredGaussianNLL(threshold=1.0, scheme="none")(mu, raw_sigma, target)
        weighted = CensoredGaussianNLL(threshold=1.0, scheme="linear")(mu, raw_sigma, target)
        # squared errors: 0.25 at position 0, 0.01 at position 1; linear weights
        # up-weight position 1 (the smaller error), so the weighted loss is smaller
        assert weighted < plain

    def test_all_censored_batch_is_finite(self):
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        mu = torch.rand(2, 5)
        raw_sigma = torch.zeros(2, 5)
        target = torch.full((2, 5), float("nan"))
        assert torch.isfinite(loss_fn(mu, raw_sigma, target))

    def test_var_reg_forms(self):
        """var_reg penalizes raw_sigma^2 or sigma^2 depending on the form."""
        mu = torch.tensor([[0.0]])
        raw_sigma = torch.tensor([[1.5]])
        target = torch.tensor([[0.5]])

        plain = CensoredGaussianNLL(var_reg=0.0)(mu, raw_sigma, target)
        reg_raw = CensoredGaussianNLL(var_reg=0.1, var_reg_form="raw_sigma")(
            mu, raw_sigma, target)
        reg_sig = CensoredGaussianNLL(var_reg=0.1, var_reg_form="sigma")(
            mu, raw_sigma, target)

        assert reg_raw.item() == pytest.approx(plain.item() + 0.1 * 1.5 ** 2, rel=1e-5)
        sigma = F.softplus(raw_sigma) + 1e-3
        assert reg_sig.item() == pytest.approx(
            plain.item() + 0.1 * sigma.item() ** 2, rel=1e-5)
