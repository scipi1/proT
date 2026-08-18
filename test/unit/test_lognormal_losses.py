"""
Unit tests for LogNormalNLL and CensoredLogNormalNLL (lognormal Tobit).

Covers: correctness of the observed term (Gaussian NLL in log-space plus the
log(y + shift) Jacobian), the censored term, the derived failure mask, the
bce_on_observed variant, gradient flow to both head channels, numerical
stability at extreme z, position weighting, and the shift handling at y=0.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from proT.training.losses import LogNormalNLL, CensoredLogNormalNLL

EPS = 1e-3    # must match the loss default
SHIFT = 1e-3  # must match the loss default


def _sigma(raw):
    return F.softplus(raw) + EPS


def _logn_nll(mu, raw_sigma, target, shift=SHIFT):
    """Reference log-normal NLL in y-space (with Jacobian)."""
    sigma = _sigma(raw_sigma)
    log_y = torch.log(target + shift)
    return (0.5 * math.log(2 * math.pi) + torch.log(sigma)
            + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y)


class TestLogNormalObservedTerm:
    def test_matches_lognormal_nll(self):
        """Observed input must match the log-normal NLL with Jacobian."""
        loss_fn = LogNormalNLL()
        mu = torch.tensor([[-5.0, -4.0, -3.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.001, 0.01, 0.1]])

        expected = _logn_nll(mu, raw_sigma, target).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_jacobian_makes_nll_comparable_to_gaussian(self):
        """Without the Jacobian the NLL would differ by -log(y + shift)."""
        loss_fn = LogNormalNLL()
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.01]])

        sigma = _sigma(raw_sigma)
        log_y = torch.log(target + SHIFT)
        gaussian_part = (0.5 * math.log(2 * math.pi) + torch.log(sigma)
                         + 0.5 * ((log_y - mu) / sigma) ** 2).mean()
        # full NLL = gaussian part + Jacobian (log_y); they must differ
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(
            (gaussian_part + log_y.mean()).item(), rel=1e-5)

    def test_nan_targets_excluded(self):
        """NaN targets must not contribute to the loss."""
        loss_fn = LogNormalNLL()
        mu = torch.tensor([[-4.0, -4.0]])
        raw_sigma = torch.zeros_like(mu)
        target_full = torch.tensor([[0.01, 0.02]])
        target_nan = torch.tensor([[0.01, float("nan")]])

        loss_full = loss_fn(mu, raw_sigma, target_full)
        loss_nan = loss_fn(mu, raw_sigma, target_nan)
        # NaN position excluded: loss equals the NLL of the first position only
        expected = _logn_nll(mu[:, :1], raw_sigma[:, :1], target_full[:, :1]).mean()
        assert loss_nan == pytest.approx(expected.item(), rel=1e-5)
        assert loss_full != pytest.approx(loss_nan.item(), rel=1e-6)

    def test_var_reg_adds_penalty(self):
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.tensor([[1.5]])
        target = torch.tensor([[0.01]])

        plain = LogNormalNLL(var_reg=0.0)(mu, raw_sigma, target)
        reg = LogNormalNLL(var_reg=0.1)(mu, raw_sigma, target)
        assert reg - plain == pytest.approx(0.1 * (1.5 ** 2), rel=1e-5)

    def test_zero_target_with_shift_is_finite(self):
        """y = 0 is a valid target as long as shift > 0."""
        loss_fn = LogNormalNLL(shift=SHIFT)
        mu = torch.tensor([[-7.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.zeros(1, 1)
        assert torch.isfinite(loss_fn(mu, raw_sigma, target))

    def test_point_prediction_is_median(self):
        mu = torch.tensor([-4.0, -5.0])
        point = LogNormalNLL.point_prediction(mu, shift=SHIFT)
        assert torch.allclose(point, torch.exp(mu) - SHIFT)


class TestCensoredLogNormalObservedTerm:
    def test_matches_lognormal_nll(self):
        """Observed-only input must reduce to the log-normal NLL."""
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        mu = torch.tensor([[-5.0, -4.0, -3.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.001, 0.01, 0.1]])

        expected = _logn_nll(mu, raw_sigma, target).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)


class TestCensoredLogNormalCensoredTerm:
    def test_matches_log_ndtr(self):
        """Censored-only input must reduce to -log_ndtr((mu - log(t+s))/sigma)."""
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        mu = torch.tensor([[-1.0, 1.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.full_like(mu, float("nan"))

        sigma = _sigma(raw_sigma)
        z = (mu - math.log(1.0 + SHIFT)) / sigma
        expected = -torch.special.log_ndtr(z).mean()

        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_censored_loss_decreases_as_mu_rises(self):
        """The censored term must push mu above log(threshold + shift)."""
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        raw_sigma = torch.zeros(1, 1)
        target = torch.full((1, 1), float("nan"))

        loss_low = loss_fn(torch.tensor([[-2.0]]), raw_sigma, target)
        loss_high = loss_fn(torch.tensor([[2.0]]), raw_sigma, target)
        assert loss_high < loss_low

    def test_numerical_stability_extreme_z(self):
        """log_ndtr must keep the loss finite even for very negative z."""
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        mu = torch.tensor([[-1e4]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.full_like(mu, float("nan"))

        loss = loss_fn(mu, raw_sigma, target)
        assert torch.isfinite(loss)


class TestCensoredLogNormalDerivedMask:
    def test_fail_probability_monotonic(self):
        mu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        sigma = torch.full_like(mu, 0.2)
        p = CensoredLogNormalNLL.fail_probability(mu, sigma, threshold=1.0, shift=SHIFT)
        assert torch.all(p[1:] > p[:-1]), "p_fail must increase with mu"
        # mu == log(threshold + shift) -> 0.5
        p_mid = CensoredLogNormalNLL.fail_probability(
            torch.tensor([math.log(1.0 + SHIFT)]), torch.tensor([0.2]),
            threshold=1.0, shift=SHIFT)
        assert p_mid.item() == pytest.approx(0.5)
        assert p[0].item() < 1e-6          # far below -> ~0
        assert p[-1].item() > 1 - 1e-6     # far above -> ~1

    def test_lognormal_is_right_skewed(self):
        """The right tail must be heavier than the left tail around the median:
        P(y >= median + d) > P(y <= median - d)."""
        mu = torch.tensor([0.0])
        sigma = torch.tensor([0.5])
        median = math.exp(mu.item()) - SHIFT  # median of y in y-space
        d = 0.5 * median
        # right tail: P(y >= median + d) via the derived failure probability
        p_right = CensoredLogNormalNLL.fail_probability(
            mu, sigma, threshold=median + d, shift=SHIFT)
        # left tail: P(y < median - d) = ndtr(-(mu - log(t + s))/sigma)
        z_left = (mu - math.log(median - d + SHIFT)) / sigma
        p_left = torch.special.ndtr(-z_left)
        assert p_right > p_left


class TestCensoredLogNormalBceOnObserved:
    def test_adds_threshold_side_term(self):
        """bce_on_observed=True must add -log_ndtr(-z) on observed positions."""
        # mu near log(threshold + shift) so the threshold-side term is ~O(1)
        mu = torch.tensor([[-0.5, 0.3]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.5, 0.6]])  # all observed

        loss_plain = CensoredLogNormalNLL(threshold=1.0)(mu, raw_sigma, target)
        loss_bce = CensoredLogNormalNLL(threshold=1.0, bce_on_observed=True)(mu, raw_sigma, target)

        sigma = _sigma(raw_sigma)
        z = (mu - math.log(1.0 + SHIFT)) / sigma
        extra = -torch.special.log_ndtr(-z).mean()
        assert loss_bce - loss_plain == pytest.approx(extra.item(), rel=1e-5)


class TestCensoredLogNormalGradientsAndWeighting:
    def test_gradient_flows_to_both_channels(self):
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        mu = torch.tensor([[-4.0, 0.5]], requires_grad=True)
        raw_sigma = torch.zeros(1, 2, requires_grad=True)
        target = torch.tensor([[0.01, float("nan")]])  # one observed, one censored

        loss_fn(mu, raw_sigma, target).backward()
        assert mu.grad is not None and torch.all(torch.isfinite(mu.grad))
        assert raw_sigma.grad is not None and torch.all(torch.isfinite(raw_sigma.grad))
        assert mu.grad.abs().sum() > 0
        assert raw_sigma.grad.abs().sum() > 0

    def test_position_weighting_changes_loss(self):
        mu = torch.tensor([[-5.0, -4.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.01, 0.05]])

        plain = CensoredLogNormalNLL(threshold=1.0, scheme="none")(mu, raw_sigma, target)
        weighted = CensoredLogNormalNLL(threshold=1.0, scheme="linear")(mu, raw_sigma, target)
        assert weighted != pytest.approx(plain.item(), rel=1e-6)

    def test_all_censored_batch_is_finite(self):
        loss_fn = CensoredLogNormalNLL(threshold=1.0)
        mu = torch.rand(2, 5) - 3.0
        raw_sigma = torch.zeros(2, 5)
        target = torch.full((2, 5), float("nan"))
        assert torch.isfinite(loss_fn(mu, raw_sigma, target))

    def test_zero_target_with_shift_is_finite(self):
        """y = 0 observed positions are finite as long as shift > 0."""
        loss_fn = CensoredLogNormalNLL(threshold=1.0, shift=SHIFT)
        mu = torch.tensor([[-7.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.zeros(1, 1)
        assert torch.isfinite(loss_fn(mu, raw_sigma, target))
