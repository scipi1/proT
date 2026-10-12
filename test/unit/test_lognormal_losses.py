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


class TestLogNormalBetaNLL:
    """beta-NLL (Seitzer et al. 2022): per-element NLL weighted by sigma**beta."""

    def test_beta_zero_is_plain_nll(self):
        """nll_beta=0 must be identical to the plain loss (backward compat)."""
        mu = torch.tensor([[-5.0, -4.0]])
        raw_sigma = torch.tensor([[-1.0, 1.0]])
        target = torch.tensor([[0.001, 0.01]])
        plain = LogNormalNLL()(mu, raw_sigma, target)
        beta0 = LogNormalNLL(nll_beta=0.0)(mu, raw_sigma, target)
        assert plain == pytest.approx(beta0.item(), rel=1e-7)

    def test_beta_weights_nll_by_sigma(self):
        """With nll_beta the loss equals mean(sigma**beta * NLL)."""
        beta = 0.5
        loss_fn = LogNormalNLL(nll_beta=beta)
        mu = torch.tensor([[-5.0, -4.0]])
        raw_sigma = torch.tensor([[-1.0, 1.0]])
        target = torch.tensor([[0.001, 0.01]])
        expected = (_sigma(raw_sigma) ** beta
                    * _logn_nll(mu, raw_sigma, target)).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-6)

    def test_beta_rebalances_mu_gradient(self):
        """Plain NLL mu-gradient ~ resid/sigma^2 collapses at large sigma;
        beta=2 makes it exactly sigma-independent (resid only)."""
        target = torch.full((1, 2), 0.01)
        raw_sigma = torch.tensor([[-2.0, 2.0]])  # small vs large sigma
        grads = {}
        for beta in (0.0, 2.0):
            mu = torch.zeros(1, 2, requires_grad=True)
            LogNormalNLL(nll_beta=beta)(mu, raw_sigma, target).backward()
            grads[beta] = mu.grad[0].detach()
        ratio_plain = (grads[0.0][0] / grads[0.0][1]).item()
        ratio_beta2 = (grads[2.0][0] / grads[2.0][1]).item()
        # beta=2: mu-gradient ratio across sigmas ~ 1 (vs >> 1 for plain NLL)
        assert ratio_plain > 10
        assert ratio_beta2 == pytest.approx(1.0, rel=1e-3)

    def test_beta_gradient_flows_to_sigma(self):
        """The sigma**beta weight must carry gradient to raw_sigma."""
        mu = torch.zeros(1, 2)
        raw_sigma = torch.zeros(1, 2, requires_grad=True)
        target = torch.full((1, 2), 0.01)
        LogNormalNLL(nll_beta=0.5)(mu, raw_sigma, target).backward()
        assert raw_sigma.grad is not None
        assert torch.all(torch.isfinite(raw_sigma.grad))
        assert raw_sigma.grad.abs().sum() > 0

    def test_per_sample_not_beta_weighted(self):
        """per_sample feeds the sample weighter / hard pool: plain NLL always."""
        mu = torch.tensor([[-5.0, -4.0]])
        raw_sigma = torch.tensor([[-1.0, 1.0]])
        target = torch.tensor([[0.001, 0.01]])
        ps_plain = LogNormalNLL().per_sample(mu, raw_sigma, target)
        ps_beta = LogNormalNLL(nll_beta=0.5).per_sample(mu, raw_sigma, target)
        assert torch.allclose(ps_plain, ps_beta)


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

    def test_var_reg_forms(self):
        """var_reg penalizes raw_sigma^2 or sigma^2 depending on the form."""
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.tensor([[1.5]])
        target = torch.tensor([[0.01]])

        plain = CensoredLogNormalNLL(var_reg=0.0)(mu, raw_sigma, target)
        reg_raw = CensoredLogNormalNLL(var_reg=0.1, var_reg_form="raw_sigma")(
            mu, raw_sigma, target)
        reg_sig = CensoredLogNormalNLL(var_reg=0.1, var_reg_form="sigma")(
            mu, raw_sigma, target)

        assert reg_raw.item() == pytest.approx(plain.item() + 0.1 * 1.5 ** 2, rel=1e-5)
        sigma = _sigma(raw_sigma)
        assert reg_sig.item() == pytest.approx(
            plain.item() + 0.1 * sigma.item() ** 2, rel=1e-5)


class TestLogNormalBetaNLL:
    """beta-NLL (Seitzer et al. 2022): the sigma**beta weight must be
    DETACHED. With gradients through the weight, the constant log-space
    Jacobian log_y makes d(loss)/d(sigma) negative everywhere -> unbounded
    variance inflation with a frozen constant mean (observed in the
    D224_beta run)."""

    def test_value_is_plain_nll_times_sigma_beta(self):
        beta = 0.5
        mu = torch.tensor([[-5.0, -4.0]])
        raw_sigma = torch.tensor([[0.3, -0.7]])
        target = torch.tensor([[0.001, 0.01]])
        plain = LogNormalNLL(nll_beta=0.0)(mu, raw_sigma, target)
        beta_loss = LogNormalNLL(nll_beta=beta)(mu, raw_sigma, target)
        expected = (_logn_nll(mu, raw_sigma, target) * _sigma(raw_sigma) ** beta).mean()
        assert beta_loss.item() == pytest.approx(expected.item(), rel=1e-6)
        assert beta_loss.item() != pytest.approx(plain.item(), rel=1e-3)

    def test_sigma_gradient_pushes_inflated_sigma_down(self):
        """With a fitted mu and an inflated sigma, the raw_sigma gradient
        must be POSITIVE (gradient descent shrinks sigma): the beta weight
        must not reward variance inflation."""
        beta = 0.5
        target = torch.tensor([[0.0025]])
        log_y = torch.log(target + SHIFT)
        mu = log_y.clone().requires_grad_(True)          # fitted mean
        raw_sigma = torch.tensor([[3.0]], requires_grad=True)  # sigma ~ 3.05 >> resid
        loss = LogNormalNLL(nll_beta=beta)(mu, raw_sigma, target)
        loss.backward()
        assert raw_sigma.grad.item() > 0

    def test_sigma_optimum_unchanged_at_resid(self):
        """The detached weight leaves the sigma optimum at sigma^2 = resid^2:
        d(loss)/d(raw_sigma) ~ 0 there (for the single-element case)."""
        beta = 0.5
        target = torch.tensor([[0.0025]])
        log_y = torch.log(target + SHIFT)
        mu = torch.tensor([[log_y.item() - 0.4]])  # resid 0.4 in log-space
        # raw_sigma such that sigma ~ resid = 0.4
        raw_opt = torch.log(torch.expm1(torch.tensor(0.4 - EPS)))
        raw_sigma = raw_opt.reshape(1, 1).requires_grad_(True)
        LogNormalNLL(nll_beta=beta)(mu, raw_sigma, target).backward()
        assert abs(raw_sigma.grad.item()) < 1e-3

    def test_mu_gradient_scales_as_resid_over_sigma_2_minus_beta(self):
        """The mu-gradient is rebalanced: d(loss)/d(mu) = sigma**beta *
        (mu - log_y)/sigma^2 = resid / sigma^(2 - beta)."""
        beta = 0.5
        target = torch.tensor([[0.0025]])
        log_y = torch.log(target + SHIFT)
        mu = torch.tensor([[log_y.item() + 0.3]], requires_grad=True)
        raw_sigma = torch.tensor([[1.0]])  # sigma = softplus(1) + eps
        LogNormalNLL(nll_beta=beta)(mu, raw_sigma, target).backward()
        sigma = _sigma(raw_sigma)
        expected = sigma.item() ** beta * 0.3 / sigma.item() ** 2
        assert mu.grad.item() == pytest.approx(expected, rel=1e-5)


class TestLogNormalSWMeanOnly:
    """sw_mean_only: sample weights act ONLY on the mean-fit (residual)
    term; the variance side (log(sigma) + constants) and the variance
    penalty are unweighted means."""

    def test_matches_legacy_without_sample_weights(self):
        """With no sample weights the split must reproduce the plain loss
        (for both var_reg_weighted modes, since uniform weights make the
        weighted penalty equal to the unweighted mean)."""
        mu = torch.tensor([[-5.0, -4.0], [-4.5, -3.5]])
        raw_sigma = torch.tensor([[0.3, -0.7], [0.1, 0.5]])
        target = torch.tensor([[0.001, 0.01], [0.004, 0.02]])
        for vr_weighted in (False, True):
            legacy = LogNormalNLL(var_reg=0.05, var_reg_form="sigma",
                                  var_reg_weighted=vr_weighted)(
                mu, raw_sigma, target)
            split = LogNormalNLL(var_reg=0.05, var_reg_form="sigma",
                                 var_reg_weighted=vr_weighted,
                                 sw_mean_only=True)(mu, raw_sigma, target)
            assert split.item() == pytest.approx(legacy.item(), rel=1e-6)

    def test_sigma_gradient_invariant_to_sample_weights(self):
        """Uniform scaling of the sample weights must not change the
        raw_sigma gradient: the variance side is excluded from the
        reweighting (and the weighted residual term is normalized)."""
        torch.manual_seed(0)
        target = torch.rand(8, 2) * 0.01
        loss_fn = LogNormalNLL(var_reg=0.05, var_reg_form="sigma",
                               sw_mean_only=True)
        grads = []
        for w_val in (1.0, 20.0):
            mu = torch.zeros(8, 2, requires_grad=True)
            raw_sigma = torch.zeros(8, 2, requires_grad=True)
            loss_fn(mu, raw_sigma, target,
                    sample_weight=torch.full((8, 1), w_val)).backward()
            grads.append(raw_sigma.grad.clone())
        assert torch.allclose(grads[0], grads[1], rtol=1e-5, atol=1e-8)

    def test_sample_weights_redistribute_mean_term(self):
        """Up-weighting one sample must move its share of the mu-gradient:
        the weighted residual term is normalized, so relative focus shifts
        toward the boosted sample."""
        torch.manual_seed(0)
        target = torch.rand(4, 2) * 0.01
        loss_fn = LogNormalNLL(sw_mean_only=True)
        mu_grads = []
        for w in (torch.ones(4, 1), torch.tensor([[10.0], [1.0], [1.0], [1.0]])):
            mu = torch.zeros(4, 2, requires_grad=True)
            raw_sigma = torch.zeros(4, 2)
            loss_fn(mu, raw_sigma, target, sample_weight=w).backward()
            mu_grads.append(mu.grad.clone())
        share_uniform = mu_grads[0][0].abs().sum() / mu_grads[0].abs().sum()
        share_boosted = mu_grads[1][0].abs().sum() / mu_grads[1].abs().sum()
        assert share_boosted > share_uniform

    def test_var_reg_penalty_unweighted(self):
        """The variance penalty enters as an unweighted mean: its value
        contribution is var_reg * mean(sigma^2) regardless of weights."""
        mu = torch.tensor([[-4.0, -4.0]])
        raw_sigma = torch.tensor([[1.5, 1.5]])
        target = torch.tensor([[0.01, 0.02]])
        loss_fn = LogNormalNLL(var_reg=0.1, var_reg_form="sigma",
                               sw_mean_only=True)
        plain = LogNormalNLL(var_reg=0.0, sw_mean_only=True)
        for w in (torch.ones(1, 1), torch.tensor([[7.0]])):
            diff = (loss_fn(mu, raw_sigma, target, sample_weight=w)
                    - plain(mu, raw_sigma, target, sample_weight=w)).item()
            sigma = _sigma(raw_sigma)
            assert diff == pytest.approx(0.1 * sigma.pow(2).mean().item(),
                                         rel=1e-5)
