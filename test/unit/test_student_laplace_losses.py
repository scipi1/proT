"""
Unit tests for StudentTNLL (log-space Student-t) and CensoredLaplaceNLL
(log-space Laplace Tobit).

Covers: correctness of the observed terms against reference formulas, the
censored term and derived failure mask, the bce_on_observed variant, NaN
handling, sample-weight broadcasting, var_reg forms, gradient flow to both
head channels, numerical stability at extreme u, and per_sample consistency.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from proT.training.losses import StudentTNLL, CensoredLaplaceNLL, LogNormalNLL

EPS = 1e-3    # must match the loss default
SHIFT = 1e-3  # must match the loss default
DF = 4.0      # must match the loss default


def _sigma(raw):
    return F.softplus(raw) + EPS


def _student_nll(mu, raw_sigma, target, shift=SHIFT, df=DF):
    """Reference log-space Student-t NLL in y-space (with Jacobian)."""
    sigma = _sigma(raw_sigma)
    log_y = torch.log(target + shift)
    z = (log_y - mu) / sigma
    log_c = (torch.lgamma(torch.tensor((df + 1.0) / 2.0, dtype=torch.float64))
             - torch.lgamma(torch.tensor(df / 2.0, dtype=torch.float64))
             - 0.5 * math.log(df * math.pi))
    return -log_c + torch.log(sigma) + 0.5 * (df + 1.0) * torch.log1p(z ** 2 / df) + log_y


def _laplace_nll(mu, raw_sigma, target, shift=SHIFT):
    """Reference log-space Laplace NLL in y-space (with Jacobian)."""
    b = _sigma(raw_sigma)
    log_y = torch.log(target + shift)
    return math.log(2.0) + torch.log(b) + torch.abs(log_y - mu) / b + log_y


def _laplace_log_sf(u):
    """Reference log survival function of Laplace(0, 1)."""
    return torch.where(u >= 0, -math.log(2.0) - u,
                       torch.log1p(-0.5 * torch.exp(u)))


class TestStudentTObservedTerm:
    def test_matches_reference(self):
        loss_fn = StudentTNLL()
        mu = torch.tensor([[-5.0, -4.0, -3.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.001, 0.01, 0.1]])

        expected = _student_nll(mu, raw_sigma, target).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_large_df_approaches_lognormal(self):
        """df -> inf must recover the log-normal NLL."""
        mu = torch.tensor([[-5.0, -4.0, -3.0]])
        raw_sigma = torch.tensor([[0.2, -0.5, 1.0]])
        target = torch.tensor([[0.001, 0.01, 0.1]])

        student = StudentTNLL(df=1e6)(mu, raw_sigma, target)
        lognormal = LogNormalNLL()(mu, raw_sigma, target)
        assert student.item() == pytest.approx(lognormal.item(), rel=1e-3)

    def test_nan_targets_excluded(self):
        loss_fn = StudentTNLL()
        mu = torch.tensor([[-4.0, -4.0]])
        raw_sigma = torch.zeros_like(mu)
        target_nan = torch.tensor([[0.01, float("nan")]])

        expected = _student_nll(mu[:, :1], raw_sigma[:, :1],
                                torch.tensor([[0.01]])).mean()
        assert loss_fn(mu, raw_sigma, target_nan) == pytest.approx(expected.item(), rel=1e-5)

    def test_sample_weight_broadcast(self):
        loss_fn = StudentTNLL()
        mu = torch.tensor([[-4.0], [-3.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.01], [0.1]])

        plain = loss_fn(mu, raw_sigma, target)
        weighted = loss_fn(mu, raw_sigma, target,
                           sample_weight=torch.tensor([[2.0], [2.0]]))
        # uniform scaling of all samples must leave the weighted mean unchanged
        assert weighted.item() == pytest.approx(plain.item(), rel=1e-5)

    def test_var_reg_forms(self):
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.tensor([[1.5]])
        target = torch.tensor([[0.01]])

        plain = StudentTNLL(var_reg=0.0)(mu, raw_sigma, target)
        reg_raw = StudentTNLL(var_reg=0.1, var_reg_form="raw_sigma")(mu, raw_sigma, target)
        reg_sig = StudentTNLL(var_reg=0.1, var_reg_form="sigma")(mu, raw_sigma, target)

        assert reg_raw.item() == pytest.approx(plain.item() + 0.1 * 1.5 ** 2, rel=1e-5)
        sigma = _sigma(raw_sigma)
        assert reg_sig.item() == pytest.approx(
            plain.item() + 0.1 * sigma.item() ** 2, rel=1e-5)

    def test_gradient_flow(self):
        mu = torch.tensor([[-4.0, -3.0]], requires_grad=True)
        raw_sigma = torch.zeros(1, 2, requires_grad=True)
        target = torch.tensor([[0.01, 0.1]])

        StudentTNLL()(mu, raw_sigma, target).backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert raw_sigma.grad is not None and torch.isfinite(raw_sigma.grad).all()

    def test_per_sample_matches_forward(self):
        loss_fn = StudentTNLL()
        mu = torch.tensor([[-4.0, -3.0], [-5.0, -3.5]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.01, 0.1], [0.002, 0.05]])

        per = loss_fn.per_sample(mu, raw_sigma, target)
        assert per.shape == (2,)
        assert per.mean().item() == pytest.approx(
            loss_fn(mu, raw_sigma, target).item(), rel=1e-5)


class TestCensoredLaplaceObservedTerm:
    def test_matches_reference(self):
        loss_fn = CensoredLaplaceNLL()
        mu = torch.tensor([[-5.0, -4.0, -3.0]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.001, 0.01, 0.1]])

        expected = _laplace_nll(mu, raw_sigma, target).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_point_prediction_is_median(self):
        mu = torch.tensor([[-4.0]])
        assert CensoredLaplaceNLL.point_prediction(mu, SHIFT) == pytest.approx(
            math.exp(-4.0) - SHIFT, rel=1e-6)


class TestCensoredLaplaceCensoredTerm:
    def test_censored_position_matches_log_sf(self):
        """NaN target = right-censored: loss must be -log SF(log(threshold+shift))."""
        loss_fn = CensoredLaplaceNLL(threshold=1.0)
        mu = torch.tensor([[-1.0]])
        raw_sigma = torch.tensor([[0.3]])
        target = torch.tensor([[float("nan")]])

        b = _sigma(raw_sigma)
        u = (math.log(1.0 + SHIFT) - mu) / b  # u < 0 branch
        expected = -_laplace_log_sf(u).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_censored_position_upper_branch(self):
        """Same check with mu below the threshold (u > 0 branch)."""
        loss_fn = CensoredLaplaceNLL(threshold=0.1)
        mu = torch.tensor([[-5.0]])
        raw_sigma = torch.tensor([[0.1]])
        target = torch.tensor([[float("nan")]])

        b = _sigma(raw_sigma)
        u = (math.log(0.1 + SHIFT) - mu) / b  # u > 0 branch
        expected = -_laplace_log_sf(u).mean()
        assert loss_fn(mu, raw_sigma, target) == pytest.approx(expected.item(), rel=1e-5)

    def test_fail_probability_bounds_and_monotonicity(self):
        mu = torch.tensor([-6.0, -2.0, 0.0])
        sigma = torch.full_like(mu, 0.5)
        p = CensoredLaplaceNLL.fail_probability(mu, sigma, threshold=1.0)
        assert ((p >= 0) & (p <= 1)).all()
        assert p[0] < p[1] < p[2]  # increasing in mu

    def test_fail_probability_matches_sf(self):
        mu = torch.tensor([-4.0, -1.0])
        sigma = torch.tensor([0.3, 0.7])
        p = CensoredLaplaceNLL.fail_probability(mu, sigma, threshold=1.0)
        u = (math.log(1.0 + SHIFT) - mu) / sigma
        assert torch.allclose(p, torch.exp(_laplace_log_sf(u)), rtol=1e-5)

    def test_bce_on_observed(self):
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.tensor([[0.2]])
        target = torch.tensor([[0.01]])

        plain = CensoredLaplaceNLL(bce_on_observed=False)(mu, raw_sigma, target)
        with_bce = CensoredLaplaceNLL(bce_on_observed=True)(mu, raw_sigma, target)
        b = _sigma(raw_sigma)
        u = (math.log(1.0 + SHIFT) - mu) / b
        # observed position adds -log F(u) = -log(1 - SF(u))
        extra = -torch.log1p(-torch.exp(_laplace_log_sf(u)))
        assert with_bce.item() == pytest.approx(plain.item() + extra.item(), rel=1e-5)

    def test_extreme_u_stable(self):
        """Very large |u| must not produce NaN/Inf."""
        loss_fn = CensoredLaplaceNLL()
        mu = torch.tensor([[-50.0, 50.0]])
        raw_sigma = torch.full_like(mu, -10.0)  # b ~ eps
        target = torch.tensor([[float("nan"), float("nan")]])
        loss = loss_fn(mu, raw_sigma, target)
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        mu = torch.tensor([[-4.0, -1.0]], requires_grad=True)
        raw_sigma = torch.zeros(1, 2, requires_grad=True)
        target = torch.tensor([[0.01, float("nan")]])

        CensoredLaplaceNLL()(mu, raw_sigma, target).backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert raw_sigma.grad is not None and torch.isfinite(raw_sigma.grad).all()

    def test_per_sample_matches_forward(self):
        loss_fn = CensoredLaplaceNLL()
        mu = torch.tensor([[-4.0, -3.0], [-5.0, -3.5]])
        raw_sigma = torch.zeros_like(mu)
        target = torch.tensor([[0.01, float("nan")], [0.002, 0.05]])

        per = loss_fn.per_sample(mu, raw_sigma, target)
        assert per.shape == (2,)
        assert per.mean().item() == pytest.approx(
            loss_fn(mu, raw_sigma, target).item(), rel=1e-5)

    def test_var_reg_forms(self):
        mu = torch.tensor([[-4.0]])
        raw_sigma = torch.tensor([[1.5]])
        target = torch.tensor([[0.01]])

        plain = CensoredLaplaceNLL(var_reg=0.0)(mu, raw_sigma, target)
        reg_sig = CensoredLaplaceNLL(var_reg=0.1, var_reg_form="sigma")(mu, raw_sigma, target)
        b = _sigma(raw_sigma)
        assert reg_sig.item() == pytest.approx(plain.item() + 0.1 * b.item() ** 2, rel=1e-5)
