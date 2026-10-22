"""
Unit tests for the closed-form CRPS used for best-checkpoint selection.

CRPS replaces val_nll as the selection metric for the probabilistic losses:
it is likewise a strictly proper scoring rule, but bounded as sigma -> 0
(where it reduces to the MAE), hence monotone in fit quality even when a
variance penalty shrinks sigma. See proT/training/losses.py.

Run with: pytest test/unit/test_crps.py -v
"""

import math
import sys
from os.path import abspath, dirname

import pytest
import torch

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from proT.training.losses import gaussian_crps, lognormal_crps


def test_lognormal_crps_matches_monte_carlo():
    """Closed form must match the CRPS definition E|X-y| - 0.5*E|X-X'|."""
    torch.manual_seed(0)
    mu = torch.tensor([[-5.6]])
    sigma = torch.tensor([[0.35]])
    y = torch.tensor([[0.0021]])
    shift = 1e-3

    closed = float(lognormal_crps(mu, sigma, y, shift))

    n = 2_000_000
    x = torch.exp(torch.randn(n) * sigma[0, 0] + mu[0, 0]) - shift
    mc = float((x - y[0, 0]).abs().mean() - 0.5 * (x[: n // 2] - x[n // 2 :]).abs().mean())

    assert closed == pytest.approx(mc, rel=2e-2)


def test_lognormal_crps_reduces_to_mae_as_sigma_goes_to_zero():
    """Key property: bounded (unlike the NLL) and equal to |residual|."""
    mu = torch.tensor([[-5.6]])
    y = torch.tensor([[0.004]])
    shift = 1e-3
    resid = abs(0.004 - (math.exp(-5.6) - shift))

    crps = float(lognormal_crps(mu, torch.tensor([[1e-6]]), y, shift))
    assert crps == pytest.approx(resid, rel=1e-4)


def test_lognormal_crps_is_proper_in_sigma():
    """CRPS is minimized at the sigma that generated the data (propriety)."""
    torch.manual_seed(0)
    true_mu, true_sigma, shift = -5.6, 0.4, 1e-3
    n = 400_000
    samples = torch.exp(torch.randn(n) * true_sigma + true_mu) - shift

    def mean_crps(sigma):
        mu_t = torch.full((n, 1), true_mu)
        s_t = torch.full((n, 1), sigma)
        return float(lognormal_crps(mu_t, s_t, samples.unsqueeze(1), shift))

    scores = {s: mean_crps(s) for s in (0.1, 0.25, 0.4, 0.6, 1.0)}
    assert min(scores, key=scores.get) == 0.4


def test_gaussian_crps_known_value():
    """CRPS(N(0,1), y=0) = 2*phi(0) - 1/sqrt(pi)."""
    val = float(gaussian_crps(torch.tensor([[0.0]]),
                              torch.tensor([[1.0]]),
                              torch.tensor([[0.0]])))
    expected = 2.0 / math.sqrt(2 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert val == pytest.approx(expected, rel=1e-6)


def test_crps_ignores_nan_targets():
    """NaN targets must be excluded from the mean, like the NLL losses."""
    mu = torch.tensor([[-5.6, -5.2]])
    sigma = torch.tensor([[0.3, 0.3]])
    both = torch.tensor([[0.002, float("nan")]])
    only = torch.tensor([[0.002]])

    a = float(lognormal_crps(mu, sigma, both, 1e-3))
    b = float(lognormal_crps(mu[:, :1], sigma[:, :1], only, 1e-3))
    assert a == pytest.approx(b, rel=1e-6)
    assert math.isfinite(a)


def test_crps_prefers_better_mean_fit():
    """A closer mu must score better at equal sigma (monotone in fit)."""
    y = torch.tensor([[0.005]])
    sigma = torch.tensor([[0.3]])
    good = torch.tensor([[math.log(0.005 + 1e-3)]])
    bad = torch.tensor([[math.log(0.020 + 1e-3)]])
    assert float(lognormal_crps(good, sigma, y, 1e-3)) < \
        float(lognormal_crps(bad, sigma, y, 1e-3))
