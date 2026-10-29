"""Tests: crossing-time CRPS, loss components and sigma temperature scaling."""

import math

import numpy as np
import pytest
import torch

from proT.training.losses import (
    CensoredLogNormalNLL, GaussianNLL, crossing_time_crps)
from proT.evaluation.calibration import fit_temperature, apply_temperature

SHIFT, THR = 1e-3, 1.0
LOG_THR = math.log(THR + SHIFT)


def _failing_batch(L=20, t_fail=10):
    """One variable, observed rising to 0.9 at t_fail, censored after."""
    pos = torch.arange(1, L + 1, dtype=torch.float32).unsqueeze(0)
    val = torch.linspace(0.1, 0.9, t_fail)
    target = torch.full((1, L), float("nan"))
    target[0, :t_fail] = val
    var = torch.ones(1, L)
    masks = CensoredLogNormalNLL.classify_termination(
        target, var, pos, expected_len=L, threshold=THR, fail_margin=0.8)
    return target, masks


def _ttf_for_crossing(masks, L, t_cross, sharp=1e-3):
    """Loss for a step path crossing the threshold at index t_cross."""
    mu = torch.full((1, L), LOG_THR - 1.0)
    mu[0, t_cross:] = LOG_THR + 1.0
    raw = torch.full((1, L), math.log(math.expm1(sharp)))
    return crossing_time_crps(mu, raw, masks, LOG_THR, L).item()


def test_ttf_minimal_at_true_failure_and_grows_with_distance():
    L, t_fail = 20, 10
    _, masks = _failing_batch(L, t_fail)
    assert masks["honest"][0, 0] == 1
    losses = {t: _ttf_for_crossing(masks, L, t) for t in range(2, L)}
    # crossing right after the anchor (index t_fail) is optimal
    assert min(losses, key=losses.get) in (t_fail - 1, t_fail)
    assert losses[t_fail + 5] > losses[t_fail + 2] > losses[t_fail]
    assert losses[t_fail - 6] > losses[t_fail - 3] > losses[t_fail]


def test_ttf_finite_with_drifted_inactive_positions():
    L = 20
    _, masks = _failing_batch(L)
    mu = torch.zeros(1, L)
    mu[0, -1] = float("inf")  # beyond support only if masked; here survival
    raw = torch.zeros(1, L, requires_grad=True)
    loss = crossing_time_crps(torch.nan_to_num(mu, posinf=1e3), raw, masks,
                              LOG_THR, L)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()


def test_components_consistent_with_observed_nll():
    target, masks = _failing_batch()
    fn = CensoredLogNormalNLL(threshold=THR, shift=SHIFT, expected_len=20)
    mu = torch.randn(1, 20) * 0.1 - 1.0
    raw = torch.randn(1, 20) * 0.1
    c = fn.components(mu, raw, target, masks)
    sigma = torch.nn.functional.softplus(raw) + fn.eps
    log_y = torch.log(torch.nan_to_num(target) + SHIFT)
    nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma) \
        + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
    assert torch.allclose(c["obs_nll"], nll[masks["observed"]].mean(), atol=1e-5)
    g = GaussianNLL().components(mu, raw, target)
    assert set(g) == {"err", "logsig", "sigma_med", "sigma_p05"}


def test_sigma_y_floor_penalty():
    """Floor violated at y=0, satisfied in the bulk; zero when wide enough."""
    fn = CensoredLogNormalNLL(threshold=THR, shift=SHIFT, expected_len=20,
                              sigma_y_floor=0.005, sigma_y_floor_weight=1.0)
    target, masks = _failing_batch()
    # tiny sigma at all positions: early y ~ 0.1 -> (y+s)*sigma << 0.005
    raw_tiny = torch.full((1, 20), math.log(math.expm1(0.001)))
    loss_tiny = fn(torch.zeros(1, 20) - 1.0, raw_tiny, target, masks=masks)
    # wide sigma: (y+s)*sigma >> floor everywhere -> zero penalty
    raw_wide = torch.full((1, 20), math.log(math.expm1(1.0)))
    nll_wide = CensoredLogNormalNLL(threshold=THR, shift=SHIFT,
                                    expected_len=20)(
        torch.zeros(1, 20) - 1.0, raw_wide, target, masks=masks)
    loss_wide = fn(torch.zeros(1, 20) - 1.0, raw_wide, target, masks=masks)
    assert loss_wide.item() == pytest.approx(nll_wide.item(), abs=1e-9)
    assert loss_tiny > 0 and torch.isfinite(loss_tiny)
    c = fn.components(torch.zeros(1, 20) - 1.0, raw_tiny, target, masks)
    assert c["yfloor_viol_frac"] > 0.9


def test_ttf_arm_train_and_val_step_logs_components():
    """One train + val step of the ttf arm config (tiny, CPU); all new
    terms finite and logged."""
    from pathlib import Path
    import pytest
    from omegaconf import OmegaConf
    from proT.training.experiment_control import update_config
    from proT.training.forecasters.encoder_forecaster import EncoderForecaster
    from tests.test_kalman_forecaster import make_batch

    cfg_path = (Path(__file__).resolve().parents[1] / "experiments" / "training"
                / "RNN" / "proTEncGRUpervar_smooth2_delta_kalman_mlp_ttf"
                / "config_proTEncGRUpervar_smooth2_delta_kalman_mlp_ttf.yaml")
    if not cfg_path.exists():
        pytest.skip("ttf config not found")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.kwargs.device = "cpu"
    cfg.model.kwargs.d_hidden = 16
    cfg.training.loss_expected_len = 15   # make_batch uses E = 15
    cfg = update_config(cfg)
    torch.manual_seed(0)
    fc = EncoderForecaster(cfg)
    logged = {}
    fc.log = lambda k, v, **kw: logged.__setitem__(k, v)

    X, Y = make_batch(cfg)
    # sample 0, var 1: honest failure (0.9 at obs 8, censored afterwards)
    val_idx = cfg["data"]["features"]["Y"]["value"]
    Y[0, :8, val_idx] = torch.linspace(0.2, 0.9, 8)
    Y[0, 8:15, val_idx] = float("nan")
    fc.train()
    loss = fc.training_step((X, Y), 0)
    assert torch.isfinite(loss)
    loss.backward()
    for k in ("train_ttf_crps", "train_cum_obs_err", "train_cum_obs_logsig",
              "train_cum_surv", "train_inc_err", "train_inc_logsig"):
        assert k in logged and torch.isfinite(torch.as_tensor(logged[k])), k
    fc.eval()
    with torch.no_grad():
        assert torch.isfinite(fc.validation_step((X, Y), 0))
    assert "val_cum_obs_err" in logged and "val_ttf_crps" in logged


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n = 20000
    mu = rng.normal(-1.0, 0.3, n)
    true_s = 0.2
    val = np.exp(mu + true_s * rng.standard_normal(n)) - SHIFT
    sigma = np.full(n, true_s / 2.0)  # model claims half the true spread
    obs = np.ones(n, dtype=bool)
    fit = fit_temperature(mu, sigma, val, obs, np.zeros(n, dtype=bool), SHIFT, THR)
    assert abs(fit["T"] - 2.0) < 0.05
    assert abs(fit["closed_form_obs_T"] - fit["T"]) < 0.01
    assert fit["nll_after"] < fit["nll_before"]
    var = np.ones(n)
    assert np.allclose(apply_temperature(sigma, var, {1: fit}), sigma * fit["T"])
