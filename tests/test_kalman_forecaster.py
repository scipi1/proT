"""Integration smoke test: EncoderForecaster built from the kalman_mlp
arm config runs one conditioned training step (per-sample fractions) and one
validation step (free run + fixed-fraction diagnostics) on random data, CPU.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CFG = (ROOT / "experiments" / "training" / "RNN"
       / "proTEncGRUpervar_smooth2_delta_kalman_mlp"
       / "config_proTEncGRUpervar_smooth2_delta_kalman_mlp.yaml")


def make_batch(cfg, B=4, L_enc=30, E=15, seed=0):
    """Random encoder input and a two-variable target with increasing values."""
    rng = np.random.default_rng(seed)
    fx, fy = cfg["data"]["features"]["X"], cfg["data"]["features"]["Y"]
    X = np.zeros((B, L_enc, 14), dtype=np.float32)
    X[:, :, 0] = 1
    X[:, :, fx["process"]] = rng.integers(1, 9, (B, L_enc))
    X[:, :, fx["occurrence"]] = rng.integers(1, 5, (B, L_enc))
    X[:, :, fx["step"]] = rng.integers(1, 9, (B, L_enc))
    X[:, :, fx["variable"]] = rng.integers(1, 407, (B, L_enc))
    X[:, :, fx["value"]] = rng.normal(size=(B, L_enc))
    X[:, :, fx["order"]] = np.arange(L_enc)
    X[:, :, fx["design"]] = rng.integers(1, 6, (B, L_enc))
    X[:, :, fx["panel_position"]] = rng.integers(1, 3, (B, L_enc))
    Y = np.zeros((B, 2 * E, 10), dtype=np.float32)
    for v in (1, 2):
        s = slice((v - 1) * E, v * E)
        Y[:, s, fy["position"]] = np.arange(1, E + 1)
        Y[:, s, fy["variable"]] = v
        Y[:, s, fy["value"]] = np.sort(rng.uniform(0, 0.7, (B, E)), axis=1)
    return torch.from_numpy(X), torch.from_numpy(Y)


@pytest.mark.skipif(not CFG.exists(), reason="kalman_mlp config not found")
def test_kalman_arm_train_and_val_step():
    from proT.training.experiment_control import update_config
    from proT.training.forecasters.encoder_forecaster import EncoderForecaster

    cfg = OmegaConf.load(CFG)
    cfg.model.kwargs.device = "cpu"
    cfg.model.kwargs.d_hidden = 16      # tiny model, same structure
    cfg = update_config(cfg)
    torch.manual_seed(0)
    fc = EncoderForecaster(cfg)
    assert fc.model.rnn_update == "innovation"
    assert hasattr(fc.model, "update_cell")
    fc.log = lambda *a, **k: None        # no Lightning trainer attached

    X, Y = make_batch(cfg)
    fc.train()
    loss = fc.training_step((X, Y), 0)
    assert loss is not None and torch.isfinite(loss)
    assert torch.is_tensor(fc.show_trg_max_idx) and fc.show_trg_max_idx.shape == (4,)
    loss.backward()
    g = fc.model.update_cell.gates[0].weight.grad
    assert g is not None and torch.isfinite(g).all()

    fc.eval()
    with torch.no_grad():
        vloss = fc.validation_step((X, Y), 0)
    assert torch.isfinite(vloss)
