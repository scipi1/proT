"""
Unit tests for the optional learning controls:

1. freeze_dec_embedding (ProT model kwarg, idea 1)
2. delayed Q/K learning (QKFreezeScheduler callback + LR ramp, idea 2)
3. position-weighted, NaN-masked MSE (proT.training.losses, idea 3)

Run with: pytest test/unit/test_learning_options.py -v
"""

import math
from types import SimpleNamespace

import pytest
import torch

from proT.core.model import ProT
from proT.training.losses import PositionWeightedMSE, build_position_weights
from proT.training.qk_schedule import (
    build_qk_ramp_lambda,
    collect_qk_params,
)
from proT.training.callbacks.training_callbacks import QKFreezeScheduler
from proT.training.forecasters.entropy_regularized_forecaster import EntropyRegularizedForecaster


# ---------------------------------------------------------------------------
# helpers: tiny ProT model / forecaster config
# ---------------------------------------------------------------------------

D_MODEL = 8
N_VAR = 4  # variable vocabulary size


def _tiny_model_kwargs(freeze_dec_embedding=False):
    """Minimal ProT kwargs: variable (nn_embedding) + value (linear) + position (pass)."""
    ds_embed = {
        "setting": {"sparse_grad": False},
        "modules": [
            {"idx": 0, "embed": "nn_embedding", "label": "variable",
             "kwargs": {"num_embeddings": N_VAR, "embedding_dim": D_MODEL}},
            {"idx": 1, "embed": "linear", "label": "value",
             "kwargs": {"input_dim": 1, "embedding_dim": D_MODEL}},
            {"idx": 2, "embed": "pass", "label": "order", "kwargs": {}},
        ],
    }
    return dict(
        model="proT",
        ds_embed_enc=ds_embed,
        comps_embed_enc="summation",
        ds_embed_dec=ds_embed,
        comps_embed_dec="summation",
        enc_attention_type="ScaledDotProduct",
        enc_mask_type="Uniform",
        dec_self_attention_type="ScaledDotProduct",
        dec_self_mask_type="Uniform",
        dec_cross_attention_type="ScaledDotProduct",
        dec_cross_mask_type="Uniform",
        n_heads=1,
        enc_causal_mask=False,
        dec_causal_mask=False,
        dropout_emb=0.0,
        dropout_data=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        enc_dropout_qkv=0.0,
        enc_attention_dropout=0.0,
        dec_self_dropout_qkv=0.0,
        dec_self_attention_dropout=0.0,
        dec_cross_dropout_qkv=0.0,
        dec_cross_attention_dropout=0.0,
        e_layers=1,
        d_layers=1,
        activation="gelu",
        norm="layer",
        use_final_norm=True,
        device="cpu",
        out_dim=1,
        d_ff=16,
        d_model_enc=D_MODEL,
        d_model_dec=D_MODEL,
        d_qk=4,
        freeze_dec_embedding=freeze_dec_embedding,
    )


def _tiny_model(freeze_dec_embedding=False):
    return ProT(**_tiny_model_kwargs(freeze_dec_embedding=freeze_dec_embedding))


def _tiny_config(training_overrides=None, freeze_dec_embedding=False):
    training = {
        "loss_fn": "mse",
        "lr": 1e-3,
        "max_epochs": 100,
        "loss_pos_weight": "none",
    }
    if training_overrides:
        training.update(training_overrides)
    return {
        "model": {"kwargs": _tiny_model_kwargs(freeze_dec_embedding=freeze_dec_embedding)},
        "training": training,
        "data": {"val_idx": 1},
    }


def _tiny_batch(batch_size=2, seq_len=6, nan_tail=False):
    """Batch of (X, Y) with feature layout [variable, value, position]."""
    torch.manual_seed(0)
    X = torch.randn(batch_size, seq_len, 3)
    X[:, :, 0] = torch.randint(0, N_VAR, (batch_size, seq_len)).float()
    X[:, :, 2] = torch.arange(seq_len).float()
    Y = torch.randn(batch_size, seq_len, 3)
    Y[:, :, 0] = torch.randint(0, N_VAR, (batch_size, seq_len)).float()
    Y[:, :, 2] = torch.arange(seq_len).float()
    if nan_tail:
        Y[:, -2:, 1] = float("nan")  # missing tail values
    return X, Y


# ---------------------------------------------------------------------------
# idea 3: position weights
# ---------------------------------------------------------------------------

def test_position_weights_none_is_uniform():
    w = build_position_weights(10, scheme="none")
    assert torch.allclose(w, torch.ones(10))


def test_position_weights_linear_increasing_normalized():
    L = 10
    w = build_position_weights(L, scheme="linear")
    assert torch.all(w[1:] > w[:-1])                    # strictly increasing
    assert math.isclose(w.mean().item(), 1.0, abs_tol=1e-6)
    # linear ramp normalized to mean 1: w_i = 2(i+1)/(L+1)
    assert math.isclose(w[0].item(), 2.0 / (L + 1), abs_tol=1e-6)
    assert math.isclose(w[-1].item(), 2.0 * L / (L + 1), abs_tol=1e-6)


def test_position_weights_exp_monotonic():
    w = build_position_weights(20, scheme="exp", alpha=2.0)
    assert torch.all(w[1:] > w[:-1])
    assert math.isclose(w.mean().item(), 1.0, abs_tol=1e-6)
    # ratio last/first = exp(alpha)
    assert math.isclose((w[-1] / w[0]).item(), math.exp(2.0), rel_tol=1e-5)


def test_position_weights_invalid_scheme():
    with pytest.raises(ValueError):
        build_position_weights(5, scheme="bogus")


# ---------------------------------------------------------------------------
# idea 3: PositionWeightedMSE
# ---------------------------------------------------------------------------

def test_weighted_mse_excludes_nan_targets():
    loss_fn = PositionWeightedMSE(scheme="none")
    predicted = torch.zeros(2, 4)
    target = torch.tensor([[1.0, 2.0, float("nan"), 4.0],
                           [0.0, 0.0, 0.0, float("nan")]])
    loss = loss_fn(predicted, target)
    # 6 valid entries: squared errors 1, 4, 16 (row 0) and 0, 0, 0 (row 1)
    expected = (1.0 ** 2 + 2.0 ** 2 + 4.0 ** 2) / 6
    assert math.isclose(loss.item(), expected, abs_tol=1e-6)


def test_weighted_mse_matches_plain_mse_without_nans():
    loss_fn = PositionWeightedMSE(scheme="none")
    predicted = torch.randn(3, 7)
    target = torch.randn(3, 7)
    loss = loss_fn(predicted, target)
    expected = torch.nn.functional.mse_loss(predicted, target)
    assert math.isclose(loss.item(), expected.item(), abs_tol=1e-6)


def test_weighted_mse_tail_weighs_more():
    loss_fn = PositionWeightedMSE(scheme="linear")
    predicted = torch.zeros(1, 3)
    loss_head = loss_fn(predicted, torch.tensor([[3.0, 0.0, 0.0]]))
    loss_tail = loss_fn(predicted, torch.tensor([[0.0, 0.0, 3.0]]))
    # L=3 linear normalized weights: [0.5, 1.0, 1.5]
    assert math.isclose(loss_head.item(), 9.0 * 0.5 / 3.0, abs_tol=1e-6)
    assert math.isclose(loss_tail.item(), 9.0 * 1.5 / 3.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# idea 1: freeze decoder embeddings
# ---------------------------------------------------------------------------

def test_freeze_dec_embedding_flags():
    model = _tiny_model(freeze_dec_embedding=True)
    dec_params = list(model.dec_embedding.parameters())
    assert len(dec_params) > 0
    assert all(not p.requires_grad for p in dec_params)
    # everything else stays trainable
    assert all(p.requires_grad for p in model.enc_embedding.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())


def test_frozen_dec_embedding_receives_no_gradient():
    model = _tiny_model(freeze_dec_embedding=True)
    X, Y = _tiny_batch()
    forecast_out, *_ = model.forward(X, Y)
    forecast_out.pow(2).mean().backward()
    assert all(p.grad is None for p in model.dec_embedding.parameters())
    # sanity: encoder and decoder attention do receive gradients
    assert any(p.grad is not None for p in model.enc_embedding.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_unfrozen_dec_embedding_receives_gradient():
    model = _tiny_model(freeze_dec_embedding=False)
    X, Y = _tiny_batch()
    forecast_out, *_ = model.forward(X, Y)
    forecast_out.pow(2).mean().backward()
    assert any(p.grad is not None for p in model.dec_embedding.parameters())


# ---------------------------------------------------------------------------
# idea 2: delayed Q/K learning
# ---------------------------------------------------------------------------

def test_collect_qk_params_scopes():
    model = _tiny_model()
    # 1 encoder layer (self) + 1 decoder layer (self + cross) = 3 attention layers
    # each contributes query+key projections with weight+bias = 4 tensors
    assert len(collect_qk_params(model, "all")) == 3 * 4
    assert len(collect_qk_params(model, "self")) == 2 * 4
    assert len(collect_qk_params(model, "cross")) == 1 * 4
    with pytest.raises(AssertionError):
        collect_qk_params(model, "bogus")


def test_qk_freeze_scheduler_toggles_requires_grad():
    model = _tiny_model()
    pl_module = SimpleNamespace(model=model)
    callback = QKFreezeScheduler(freeze_epochs=2, scope="all")
    qk_params = collect_qk_params(model, "all")

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), pl_module)
    assert all(not p.requires_grad for p in qk_params)

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=1), pl_module)
    assert all(not p.requires_grad for p in qk_params)

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=2), pl_module)
    assert all(p.requires_grad for p in qk_params)

    # non-Q/K params are never touched
    assert all(p.requires_grad for p in model.enc_embedding.parameters())


def test_qk_ramp_lambda_shape():
    lr_lambda = build_qk_ramp_lambda(max_epochs=100, end_frac=0.95, start_factor=1e-4)
    assert math.isclose(lr_lambda(0), 1e-4, abs_tol=1e-8)   # very low start
    assert lr_lambda(10) < lr_lambda(47)                    # linearly increasing
    assert math.isclose(lr_lambda(95), 1.0, abs_tol=1e-8)   # full LR at 95%
    assert lr_lambda(150) == 1.0                            # constant afterwards


# ---------------------------------------------------------------------------
# forecaster-level plumbing
# ---------------------------------------------------------------------------

def test_forecaster_step_with_all_options():
    """Smoke test: forward + loss with frozen dec emb, NaN tail, linear weights."""
    config = _tiny_config(
        training_overrides={"loss_pos_weight": "linear"},
        freeze_dec_embedding=True,
    )
    forecaster = EntropyRegularizedForecaster(config)
    X, Y = _tiny_batch(nan_tail=True)
    loss, predicted, _ = forecaster._step((X, Y), stage="train")
    assert torch.isfinite(loss)
    assert predicted.shape == (X.size(0), X.size(1))
    # dec embeddings frozen via config plumbing
    assert all(not p.requires_grad for p in forecaster.model.dec_embedding.parameters())


def test_forecaster_default_config_unchanged():
    """Defaults keep previous behavior: trainable dec emb, uniform loss."""
    config = _tiny_config()
    forecaster = EntropyRegularizedForecaster(config)
    assert all(p.requires_grad for p in forecaster.model.dec_embedding.parameters())
    X, Y = _tiny_batch()
    loss, _, _ = forecaster._step((X, Y), stage="train")
    assert torch.isfinite(loss)


def test_forecaster_qk_lr_ramp_optimizer():
    """qk_lr_ramp builds a 2-group AdamW with a LambdaLR scheduler."""
    config = _tiny_config(training_overrides={"qk_lr_ramp": True})
    forecaster = EntropyRegularizedForecaster(config)
    optimizers, schedulers = forecaster.configure_optimizers()
    assert len(optimizers) == 1
    assert len(optimizers[0].param_groups) == 2  # other params + Q/K params
    assert len(schedulers) == 1
    lr_lambdas = schedulers[0]["scheduler"].lr_lambdas
    assert lr_lambdas[0](0) == 1.0                      # base group: constant
    assert math.isclose(lr_lambdas[1](0), 1e-4, abs_tol=1e-8)  # Q/K group: low start
    assert math.isclose(lr_lambdas[1](95), 1.0, abs_tol=1e-8)  # full at 95%


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
