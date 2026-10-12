"""
Unit tests for the encoder-only ProTEncoder (proT_enc) and EncoderForecaster.

Covers: output shapes for both head variants, attention pooling, the
forcing-boundary logic (real IST at inference), censored-loss backward,
checkpoint roundtrip, and the scheduled curriculum.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from proT.core.model_encoder import ProTEncoder
from proT.training.forecasters.encoder_forecaster import EncoderForecaster
from proT.training.losses import CensoredGaussianNLL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_config(head_type="gru", output_dim=None, d_hidden=8, rnn_n_layers=1,
                out_dim=2, d_model_enc=16, d_model_dec=16, e_layers=2,
                n_heads=2, d_qk=8, d_ff=32, dropout=0.0,
                n_dec_vars=None, dec_var_idx=None, **extra):
    """Tiny config for fast unit tests. Extra kwargs land in model kwargs."""
    cfg = {
        "model": {
            "model_object": "proT_enc",
            "kwargs": {
                "model": "proT_enc",
                "ds_embed_enc": {
                    "setting": {"sparse_grad": False},
                    "modules": [
                        {"idx": 0, "embed": "nn_embedding", "label": "variable",
                         "kwargs": {"num_embeddings": 5, "embedding_dim": d_model_enc}},
                        {"idx": 1, "embed": "linear", "label": "value",
                         "kwargs": {"input_dim": 1, "embedding_dim": d_model_enc}},
                        {"idx": 1, "embed": "mask", "label": "value_missing", "kwargs": {}},
                        {"idx": 2, "embed": "pass", "label": "order", "kwargs": {}},
                    ],
                },
                "comps_embed_enc": "summation",
                "ds_embed_dec": {
                    "setting": {"sparse_grad": False},
                    "modules": [
                        {"idx": 0, "embed": "nn_embedding", "label": "variable",
                         "kwargs": {"num_embeddings": 3, "embedding_dim": d_model_dec}},
                        {"idx": 1, "embed": "nn_embedding", "label": "position",
                         "kwargs": {"num_embeddings": 20, "embedding_dim": d_model_dec}},
                        {"idx": 2, "embed": "linear", "label": "value",
                         "kwargs": {"input_dim": 1, "embedding_dim": d_model_dec}},
                        {"idx": 2, "embed": "mask", "label": "value_missing", "kwargs": {}},
                        {"idx": 2, "embed": "mask_given", "label": "value_given", "kwargs": {}},
                        {"idx": 1, "embed": "pass", "label": "order", "kwargs": {}},
                        {"idx": 3, "embed": "nn_embedding", "label": "online_pos_mask",
                         "kwargs": {"num_embeddings": 3, "embedding_dim": d_model_dec}},
                    ],
                },
                "comps_embed_dec": "summation",
                "enc_attention_type": "ScaledDotProduct",
                "enc_mask_type": "Uniform",
                "n_heads": n_heads,
                "enc_causal_mask": False,
                "enc_group_mask": False,
                "dropout_emb": dropout,
                "dropout_data": dropout,
                "dropout_attn_out": dropout,
                "dropout_ff": dropout,
                "enc_dropout_qkv": dropout,
                "enc_attention_dropout": dropout,
                "e_layers": e_layers,
                "activation": "gelu",
                "norm": "layer",
                "use_final_norm": True,
                "device": "cpu",
                "out_dim": out_dim,
                "d_ff": d_ff,
                "d_model_enc": d_model_enc,
                "d_model_dec": d_model_dec,
                "d_qk": d_qk,
                "head_type": head_type,
                "output_dim": output_dim,
                "d_hidden": d_hidden,
                "rnn_n_layers": rnn_n_layers,
                "rnn_dropout": 0.0,
                "dec_val_idx": 2,
                "dec_pos_idx": 1,
                "n_dec_vars": n_dec_vars,
                "dec_var_idx": dec_var_idx,
                **extra,
            },
        },
        "training": {
            "loss_fn": "gaussian_censored",
            "loss_censor_threshold": 1.0,
            "loss_bce_on_observed": False,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "lam": 1.0,
            "gamma": 0.1,
            "entropy_regularizer": False,
            "max_epochs": 10,
            "target_show_mode": "scheduled",
            "epoch_show_trg": 2,
            "show_trg_max_idx_start": 5,
            "show_trg_max_idx_end": 0,
            "show_trg_schedule": "linear",
            "show_trg_schedule_end_frac": 1.0,
            "show_trg_random_draw": False,
        },
        "data": {
            "val_idx": 2,
            "pos_idx": 1,
        },
    }
    return cfg


def _tiny_batch(B=4, L_enc=6, L_dec=10, D_enc=3, D_dec=4, nan_tail=False):
    """Tiny batch of (X, Y) with optional NaN tail in the target value."""
    X = torch.randn(B, L_enc, D_enc)
    X[:, :, 0] = torch.randint(1, 5, (B, L_enc)).float()  # variable ids
    X[:, :, 2] = torch.arange(L_enc).float().unsqueeze(0).expand(B, -1)  # order

    Y = torch.randn(B, L_dec, D_dec)
    Y[:, :, 0] = torch.randint(1, 3, (B, L_dec)).float()  # variable ids
    Y[:, :, 1] = torch.arange(1, L_dec + 1).float().unsqueeze(0).expand(B, -1)  # position
    Y[:, :, 3] = 0.0  # given indicator (overwritten by forecaster)

    if nan_tail:
        # simulate right-censoring: last 30% of positions are NaN
        n_nan = int(0.3 * L_dec)
        Y[:, -n_nan:, 2] = float("nan")

    return X, Y


def _tiny_batch_pervar(B=4, L_enc=6, L_per_var=5, D_enc=3, D_dec=4, n_vars=2,
                       nan_tail=False):
    """Tiny batch with a variable-major target layout: [var 1 block, var 2 block].

    Positions restart from 1 within each variable block.
    """
    X = torch.randn(B, L_enc, D_enc)
    X[:, :, 0] = torch.randint(1, 5, (B, L_enc)).float()  # variable ids
    X[:, :, 2] = torch.arange(L_enc).float().unsqueeze(0).expand(B, -1)  # order

    L_dec = L_per_var * n_vars
    Y = torch.randn(B, L_dec, D_dec)
    var_ids = torch.cat([torch.full((L_per_var,), float(v)) for v in range(1, n_vars + 1)])
    Y[:, :, 0] = var_ids.unsqueeze(0).expand(B, -1)  # variable ids (same layout for all)
    pos = torch.arange(1, L_per_var + 1).float().repeat(n_vars)
    Y[:, :, 1] = pos.unsqueeze(0).expand(B, -1)  # position restarts per variable
    Y[:, :, 3] = 0.0  # given indicator (overwritten by forecaster)

    if nan_tail:
        # simulate right-censoring: last position of each variable block is NaN
        Y[:, [L_per_var - 1, L_dec - 1], 2] = float("nan")

    return X, Y


# ---------------------------------------------------------------------------
# ProTEncoder model
# ---------------------------------------------------------------------------

class TestProTEncoder:
    def test_gru_forward_shapes(self):
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch()
        out, recon, (enc_att, dec_att, pool_att), enc_mask, (enc_ent, dec_ent, pool_ent) = \
            model.forward(input_tensor=X, target_tensor=Y)
        assert out.shape == (4, 10, 2)          # (B, L_dec, out_dim)
        assert recon.shape == (4, 6, 1)         # (B, L_enc, 1)
        assert len(enc_att) == 2                # e_layers
        assert dec_att == []                    # no decoder self-attention
        assert len(pool_att) == 1               # pooling attention
        assert len(enc_ent) == 2
        assert dec_ent == []
        assert pool_ent == []

    def test_scalar_forward_shapes(self):
        cfg = _tiny_config(head_type="scalar", output_dim=5, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, _ = _tiny_batch()
        out, recon, (enc_att, dec_att, pool_att), enc_mask, (enc_ent, dec_ent, pool_ent) = \
            model.forward(input_tensor=X, target_tensor=None)
        assert out.shape == (4, 5, 2)           # (B, output_dim, out_dim)
        assert recon.shape == (4, 6, 1)
        assert len(enc_att) == 2
        assert dec_att == []
        assert len(pool_att) == 1

    def test_pooling_respects_missing_mask(self):
        """Tokens with missing values should not contribute to the pooled context."""
        cfg = _tiny_config(head_type="scalar", output_dim=3, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, _ = _tiny_batch()
        # blank out the value channel of the first two encoder tokens
        X[:, :2, 1] = float("nan")
        out_masked, _, _, _, _ = model.forward(input_tensor=X, target_tensor=None)
        X2, _ = _tiny_batch()
        out_full, _, _, _, _ = model.forward(input_tensor=X2, target_tensor=None)
        # the pooled context should differ when tokens are masked
        assert not torch.allclose(out_masked, out_full)

    def test_gru_teacher_forcing_boundary(self):
        """With show_trg_max_idx=k, positions <= k must receive the real value."""
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch(L_dec=10)
        k = 4
        trg_pos_mask = (Y[:, :, 1] > k).unsqueeze(-1)  # True = predict, False = given
        out_forced, _, _, _, _ = model.forward(
            input_tensor=X, target_tensor=Y, trg_pos_mask=trg_pos_mask)
        out_free, _, _, _, _ = model.forward(
            input_tensor=X, target_tensor=Y, trg_pos_mask=None)
        # the forced and free runs should differ after the forcing boundary
        assert not torch.allclose(out_forced[:, k:, :], out_free[:, k:, :])

    def test_backward_censored(self):
        """CensoredGaussianNLL must backpropagate through the GRU head."""
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch(nan_tail=True)
        out, _, _, _, _ = model.forward(input_tensor=X, target_tensor=Y)
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        loss = loss_fn(out[:, :, 0], out[:, :, 1], Y[:, :, 2])
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.parameters())

    def test_backward_scalar(self):
        """Scalar head must backpropagate."""
        cfg = _tiny_config(head_type="scalar", output_dim=4, out_dim=2)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, _ = _tiny_batch()
        out, _, _, _, _ = model.forward(input_tensor=X, target_tensor=None)
        loss = out.pow(2).mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


class TestProTEncoderPerVar:
    """Tests for the per-variable GRU head (head_type="gru_per_var")."""

    def _pervar_config(self, **kw):
        return _tiny_config(head_type="gru_per_var", d_hidden=8, out_dim=2,
                            n_dec_vars=2, dec_var_idx=0, **kw)

    def test_gru_per_var_forward_shapes(self):
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch_pervar(L_per_var=5)
        out, recon, (enc_att, dec_att, pool_att), enc_mask, (enc_ent, dec_ent, pool_ent) = \
            model.forward(input_tensor=X, target_tensor=Y)
        assert out.shape == (4, 10, 2)          # (B, L_dec, out_dim)
        assert recon.shape == (4, 6, 1)
        assert dec_att == []
        assert len(pool_att) == 1

    def test_gru_per_var_stream_independence(self):
        """Variable 2 predictions must not depend on variable 1's given values."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        model.eval()
        X, Y = _tiny_batch_pervar(L_per_var=5)
        L_per_var = 5
        k = 3  # force the first k positions of each variable
        trg_pos_mask = (Y[:, :, 1] > k).unsqueeze(-1)  # True = predict, False = given

        with torch.no_grad():
            out1, _, _, _, _ = model.forward(
                input_tensor=X, target_tensor=Y, trg_pos_mask=trg_pos_mask)
            # perturb variable 1's given values
            Y2 = Y.clone()
            Y2[:, :L_per_var, 2] += 1.0
            out2, _, _, _, _ = model.forward(
                input_tensor=X, target_tensor=Y2, trg_pos_mask=trg_pos_mask)

        # variable 1 predictions change (different given values) ...
        assert not torch.allclose(out1[:, :L_per_var, :], out2[:, :L_per_var, :])
        # ... but variable 2 predictions are identical (independent stream)
        assert torch.allclose(out1[:, L_per_var:, :], out2[:, L_per_var:, :])

    def test_gru_per_var_distinct_initial_states(self):
        """Each variable's rollout must start from its own h0 (per-variable GRUs)."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        assert isinstance(model.rnn, torch.nn.ModuleList) and len(model.rnn) == 2
        assert isinstance(model.rnn_head, torch.nn.ModuleList) and len(model.rnn_head) == 2
        # the h0 projection is shared across variables: it maps each variable's
        # context slice (n_ctx_queries * d_model_enc) to n_layers * d_hidden
        assert model.h0_projection.out_features == 1 * 8

    def test_gru_per_var_multi_ctx_queries_and_feed(self):
        """C1+P1: k pool queries per variable, context MLP, per-step ctx feed."""
        cfg = self._pervar_config(n_ctx_queries=4, ctx_mlp_hidden=16,
                                  rnn_ctx_feed=True, rnn_ctx_feed_dim=6)
        model = ProTEncoder(**cfg["model"]["kwargs"])
        # 2 variables x 4 queries
        assert model.pool_query.shape == (1, 8, model.pool_query.shape[-1])
        # GRU input grows by the ctx-feed dim w.r.t. the baseline (no feed)
        base = ProTEncoder(**self._pervar_config()["model"]["kwargs"])
        assert model.rnn[0].input_size == base.rnn[0].input_size + 6
        X, Y = _tiny_batch_pervar(L_per_var=5)
        trg_pos_mask = (Y[:, :, 1] > 2).unsqueeze(-1)
        out, _, _, _, _ = model.forward(
            input_tensor=X, target_tensor=Y, trg_pos_mask=trg_pos_mask)
        assert out.shape == (Y.shape[0], Y.shape[1], 2)
        assert torch.isfinite(out).all()

    def test_gru_per_var_teacher_forcing_boundary(self):
        """Forced and free runs must differ after the forcing boundary (both variables)."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch_pervar(L_per_var=5)
        k = 2
        trg_pos_mask = (Y[:, :, 1] > k).unsqueeze(-1)
        out_forced, _, _, _, _ = model.forward(
            input_tensor=X, target_tensor=Y, trg_pos_mask=trg_pos_mask)
        out_free, _, _, _, _ = model.forward(
            input_tensor=X, target_tensor=Y, trg_pos_mask=None)
        assert not torch.allclose(out_forced[:, k:, :], out_free[:, k:, :])

    def test_gru_per_var_ragged_layout(self):
        """Truncated/shifted variable blocks and NaN tail padding must work."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        model.eval()
        X, Y = _tiny_batch_pervar(L_per_var=5)
        # sample 1: variable 1 truncated by one token (packed) -> variable 2
        # block shifted by one position, NaN padding at the tail
        Y[1, :, 0] = torch.cat([
            torch.full((4,), 1.0), torch.full((5,), 2.0),
            torch.tensor([float("nan")])])
        # sample 2: NaN padding at the tail of the variable 2 block
        Y[2, -2:, 0] = float("nan")
        with torch.no_grad():
            out, _, _, _, _ = model.forward(input_tensor=X, target_tensor=Y)
        assert out.shape == (4, 10, 2)
        assert torch.isfinite(out).all()
        # padded positions (NaN variable id) keep zero output
        assert (out[1, -1, :] == 0).all()
        assert (out[2, -2:, :] == 0).all()

    def test_gru_per_var_scatter_back_consistency(self):
        """A shifted variable block must yield the same predictions as the
        aligned block when the per-variable inputs are identical."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        model.eval()
        X, Y = _tiny_batch_pervar(B=2, L_per_var=5)
        # identical encoder input -> identical h0 for both samples
        X[1] = X[0]
        # sample 1: drop the last var-1 token, shift the var-2 block one position
        # to the left, NaN padding at the tail; var-2 covariates identical to
        # sample 0's var-2 block
        Y[1, :, 0] = torch.cat([
            torch.full((4,), 1.0), torch.full((5,), 2.0),
            torch.tensor([float("nan")])])
        Y[1, 4:9, 1:] = Y[0, 5:10, 1:]
        with torch.no_grad():
            out, _, _, _, _ = model.forward(input_tensor=X, target_tensor=Y)
        # var-2 predictions of the shifted sample match the aligned sample
        assert torch.allclose(out[1, 4:9, :], out[0, 5:10, :], atol=1e-5)
        # padded position keeps zero output
        assert (out[1, -1, :] == 0).all()

    def test_backward_censored_per_var(self):
        """CensoredGaussianNLL must backpropagate through both per-variable GRUs."""
        cfg = self._pervar_config()
        model = ProTEncoder(**cfg["model"]["kwargs"])
        X, Y = _tiny_batch_pervar(nan_tail=True)
        out, _, _, _, _ = model.forward(input_tensor=X, target_tensor=Y)
        loss_fn = CensoredGaussianNLL(threshold=1.0)
        loss = loss_fn(out[:, :, 0], out[:, :, 1], Y[:, :, 2])
        loss.backward()
        assert torch.isfinite(loss)
        # both per-variable GRUs and heads receive gradients
        for rnn, head in zip(model.rnn, model.rnn_head):
            assert rnn.weight_ih_l0.grad is not None
            assert head.weight.grad is not None


# ---------------------------------------------------------------------------
# EncoderForecaster
# ---------------------------------------------------------------------------

class TestEncoderForecaster:
    def test_forecaster_step_gru(self):
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch(nan_tail=True)
        loss, predicted, _ = forecaster._step((X, Y), stage="train")
        assert torch.isfinite(loss)
        assert predicted.shape == (4, 10)

    def test_forecaster_step_scalar(self):
        cfg = _tiny_config(head_type="scalar", output_dim=5, out_dim=2)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch()
        # scalar head: the target is a D-dimensional vector of scalars
        Y_scalar = Y[:, :5, :].clone()
        loss, predicted, _ = forecaster._step((X, Y_scalar), stage="train")
        assert torch.isfinite(loss)
        assert predicted.shape == (4, 5)

    def test_scheduled_bound(self):
        cfg = _tiny_config(head_type="gru")
        forecaster = EncoderForecaster(cfg)
        # before curriculum start
        assert forecaster.curriculum.scheduled_bound(0) == 5
        assert forecaster.curriculum.scheduled_bound(2) == 5
        # at the end
        assert forecaster.curriculum.scheduled_bound(9) == 0
        # monotonic decrease
        bounds = [forecaster.curriculum.scheduled_bound(e) for e in range(2, 10)]
        assert all(b1 >= b2 for b1, b2 in zip(bounds, bounds[1:]))

    def test_real_ist_inference(self):
        """At inference, show_trg_max_idx=k feeds the real value for positions <= k."""
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        forecaster = EncoderForecaster(cfg)
        forecaster.show_trg_active = True
        forecaster.show_trg_max_idx = 4
        X, Y = _tiny_batch(L_dec=10)
        out_forced, _, _, _, _ = forecaster.forward(
            data_input=X, data_trg=Y, predict_mode=True)
        forecaster.show_trg_max_idx = 0
        out_free, _, _, _, _ = forecaster.forward(
            data_input=X, data_trg=Y, predict_mode=True)
        assert not torch.allclose(out_forced[:, 4:, :], out_free[:, 4:, :])

    def test_checkpoint_roundtrip(self):
        """The forecaster must save and load from a checkpoint."""
        import tempfile
        import os
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, TensorDataset
        cfg = _tiny_config(head_type="gru", d_hidden=8, out_dim=2)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch()
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.ckpt")
            trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(forecaster, train_dataloaders=dl)
            trainer.save_checkpoint(ckpt)
            loaded = EncoderForecaster.load_from_checkpoint(ckpt)
        assert loaded.model.head_type == "gru"
        out1, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        out2, _, _, _, _ = loaded.forward(data_input=X, data_trg=Y)
        assert torch.allclose(out1, out2)  # type: ignore

    def test_forecaster_step_gru_per_var(self):
        cfg = _tiny_config(head_type="gru_per_var", d_hidden=8, out_dim=2,
                           n_dec_vars=2, dec_var_idx=0)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch_pervar(nan_tail=True)
        loss, predicted, _ = forecaster._step((X, Y), stage="train")
        assert torch.isfinite(loss)
        assert predicted.shape == (4, 10)

    def test_checkpoint_roundtrip_per_var(self):
        """The per-variable GRU forecaster must save and load from a checkpoint."""
        import tempfile
        import os
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader, TensorDataset
        cfg = _tiny_config(head_type="gru_per_var", d_hidden=8, out_dim=2,
                           n_dec_vars=2, dec_var_idx=0)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch_pervar()
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.ckpt")
            trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(forecaster, train_dataloaders=dl)
            trainer.save_checkpoint(ckpt)
            loaded = EncoderForecaster.load_from_checkpoint(ckpt)
        assert loaded.model.head_type == "gru_per_var"
        out1, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        out2, _, _, _, _ = loaded.forward(data_input=X, data_trg=Y)
        assert torch.allclose(out1, out2)  # type: ignore


# ---------------------------------------------------------------------------
# Delta (incremental) output mode
# ---------------------------------------------------------------------------

def _delta_config(head_type="gru", monotone=False, **kw):
    cfg = _tiny_config(head_type=head_type, out_dim=2, **kw)
    cfg["model"]["kwargs"]["rnn_output_mode"] = "delta"
    cfg["model"]["kwargs"]["rnn_monotone"] = monotone
    return cfg


def _sigma(out, eps=1e-3):
    """Loss convention: sigma = softplus(raw_sigma) + eps."""
    return F.softplus(out[:, :, 1]) + eps


class TestDeltaMode:

    def test_output_shapes_and_variance_accumulation(self):
        """Without forcing, the cumulative sigma must be non-decreasing."""
        cfg = _delta_config()
        forecaster = EncoderForecaster(cfg)
        forecaster.eval()
        X, Y = _tiny_batch()
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        assert out.shape == (4, 10, 2)
        sigma = _sigma(out)
        assert torch.all(sigma[:, 1:] >= sigma[:, :-1] - 1e-6)
        assert forecaster.model.last_step_deltas.shape == (4, 10, 2)

    def test_cumulative_matches_increment_cumsum(self):
        """With nothing given, mu channel = cumsum(delta_mu), sigma channel
        = sqrt(cumsum(delta_var))."""
        cfg = _delta_config()
        forecaster = EncoderForecaster(cfg)
        forecaster.eval()
        X, Y = _tiny_batch()
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        d = forecaster.model.last_step_deltas
        # increment variance: sigma_inc = softplus(raw) + eps = sqrt(d_var)
        d_var = (F.softplus(d[:, :, 1]) + 1e-3) ** 2
        mu_cum, sigma_cum = ProTEncoder.increments_to_cumulative(
            d[:, :, 0], d_var)
        assert torch.allclose(out[:, :, 0], mu_cum, atol=1e-5)
        assert torch.allclose(_sigma(out), sigma_cum, atol=1e-4)

    def test_variance_reset_under_forcing(self):
        """After a given (measured) position the variance restarts from zero:
        at the first hidden position the cumulative sigma equals the single
        increment sigma."""
        cfg = _delta_config()
        forecaster = EncoderForecaster(cfg)
        forecaster.eval()
        forecaster.show_trg_active = True
        forecaster.show_trg_max_idx = 4  # positions 1..4 given (idx 0..3)
        X, Y = _tiny_batch(L_dec=10)
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(
                data_input=X, data_trg=Y, predict_mode=True)
        d = forecaster.model.last_step_deltas
        inc_sigma = F.softplus(d[:, :, 1]) + 1e-3
        cum_sigma = _sigma(out)
        # first hidden position (idx 4): cumulative == single increment
        assert torch.allclose(cum_sigma[:, 4], inc_sigma[:, 4], atol=1e-4)
        # later hidden positions keep accumulating
        assert torch.all(cum_sigma[:, 5:] >= cum_sigma[:, 4:5] - 1e-6)

    def test_monotone_addon(self):
        """rnn_monotone constrains delta_mu >= 0: the path is non-decreasing."""
        cfg = _delta_config(monotone=True)
        forecaster = EncoderForecaster(cfg)
        forecaster.eval()
        X, Y = _tiny_batch()
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        mu = out[:, :, 0]  # y-space here (gaussian_censored)
        assert torch.all(mu[:, 1:] >= mu[:, :-1] - 1e-6)

    def test_increment_targets(self):
        """_increment_targets must match the diff of the (y-space) target."""
        cfg = _delta_config()
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch()
        d = forecaster._increment_targets(Y)
        y = Y[:, :, 2]
        assert torch.allclose(d[:, 0], y[:, 0])
        assert torch.allclose(d[:, 1:], y[:, 1:] - y[:, :-1])

    def test_increment_targets_per_var(self):
        """gru_per_var: increments restart from base 0 at variable boundaries
        and NaN (censored) positions propagate."""
        cfg = _delta_config(head_type="gru_per_var", n_dec_vars=2, dec_var_idx=0)
        forecaster = EncoderForecaster(cfg)
        X, Y = _tiny_batch_pervar(nan_tail=True)  # L=10, blocks of 5
        d = forecaster._increment_targets(Y)
        y = Y[:, :, 2]
        assert torch.allclose(d[:, 0], y[:, 0])
        assert torch.allclose(d[:, 1:4], y[:, 1:4] - y[:, 0:3])
        assert torch.allclose(d[:, 5], y[:, 5])          # var boundary: base 0
        assert torch.isnan(d[:, 4]).all()                # censored position
        assert torch.isnan(d[:, 9]).all()

    @pytest.mark.parametrize("head_type,kw", [
        ("gru", {}),
        ("gru_per_var", {"n_dec_vars": 2, "dec_var_idx": 0}),
    ])
    def test_step_and_backward(self, head_type, kw):
        """The combined cumulative + increment loss is finite and trains."""
        cfg = _delta_config(head_type=head_type, **kw)
        forecaster = EncoderForecaster(cfg)
        forecaster.train()
        X, Y = _tiny_batch(nan_tail=True) if head_type == "gru" \
            else _tiny_batch_pervar(nan_tail=True)
        loss, predicted, _ = forecaster._step((X, Y), stage="train")
        assert torch.isfinite(loss)
        loss.backward()
        assert forecaster.model.rnn_head.weight.grad is not None \
            if head_type == "gru" else \
            forecaster.model.rnn_head[0].weight.grad is not None


class TestDeltaLogSpaceBase:
    """Log-space base fix (delta mode + lognormal loss).

    In log-space z = log(y + shift), so y = 0 maps to z = log(shift), NOT 0.
    Before the fix the rollout state and the increment-target base were
    initialized at z = 0 (y ~ 1), making the first increment of every curve
    ~ -6.9 and unmatchable for a monotone (delta_mu >= 0) head.
    """

    SHIFT = 1e-3

    def _cfg(self, head_type="gru", monotone=True, **kw):
        cfg = _delta_config(head_type=head_type, monotone=monotone, **kw)
        cfg["training"]["loss_fn"] = "lognormal_censored"
        cfg["training"]["loss_log_shift"] = self.SHIFT
        return cfg

    @staticmethod
    def _smooth_batch(B=4, L_dec=10, D_dec=4, nan_tail=False):
        """Monotone targets starting exactly at 0 (like the idealized ISTs)."""
        Y = torch.zeros(B, L_dec, D_dec)
        Y[:, :, 0] = torch.randint(1, 3, (B, L_dec)).float()
        Y[:, :, 1] = torch.arange(1, L_dec + 1).float().unsqueeze(0).expand(B, -1)
        Y[:, :, 2] = 0.05 * (Y[:, :, 1] - 1) * torch.rand(B, 1)  # monotone from 0
        Y[:, :, 3] = 0.0
        if nan_tail:
            Y[:, -2:, 2] = float("nan")
        X = torch.randn(B, 6, 3)
        X[:, :, 0] = torch.randint(1, 5, (B, 6)).float()
        X[:, :, 2] = torch.arange(6).float().unsqueeze(0).expand(B, -1)
        return X, Y

    def test_increment_targets_logspace_base(self):
        """d_0 = z_0 - log(shift); roundtrip cumsum(d) + log(shift) == z."""
        forecaster = EncoderForecaster(self._cfg())
        _, Y = self._smooth_batch()
        d = forecaster._increment_targets(Y)
        y = Y[:, :, 2]
        z = torch.log(y + self.SHIFT)
        z0 = math.log(self.SHIFT)
        assert torch.allclose(d[:, 0], z[:, 0] - z0)
        assert torch.allclose(d[:, 1:], z[:, 1:] - z[:, :-1])
        # roundtrip: z0 + cumsum(increments) reconstructs z
        assert torch.allclose(z0 + d.cumsum(dim=1), z)
        # y_0 = 0 -> z_0 = log(shift) -> d_0 = 0 (matchable by monotone head)
        assert torch.allclose(d[:, 0], torch.zeros_like(d[:, 0]), atol=1e-6)

    def test_increment_targets_per_var_logspace_base(self):
        """gru_per_var: the log(shift) base is used at variable boundaries."""
        cfg = self._cfg(head_type="gru_per_var", n_dec_vars=2, dec_var_idx=0)
        forecaster = EncoderForecaster(cfg)
        _, Y = self._smooth_batch()
        # variable blocks of 5
        Y[:, :, 0] = torch.cat([torch.ones(5), torch.full((5,), 2.0)])
        Y[:, :, 1] = torch.arange(1, 6).float().repeat(2)
        d = forecaster._increment_targets(Y)
        z = torch.log(Y[:, :, 2] + self.SHIFT)
        z0 = math.log(self.SHIFT)
        assert torch.allclose(d[:, 0], z[:, 0] - z0)
        assert torch.allclose(d[:, 5], z[:, 5] - z0)      # boundary: base z0
        assert torch.allclose(d[:, 1:5], z[:, 1:5] - z[:, 0:4])
        assert torch.allclose(d[:, 6:], z[:, 6:] - z[:, 5:-1])

    def test_increment_targets_yspace_unchanged(self):
        """Gaussian (y-space) delta mode keeps the base-0 convention."""
        forecaster = EncoderForecaster(_delta_config())
        _, Y = self._smooth_batch()
        d = forecaster._increment_targets(Y)
        y = Y[:, :, 2]
        assert torch.allclose(d[:, 0], y[:, 0])
        assert torch.allclose(d[:, 1:], y[:, 1:] - y[:, :-1])

    def test_rollout_init_at_log_shift(self):
        """With a zeroed head (d_mu = 0, non-monotone) and nothing given, the
        cumulative mu stays at log(shift) -> y-median exactly 0."""
        forecaster = EncoderForecaster(self._cfg(monotone=False))
        forecaster.eval()
        model = forecaster.model
        assert model.feedback_shift == self.SHIFT
        assert model._delta_z0() == pytest.approx(math.log(self.SHIFT))
        with torch.no_grad():
            model.rnn_head.weight.zero_()
            model.rnn_head.bias.zero_()
        X, Y = self._smooth_batch()
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        assert torch.allclose(out[:, :, 0],
                              torch.full_like(out[:, :, 0],
                                              math.log(self.SHIFT)),
                              atol=1e-6)
        # y-space median = exp(mu) - shift = 0: the init expresses y = 0
        median = torch.exp(out[:, :, 0]) - self.SHIFT
        assert torch.allclose(median, torch.zeros_like(median), atol=1e-7)

    def test_rollout_init_yspace_unchanged(self):
        """Gaussian delta mode: zeroed head -> cumulative mu stays at 0."""
        forecaster = EncoderForecaster(_delta_config(monotone=False))
        forecaster.eval()
        model = forecaster.model
        assert model.feedback_shift is None
        assert model._delta_z0() == 0.0
        with torch.no_grad():
            model.rnn_head.weight.zero_()
            model.rnn_head.bias.zero_()
        X, Y = self._smooth_batch()
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        assert torch.allclose(out[:, :, 0], torch.zeros_like(out[:, :, 0]),
                              atol=1e-6)

    def test_rollout_init_per_var(self):
        """Same init check for the gru_per_var rollout (both variables)."""
        cfg = self._cfg(head_type="gru_per_var", monotone=False,
                        n_dec_vars=2, dec_var_idx=0)
        forecaster = EncoderForecaster(cfg)
        forecaster.eval()
        model = forecaster.model
        for head in model.rnn_head:
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
        X, Y = self._smooth_batch()
        Y[:, :, 0] = torch.cat([torch.ones(5), torch.full((5,), 2.0)])
        Y[:, :, 1] = torch.arange(1, 6).float().repeat(2)
        with torch.no_grad():
            out, _, _, _, _ = forecaster.forward(data_input=X, data_trg=Y)
        assert torch.allclose(out[:, :, 0],
                              torch.full_like(out[:, :, 0],
                                              math.log(self.SHIFT)),
                              atol=1e-6)

    def test_increments_to_cumulative_z0(self):
        """increments_to_cumulative honors the z0 start (default 0)."""
        d_mu = torch.randn(3, 7).abs()
        d_var = torch.rand(3, 7)
        mu0, s0 = ProTEncoder.increments_to_cumulative(d_mu, d_var)
        mu1, s1 = ProTEncoder.increments_to_cumulative(
            d_mu, d_var, z0=math.log(self.SHIFT))
        assert torch.allclose(mu0, d_mu.cumsum(dim=1))
        assert torch.allclose(mu1, math.log(self.SHIFT) + d_mu.cumsum(dim=1))
        assert torch.allclose(s0, s1)

    @pytest.mark.parametrize("head_type,kw", [
        ("gru", {}),
        ("gru_per_var", {"n_dec_vars": 2, "dec_var_idx": 0}),
    ])
    def test_step_and_backward_logspace(self, head_type, kw):
        """Full train step with lognormal_censored + delta on smooth targets:
        finite loss, finite gradients."""
        cfg = self._cfg(head_type=head_type, **kw)
        forecaster = EncoderForecaster(cfg)
        forecaster.train()
        X, Y = self._smooth_batch(nan_tail=True)
        if head_type == "gru_per_var":
            Y[:, :, 0] = torch.cat([torch.ones(5), torch.full((5,), 2.0)])
            Y[:, :, 1] = torch.arange(1, 6).float().repeat(2)
            Y[:, 4, 2] = float("nan")
            Y[:, 9:, 2] = float("nan")
        loss, _, _ = forecaster._step((X, Y), stage="train")
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in forecaster.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestDrawWindow:
    """Random-draw window (show_trg_draw_lower/upper) of the forcing curriculum."""

    def _curriculum(self, **kw):
        from proT.training.curriculum import ForcingCurriculum
        defaults = dict(mode="scheduled", units="fraction", epoch_show_trg=0,
                        max_epochs=10, start=0.9, end=0.0, random_draw=True)
        defaults.update(kw)
        return ForcingCurriculum(**defaults)

    def test_draw_within_window(self):
        cur = self._curriculum(draw_lower=0.7, draw_upper=0.9)
        for _ in range(50):
            b = cur.draw(0)  # warmup-free: epoch 0 -> bound = start = 0.9
            assert 0.7 <= b <= 0.9

    def test_default_window_unchanged(self):
        cur = self._curriculum()
        for _ in range(50):
            b = cur.draw(0)
            assert 0.0 <= b <= 0.9

    def test_window_collapses_below_bound(self):
        """Once the annealed bound falls below draw_lower, the draw returns
        the bound itself (no invalid range)."""
        cur = self._curriculum(draw_lower=0.7, draw_upper=0.9)
        assert cur.draw(9) == 0.0  # end of schedule: bound 0, no draw

    def test_window_clamped_to_annealed_bound(self):
        cur = self._curriculum(draw_lower=0.7, draw_upper=0.9, max_epochs=11)
        # mid-schedule bound ~0.45 < draw_lower: window collapses to the bound
        for _ in range(20):
            b = cur.draw(5)
            assert 0.0 <= b <= 0.5

    def test_draw_lower_anneals(self):
        """The lower window end anneals draw_lower -> draw_lower_end on the
        bound's schedule: [0.85, 0.9] at start, widening down to 0 at end."""
        cur = self._curriculum(draw_lower=0.85, draw_lower_end=0.0, max_epochs=11)
        assert cur._draw_lower_at(0) == 0.85
        assert abs(cur._draw_lower_at(5) - 0.425) < 1e-6  # linear midpoint
        assert cur._draw_lower_at(10) == 0.0
        lows, highs = [], []
        for _ in range(50):
            b0, b5, b10 = cur.draw(0), cur.draw(5), cur.draw(10)
            lows.append((b0, b5, b10))
        assert all(0.85 <= b <= 0.9 for b, _, _ in lows)   # start: [0.85, 0.9]
        assert all(0.425 <= b <= 0.45 for _, b, _ in lows)  # mid: [0.425, 0.45]
        assert all(b == 0.0 for _, _, b in lows)            # end: collapsed to 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
