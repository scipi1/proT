"""Unit tests for the GRU head with probabilistic (lognormal) losses.

Covers the three changes enabling GRU-head NLL training:
1. EncoderForecaster._step feeds the FULL target sequence to the loss for
   GRU heads (it previously sliced the first `output_dim` positions, which is
   only correct for the scalar head).
2. Autoregressive feedback is in y-space: for log-space losses the model
   feeds back exp(mu) - shift instead of raw mu (feedback_shift mechanism).
3. h0 is a plain linear projection of the pooled context (no tanh).
"""

import pytest
import torch

from proT.core.model_encoder import ProTEncoder
from proT.training.forecasters.encoder_forecaster import EncoderForecaster


def _tiny_gru_kwargs(head_type="gru", out_dim=2, d_model=16, d_hidden=8,
                     e_layers=1, n_heads=2, d_qk=8, d_ff=32):
    """Tiny GRU-head model kwargs for fast unit tests."""
    kw = {
        "model": "proT_enc",
        "ds_embed_enc": {
            "setting": {"sparse_grad": False},
            "modules": [
                {"idx": 0, "embed": "nn_embedding", "label": "variable",
                 "kwargs": {"num_embeddings": 5, "embedding_dim": d_model}},
                {"idx": 1, "embed": "linear", "label": "value",
                 "kwargs": {"input_dim": 1, "embedding_dim": d_model}},
                {"idx": 1, "embed": "mask", "label": "value_missing", "kwargs": {}},
                {"idx": 2, "embed": "pass", "label": "order", "kwargs": {}},
            ],
        },
        "comps_embed_enc": "summation",
        "ds_embed_dec": {
            "setting": {"sparse_grad": False},
            "modules": [
                {"idx": 0, "embed": "nn_embedding", "label": "variable",
                 "kwargs": {"num_embeddings": 3, "embedding_dim": d_model}},
                {"idx": 2, "embed": "nn_embedding", "label": "position",
                 "kwargs": {"num_embeddings": 10, "embedding_dim": d_model}},
            ],
        },
        "comps_embed_dec": "summation",
        "enc_attention_type": "ScaledDotProduct",
        "enc_mask_type": "Uniform",
        "n_heads": n_heads,
        "enc_causal_mask": False,
        "enc_group_mask": False,
        "dropout_emb": 0.0, "dropout_data": 0.0, "dropout_attn_out": 0.0,
        "dropout_ff": 0.0, "enc_dropout_qkv": 0.0, "enc_attention_dropout": 0.0,
        "e_layers": e_layers,
        "activation": "gelu",
        "norm": "layer",
        "use_final_norm": True,
        "device": "cpu",
        "out_dim": out_dim,
        "d_ff": d_ff,
        "d_model_enc": d_model,
        "d_model_dec": d_model,
        "d_qk": d_qk,
        "head_type": head_type,
        "output_dim": None,
        "d_hidden": d_hidden,
        "rnn_n_layers": 1,
        "rnn_dropout": 0.0,
        "dec_val_idx": 1,
        "dec_pos_idx": 2,
    }
    if head_type == "gru_per_var":
        kw["n_dec_vars"] = 2
        kw["dec_var_idx"] = 0
    return kw


def _tiny_batch(B=3, L_enc=6, L_dec=5):
    """X: (B, L_enc, 3) [variable, value, order]; Y: (B, L_dec, 3)
    [variable, value, position]."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(B, L_enc, 3, generator=g)
    X[:, :, 0] = torch.randint(1, 5, (B, L_enc), generator=g).float()
    X[:, :, 2] = torch.arange(L_enc).float().unsqueeze(0).expand(B, -1)
    Y = torch.zeros(B, L_dec, 3)
    Y[:, :, 0] = 1  # variable id
    Y[:, :, 1] = torch.rand(B, L_dec, generator=g) + 0.1  # positive values
    Y[:, :, 2] = torch.arange(1, L_dec + 1).float().unsqueeze(0).expand(B, -1)
    return X, Y


def _tiny_forecaster(head_type="gru", loss_fn="lognormal"):
    """EncoderForecaster with a GRU head, no trainer attached."""
    config = {
        "model": {"kwargs": _tiny_gru_kwargs(head_type=head_type)},
        "training": {
            "loss_fn": loss_fn,
            "loss_log_shift": 1e-3,
            "loss_var_reg": 0.05,
            "loss_var_reg_form": "sigma",
            "loss_var_reg_weighted": True,
            "loss_sw_mean_only": True,
            "lr": 1e-3,
            "max_epochs": 10,
            "entropy_regularizer": False,
        },
        "data": {"val_idx": 1, "pos_idx": 2},
    }
    fc = EncoderForecaster(config)
    # self.log requires a registered trainer; stub it out for the unit test
    fc.log = lambda *a, **k: None
    return fc


class TestGRUFullSequenceLoss:
    """Gap 1: the loss must consume ALL positions for GRU heads."""

    @pytest.mark.parametrize("head_type", ["gru", "gru_per_var"])
    def test_late_positions_enter_the_loss(self, head_type):
        """Changing a target AFTER the first output_dim positions must change
        the loss (the old code sliced trg_val[:, :output_dim])."""
        fc = _tiny_forecaster(head_type)
        X, Y = _tiny_batch()
        loss1, _, _ = fc._step((X, Y), stage="train")

        Y2 = Y.clone()
        Y2[:, 4, 1] = Y2[:, 4, 1] + 1.0  # position index 4 (> output_dim=2)
        loss2, _, _ = fc._step((X, Y2), stage="train")

        assert not torch.allclose(loss1, loss2), \
            "loss is insensitive to late positions: targets are still sliced"

    def test_nan_positions_are_excluded(self):
        """NaN targets (padded tail) are excluded from the plain lognormal
        loss: making a valid position NaN changes the loss, which stays
        finite."""
        fc = _tiny_forecaster("gru")
        X, Y = _tiny_batch()
        loss1, _, _ = fc._step((X, Y), stage="train")
        Y2 = Y.clone()
        Y2[:, 3:, 1] = float("nan")
        loss2, _, _ = fc._step((X, Y2), stage="train")
        assert torch.isfinite(loss2)
        assert not torch.allclose(loss1, loss2)


class TestFeedbackValue:
    """Gap 2: autoregressive feedback in y-space for log-space losses."""

    def test_identity_without_shift(self):
        model = ProTEncoder(**_tiny_gru_kwargs())
        mu = torch.tensor([0.5, -0.3])
        assert torch.equal(model._feedback_value(mu), mu)

    def test_logspace_with_shift(self):
        model = ProTEncoder(**_tiny_gru_kwargs())
        model.feedback_shift = 1e-3
        mu = torch.tensor([0.5, -0.3])
        assert torch.allclose(model._feedback_value(mu), torch.exp(mu) - 1e-3)

    def test_forecaster_sets_feedback_shift(self):
        """Log-space loss -> feedback_shift set; the AR rollout then differs
        from identity feedback (the path is actually wired)."""
        fc = _tiny_forecaster("gru")
        assert fc.model.feedback_shift == pytest.approx(1e-3)
        X, Y = _tiny_batch()
        out_shifted, _, _, _, _ = fc.model.forward(X, Y)
        fc.model.feedback_shift = None
        out_identity, _, _, _, _ = fc.model.forward(X, Y)
        assert not torch.allclose(out_shifted, out_identity), \
            "feedback_shift has no effect: feedback path not wired"

    def test_forecaster_unsets_feedback_shift_for_mse(self):
        fc = _tiny_forecaster("gru", loss_fn="mse")
        assert fc.model.feedback_shift is None


class TestH0Projection:
    """Gap 3: h0 is a plain linear projection (tanh removed)."""

    @pytest.mark.parametrize("head_type", ["gru", "gru_per_var"])
    def test_forward_shapes(self, head_type):
        model = ProTEncoder(**_tiny_gru_kwargs(head_type=head_type))
        X, Y = _tiny_batch()
        out, _, _, _, _ = model.forward(X, Y)
        assert out.shape == (3, 5, 2)
        assert torch.isfinite(out).all()

    def test_h0_is_unbounded_linear(self):
        """Captured h0_projection output can exceed the tanh range [-1, 1]
        and the rollout still runs (the GRU gates absorb the magnitude)."""
        model = ProTEncoder(**_tiny_gru_kwargs())
        with torch.no_grad():
            model.h0_projection.weight.mul_(100.0)
        captured = {}
        hook = model.h0_projection.register_forward_hook(
            lambda m, i, o: captured.__setitem__("h0", o.detach()))
        X, Y = _tiny_batch()
        out, _, _, _, _ = model.forward(X, Y)
        hook.remove()
        assert captured["h0"].abs().max() > 1.0
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
