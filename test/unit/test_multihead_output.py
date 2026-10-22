"""Unit tests for the multi-head output pooling of the ProTEncoder scalar head
(n_output_heads above 1: one pool query + dedicated MLP per scalar target)
and for the var_reg_weighted option of the log-normal NLL losses.
"""

import math
import pytest
import torch

from proT.core.model_encoder import ProTEncoder
from proT.training.losses import LogNormalNLL, CensoredLogNormalNLL


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def _tiny_scalar_config(output_dim=2, out_dim=2, d_model_enc=16, e_layers=2,
                        n_heads=2, d_qk=8, d_ff=32, n_output_heads=1,
                        n_head_outputs=None, head_hidden_dim=None,
                        head_var_embed=False):
    """Tiny scalar-head config for fast unit tests."""
    return {
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
                 "kwargs": {"num_embeddings": 3, "embedding_dim": d_model_enc}},
            ],
        },
        "comps_embed_dec": "summation",
        "enc_attention_type": "ScaledDotProduct",
        "enc_mask_type": "Uniform",
        "n_heads": n_heads,
        "enc_causal_mask": False,
        "enc_group_mask": False,
        "dropout_emb": 0.0,
        "dropout_data": 0.0,
        "dropout_attn_out": 0.0,
        "dropout_ff": 0.0,
        "enc_dropout_qkv": 0.0,
        "enc_attention_dropout": 0.0,
        "e_layers": e_layers,
        "activation": "gelu",
        "norm": "layer",
        "use_final_norm": True,
        "device": "cpu",
        "out_dim": out_dim,
        "d_ff": d_ff,
        "d_model_enc": d_model_enc,
        "d_model_dec": d_model_enc,
        "d_qk": d_qk,
        "head_type": "scalar",
        "output_dim": output_dim,
        "n_output_heads": n_output_heads,
        "n_head_outputs": n_head_outputs,
        "head_hidden_dim": head_hidden_dim,
        "head_var_embed": head_var_embed,
    }


def _tiny_batch(B=4, L_enc=6, D_enc=3):
    X = torch.randn(B, L_enc, D_enc)
    X[:, :, 0] = torch.randint(1, 5, (B, L_enc)).float()  # variable ids
    X[:, :, 2] = torch.arange(L_enc).float().unsqueeze(0).expand(B, -1)  # order
    return X


# -------------------------------------------------------------------------
# multi-head output pooling
# -------------------------------------------------------------------------

class TestMultiHeadOutput:

    def test_single_head_backward_compat(self):
        """Default n_output_heads=1 reproduces the legacy scalar head exactly:
        same output shape and same state_dict keys (old checkpoints load)."""
        model = ProTEncoder(**_tiny_scalar_config())
        X = _tiny_batch()
        out, _, (_, _, pool_att), _, _ = model(X, None)
        assert out.shape == (4, 2, 2)  # (B, output_dim, out_dim)
        assert model.pool_query.shape == (1, 1, 16)
        assert "scalar_head.weight" in dict(model.named_parameters())
        assert not hasattr(model, "pool_query_embed")

    def test_multi_head_shapes_and_backward(self):
        """n_output_heads=2: one context + one MLP per target, stacked output."""
        model = ProTEncoder(**_tiny_scalar_config(
            n_output_heads=2, head_hidden_dim=32, head_var_embed=True))
        X = _tiny_batch()
        out, _, (_, _, pool_att), _, _ = model(X, None)
        assert out.shape == (4, 2, 2)  # (B, n_output_heads, n_head_outputs)
        assert model.pool_query.shape == (1, 2, 16)
        assert model.pool_query_embed.weight.shape == (2, 16)
        loss = out.square().mean()
        loss.backward()
        # every head MLP and the query embedding must receive gradients
        for h in range(2):
            for par in model.scalar_head[h].parameters():
                assert par.grad is not None and torch.isfinite(par.grad).all()
        assert model.pool_query_embed.weight.grad is not None
        assert torch.isfinite(model.pool_query_embed.weight.grad).all()

    def test_multi_head_linear_variant(self):
        """head_hidden_dim=None: per-head single Linear (no MLP hidden layer)."""
        model = ProTEncoder(**_tiny_scalar_config(
            n_output_heads=2, head_hidden_dim=None, head_var_embed=False))
        out, _, _, _, _ = model(_tiny_batch(), None)
        assert out.shape == (4, 2, 2)
        assert isinstance(model.scalar_head[0], torch.nn.Linear)

    def test_multi_head_custom_n_head_outputs(self):
        """n_head_outputs sets the per-head output width."""
        model = ProTEncoder(**_tiny_scalar_config(
            n_output_heads=2, n_head_outputs=4, head_hidden_dim=8))
        out, _, _, _, _ = model(_tiny_batch(), None)
        assert out.shape == (4, 2, 4)

    def test_multi_head_must_match_output_dim(self):
        with pytest.raises(AssertionError, match="must equal output_dim"):
            ProTEncoder(**_tiny_scalar_config(output_dim=2, n_output_heads=3))

    def test_multi_head_rejects_gru(self):
        cfg = _tiny_scalar_config(n_output_heads=2)
        cfg["head_type"] = "gru"
        with pytest.raises(AssertionError, match="only supported for the scalar head"):
            ProTEncoder(**cfg)

    def test_heads_are_functionally_independent(self):
        """Zeroing head 1s MLP must leave head 0s output unchanged."""
        model = ProTEncoder(**_tiny_scalar_config(
            n_output_heads=2, head_hidden_dim=16))
        model.eval()
        X = _tiny_batch()
        with torch.no_grad():
            out_ref = model(X, None)[0]
            for par in model.scalar_head[1].parameters():
                par.zero_()
            out_pert = model(X, None)[0]
        assert torch.allclose(out_ref[:, 0], out_pert[:, 0])
        assert not torch.allclose(out_ref[:, 1], out_pert[:, 1])


# -------------------------------------------------------------------------
# var_reg_weighted
# -------------------------------------------------------------------------

def _lognormal_inputs(B=8, D=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(B, D, generator=g)
    raw_sigma = torch.randn(B, D, generator=g)
    target = torch.rand(B, D, generator=g) * 0.05  # positive slopes
    return mu, raw_sigma, target


class TestVarRegWeighted:

    def test_legacy_mode_unchanged(self):
        """var_reg_weighted=False reproduces the legacy additive penalty."""
        mu, raw_sigma, target = _lognormal_inputs()
        loss_fn = LogNormalNLL(var_reg=0.1, var_reg_form="sigma",
                               var_reg_weighted=False)
        loss = loss_fn(mu, raw_sigma, target)
        sigma = torch.nn.functional.softplus(raw_sigma) + loss_fn.eps
        log_y = torch.log(target + loss_fn.shift)
        nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma)             + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
        expected = nll.mean() + 0.1 * (sigma ** 2).mean()
        assert torch.allclose(loss, expected)

    def test_weighted_mode_matches_manual(self):
        """var_reg_weighted=True: penalty inside the (sample-)weighted mean."""
        mu, raw_sigma, target = _lognormal_inputs()
        sw = torch.rand(8, 1, generator=torch.Generator().manual_seed(1)) + 0.5
        loss_fn = LogNormalNLL(var_reg=0.1, var_reg_form="sigma",
                               var_reg_weighted=True)
        loss = loss_fn(mu, raw_sigma, target, sample_weight=sw)
        sigma = torch.nn.functional.softplus(raw_sigma) + loss_fn.eps
        log_y = torch.log(target + loss_fn.shift)
        nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma)             + 0.5 * ((log_y - mu) / sigma) ** 2 + log_y
        nll = nll + 0.1 * sigma ** 2
        sw_full = sw.expand_as(nll)  # (B, 1) broadcast over the D targets
        expected = (nll * sw_full).sum() / sw_full.sum().clamp_min(1e-8)
        assert torch.allclose(loss, expected)

    def test_weighted_mode_equals_legacy_under_uniform_weights(self):
        """With uniform weights the weighted penalty reduces to
        lambda * mean(pen), i.e. exactly the legacy mode; with non-uniform
        weights the two modes differ (per-sample penalty balance follows
        the sample weights in weighted mode, not in legacy mode)."""
        mu, raw_sigma, target = _lognormal_inputs()
        kw = dict(var_reg=0.1, var_reg_form="sigma")
        weighted = LogNormalNLL(var_reg_weighted=True, **kw)
        legacy = LogNormalNLL(var_reg_weighted=False, **kw)
        # no sample weights at all: uniform case
        assert torch.allclose(weighted(mu, raw_sigma, target),
                              legacy(mu, raw_sigma, target))
        # non-uniform weights: modes diverge
        sw = torch.rand(8, 1, generator=torch.Generator().manual_seed(2)) + 0.5
        assert not torch.allclose(weighted(mu, raw_sigma, target, sample_weight=sw),
                                  legacy(mu, raw_sigma, target, sample_weight=sw))

    def test_weighted_mode_backward(self):
        mu, raw_sigma, target = _lognormal_inputs()
        mu.requires_grad_()
        raw_sigma.requires_grad_()
        loss_fn = LogNormalNLL(var_reg=0.1, var_reg_form="sigma",
                               nll_beta=0.5, var_reg_weighted=True)
        loss = loss_fn(mu, raw_sigma, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(raw_sigma.grad).all()

    def test_censored_lognormal_weighted(self):
        """Same option on CensoredLogNormalNLL, with censored positions."""
        mu, raw_sigma, target = _lognormal_inputs()
        target[:, -1] = float("nan")  # censor one position per sample
        loss_fn = CensoredLogNormalNLL(threshold=0.06, var_reg=0.1,
                                       var_reg_form="sigma",
                                       var_reg_weighted=True)
        loss = loss_fn(mu, raw_sigma, target)
        assert torch.isfinite(loss)
        loss_legacy = CensoredLogNormalNLL(threshold=0.06, var_reg=0.1,
                                           var_reg_form="sigma",
                                           var_reg_weighted=False)
        assert torch.isfinite(loss_legacy(mu, raw_sigma, target))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
