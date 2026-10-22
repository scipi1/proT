"""
Unit tests for the encoder benchmark (proT/baseline/baseline_models.py).

Checks: output shape, identical embedding/head across encoders, padding and
token-permutation invariance, causality, no time2vec, Optuna samplers touch
only encoder keys, Lightning step on scalar targets.
"""

import copy

import pytest
import torch

from omegaconf import OmegaConf

from proT.baseline.baseline_models import BenchmarkModel, BENCHMARK_ENCODERS
from proT.baseline.baseline_pl_modules import BenchmarkForecaster

D = 16
ENCODERS = {
    "proT_bench": {"e_layers": 1, "n_heads": 2, "d_ff": 32, "d_qk": 8, "dropout": 0.0},
    "GRU": {"d_hidden": 12, "n_layers": 2, "dropout": 0.0},
    "LSTM": {"d_hidden": 12, "n_layers": 1, "dropout": 0.0},
    "TCN": {"channels": [8, 8], "kernel": 3, "dropout": 0.0},
    "MLP": {"hidden": [20, 20], "dropout": 0.0},
}
HEAD = {"d_head": 8, "hidden": [8, 8], "n_out": 2, "dropout": 0.0}

# input features: 0 value, 1 order, 2 variable
DS_EMBED = {
    "setting": {"sparse_grad": False},
    "modules": [
        {"idx": 2, "embed": "nn_embedding", "label": "variable",
         "kwargs": {"num_embeddings": 5, "embedding_dim": D, "padding_idx": 0}},
        {"idx": 0, "embed": "linear", "label": "value",
         "kwargs": {"input_dim": 1, "embedding_dim": D}},
        {"idx": 0, "embed": "mask", "label": "value_missing", "kwargs": {}},
        {"idx": 1, "embed": "pass", "label": "order", "kwargs": {}},
    ],
}


def make_model(enc, ds_embed=DS_EMBED):
    torch.manual_seed(0)
    m = BenchmarkModel(d_model=D, comps_embed="summation", ds_embed_in=copy.deepcopy(ds_embed),
                       encoder={"type": enc, "kwargs": ENCODERS[enc]}, head=HEAD)
    return m.cpu().eval()


def make_input(B=3, L=7, n_pad=2):
    torch.manual_seed(1)
    X = torch.zeros(B, L, 3)
    X[:, :, 0] = torch.randn(B, L)
    X[:, :, 1] = torch.arange(1, L + 1).float()
    X[:, :, 2] = torch.randint(1, 5, (B, L)).float()
    if n_pad:
        X[:, L - n_pad:, :] = float("nan")      # padded tail
    return X


@pytest.mark.parametrize("enc", BENCHMARK_ENCODERS)
def test_output_shape(enc):
    out = make_model(enc)(make_input())
    assert out.shape == (3, HEAD["n_out"])
    assert torch.isfinite(out).all()


def test_shared_embedding_and_head_identical():
    counts = {e: make_model(e).param_counts() for e in BENCHMARK_ENCODERS}
    assert len({c["embedding"] for c in counts.values()}) == 1
    assert len({c["head"] for c in counts.values()}) == 1


@pytest.mark.parametrize("enc", BENCHMARK_ENCODERS)
def test_padding_and_shuffle_invariance(enc):
    """Extra padding and a token permutation (same order feature) give the same output."""
    m = make_model(enc)
    X = make_input(n_pad=2)
    X_more_pad = torch.cat([X, torch.full((3, 3, 3), float("nan"))], dim=1)
    perm = torch.randperm(X.size(1))
    with torch.no_grad():
        ref = m(X)
        assert torch.allclose(ref, m(X_more_pad), atol=1e-5)
        assert torch.allclose(ref, m(X[:, perm]), atol=1e-5)


@pytest.mark.parametrize("enc", BENCHMARK_ENCODERS)
def test_order_feature_drives_sequence(enc):
    """Order-aware encoders change output when the chain order is reversed."""
    m = make_model(enc)
    X = make_input(n_pad=0)
    X_rev = X.clone()
    X_rev[:, :, 1] = X[:, :, 1].flip(-1)
    with torch.no_grad():
        same = torch.allclose(m(X), m(X_rev), atol=1e-6)
    assert same == (enc == "MLP")                # MLP is order-agnostic


def test_transformer_token_causal():
    """Transformer token i output is unchanged when a later token j > i changes."""
    m = make_model("proT_bench")
    X = make_input(n_pad=0)
    X2 = X.clone()
    X2[:, 5, 0] += 10.0

    def enc_out(Xi):
        Xs, valid, _ = m.sort_chain(Xi)
        x = m.embedding(Xs)
        B, L, _ = x.shape
        pos = torch.arange(L).float().view(1, L, 1).expand(B, -1, -1)
        miss = (~valid).unsqueeze(-1)
        return m.encoder.encoder(X=x, mask_miss_k=miss, mask_miss_q=miss,
                                 enc_input_pos=pos, causal_mask=True)[0]

    with torch.no_grad():
        a, b = enc_out(X), enc_out(X2)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-6)


def test_time2vec_rejected():
    ds = copy.deepcopy(DS_EMBED)
    ds["modules"].append({"idx": 0, "embed": "time2vec", "label": "time",
                          "kwargs": {"input_dim": 1, "embed_dim": D}})
    with pytest.raises(ValueError):
        make_model("GRU", ds_embed=ds)


def test_optuna_samplers_only_encoder_keys():
    optuna = pytest.importorskip("optuna")
    from proT.euler_optuna.cli import sample_params_for_optuna, ENC
    for enc in BENCHMARK_ENCODERS:
        cfg = OmegaConf.create({"model": {"model_object": "BENCH",
                                          "kwargs": {"encoder": {"type": enc}}}})
        trial = optuna.trial.FixedTrial({
            "dropout": 0.1, "lr": 1e-3, "d_hidden": 64, "n_layers": 2,
            "tcn_arch": "tcn_s-32-32-64", "kernel": 3, "mlp_arch": "mlp_s-128x2",
            "e_layers": 1, "n_heads": 1, "d_ff": 200, "d_qk": 100})
        params = sample_params_for_optuna(trial, cfg)
        assert params
        assert all(k.startswith(ENC + ".") or k == "training.lr" for k in params)


def test_forecaster_step_scalar_target():
    config = {
        "model": {"model_object": "BENCH",
                  "kwargs": {"d_model": D, "comps_embed": "summation",
                             "ds_embed_in": copy.deepcopy(DS_EMBED),
                             "encoder": {"type": "GRU", "kwargs": ENCODERS["GRU"]},
                             "head": HEAD}},
        "data": {"val_idx": 2},
        "training": {"loss_fn": "mse", "lr": 1e-3},
    }
    fc = BenchmarkForecaster(config).cpu()
    fc.log = lambda *a, **k: None
    X = make_input()
    Y = torch.zeros(3, 2, 3)
    Y[:, :, 2] = torch.randn(3, 2)
    Y[0, 1, 2] = float("nan")                    # missing scalar is ignored
    loss, pred, trg = fc._step((X, Y), stage="train")
    assert pred.shape == (3, 2) and torch.isfinite(loss)
    loss.backward()
