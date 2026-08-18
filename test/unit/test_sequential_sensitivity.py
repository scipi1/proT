"""
Unit tests for proT.proj_specific.GSA.sequential_sensitivity.

Uses a synthetic weighted-sum "model" with known sensitivities:
    y[b] = sum_l w[b, var[b, l]] * nan_to_num(val[b, l])
so masking variable v changes the prediction by w[b, v] * val. The greedy
ranking must recover the variables in decreasing |w * val| order.
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from proT.evaluation.predict import predict_test_from_ckpt
from proT.evaluation.predictors import PredictionResult
from proT.proj_specific.GSA.sequential_sensitivity import (
    DEFAULT_SIGMA_EPS,
    _apply_mask,
    _baseline_scale,
    _build_dataframe,
    _candidate_mask,
    _collect_candidate_values,
    _error_pct,
    _get_error_fn,
    _is_probabilistic_config,
    _kl_per_sample,
    _masked_mean,
    _mse_per_sample,
    _resolve_error_metric,
    _to_bpc,
    _w2_per_sample,
    get_sequential_sensitivity_df,
)

CTRL_IDX = 0
VAL_IDX = 1
VAR_IDS = [1, 2, 3, 4, 5]
WEIGHTS = {1: 0.1, 2: 0.5, 3: 1.0, 4: 0.2, 5: 0.05}
# Expected greedy order (all values positive): 3, 2, 4, 1, 5
EXPECTED_ORDER = [3, 2, 4, 1, 5]


# ---------------------------------------------------------------------------
# Synthetic model fixtures
# ---------------------------------------------------------------------------

def _make_data(B=6, L=5, seed=0):
    """X: (B, L, 2); channel 0 = variable id, channel 1 = value (fixed to 1).

    With val = 1 everywhere, masking variable v changes the prediction by
    exactly w[v], so the greedy ranking must follow the weight order.
    """
    ctrl = torch.tensor(VAR_IDS, dtype=torch.float32).repeat(B, 1)
    val = torch.ones(B, L)
    X = torch.stack([ctrl, val], dim=-1)
    Y = torch.zeros(B, 1, 2)
    return X, Y


def _weighted_sum(X, weights):
    """y[b] = sum_l w[var] * nan_to_num(val); weights: dict or (B, V) tensor."""
    ctrl = X[:, :, CTRL_IDX].long()
    val = torch.nan_to_num(X[:, :, VAL_IDX], nan=0.0)
    if isinstance(weights, dict):
        w_table = torch.zeros(max(weights) + 1)
        for v, w in weights.items():
            w_table[v] = w
        w = w_table[ctrl]  # (B, L)
    else:
        w = torch.gather(weights, 1, ctrl)  # (B, L) per-sample weights
    return (w * val).sum(dim=1, keepdim=True)  # (B, 1)


def _make_mock_predict_fn(X, Y, weights):
    """Mock with the predict_test_from_ckpt signature (reference backend)."""
    def predict_fn(config=None, datadir_path=None, checkpoint_path=None,
                   external_dataset=None, dataset_label="test", cluster=False,
                   input_conditioning_fn=None):
        X_in = X.clone()
        if input_conditioning_fn is not None:
            X_in = input_conditioning_fn(X_in)
        y = _weighted_sum(X_in, weights)
        return PredictionResult(
            inputs=X_in.numpy(),
            outputs=y.numpy(),
            targets=Y.numpy(),
            attention_weights=None,
            metadata={},
        )
    return predict_fn


class _MockDataModule:
    def __init__(self, X, Y):
        ds = TensorDataset(X, Y)
        self.train_ds = ds
        self.test_ds = ds
        self.all_ds = ds

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


class _MockPredictor:
    """Mock predictor with the BasePredictor interface (optimized backend)."""

    def __init__(self, X, Y, weights):
        self.device = torch.device("cpu")
        self._X, self._Y, self._weights = X, Y, weights

    def create_data_module(self, external_dataset=None, cluster=False):
        return _MockDataModule(self._X, self._Y)

    def _forward(self, X, trg, **kwargs):
        return _weighted_sum(X, self._weights)

    def _process_forward_output(self, output):
        return {"forecast": output, "attention_weights": None}


def _make_config(loss_fn="mse", out_dim=1, batch_size=4):
    return {
        "training": {"seed": 42, "batch_size": batch_size, "loss_fn": loss_fn},
        "model": {"kwargs": {"out_dim": out_dim}},
        "data": {"features": {"X": {"variable": CTRL_IDX, "value": VAL_IDX}}},
    }


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_to_bpc_shapes(self):
        assert _to_bpc(torch.zeros(4)).shape == (4, 1, 1)
        assert _to_bpc(torch.zeros(4, 7)).shape == (4, 7, 1)
        assert _to_bpc(torch.zeros(4, 7, 2)).shape == (4, 7, 2)

    def test_mse_rmse_mae(self):
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        masked = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        valid = torch.ones(2, 2, dtype=torch.bool)
        mse = _mse_per_sample(_to_bpc(base), _to_bpc(masked), valid)
        assert torch.allclose(mse, torch.tensor([0.5, (4.0 + 9.0) / 2]))
        rmse_fn, is_sq = _get_error_fn("rmse")
        assert not is_sq
        assert torch.allclose(rmse_fn(base, masked), torch.sqrt(mse))
        mae_fn, _ = _get_error_fn("mae")
        assert torch.allclose(mae_fn(base, masked), torch.tensor([0.5, 2.5]))

    def test_mse_nan_positions_excluded(self):
        base = torch.tensor([[1.0, float("nan")]])
        masked = torch.tensor([[0.0, 5.0]])
        err_fn, is_sq = _get_error_fn("mse")
        assert is_sq
        err = err_fn(base, masked)
        assert torch.allclose(err, torch.tensor([1.0]))  # only position 0 counts

    def test_masked_mean_all_invalid(self):
        x = torch.ones(2, 3)
        valid = torch.zeros(2, 3, dtype=torch.bool)
        out = _masked_mean(x, valid)
        assert torch.isnan(out).all()

    def test_w2_closed_form(self):
        # N(0, 1) vs N(1, 2): W2^2 = (0-1)^2 + (1-2)^2 = 2
        def raw_sigma(sigma):
            return math.log(math.exp(sigma - DEFAULT_SIGMA_EPS) - 1.0)
        base = torch.tensor([[[0.0, raw_sigma(1.0)]]])       # (1, 1, 2)
        masked = torch.tensor([[[1.0, raw_sigma(2.0)]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        w2 = _w2_per_sample(base, masked, valid)
        assert torch.allclose(w2, torch.tensor([math.sqrt(2.0)]), atol=1e-4)

    def test_w2_sigma_change_only(self):
        # Same mu, different sigma: W2 = |s1 - s2| (accounts for variance)
        def raw_sigma(sigma):
            return math.log(math.exp(sigma - DEFAULT_SIGMA_EPS) - 1.0)
        base = torch.tensor([[[0.0, raw_sigma(1.0)]]])
        masked = torch.tensor([[[0.0, raw_sigma(3.0)]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        w2 = _w2_per_sample(base, masked, valid)
        assert torch.allclose(w2, torch.tensor([2.0]), atol=1e-4)

    def test_kl_closed_form(self):
        # KL(N(0,1) || N(1,2)) = log(2) + (1 + 1)/(2*4) - 1/2
        def raw_sigma(sigma):
            return math.log(math.exp(sigma - DEFAULT_SIGMA_EPS) - 1.0)
        base = torch.tensor([[[0.0, raw_sigma(1.0)]]])
        masked = torch.tensor([[[1.0, raw_sigma(2.0)]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        kl = _kl_per_sample(base, masked, valid)
        expected = math.log(2.0) + 2.0 / 8.0 - 0.5
        assert torch.allclose(kl, torch.tensor([expected]), atol=1e-4)

    def test_baseline_scale_and_pct(self):
        base = torch.tensor([[3.0, 4.0]])  # mean sq = 12.5, rms = sqrt(12.5)
        scale_sq = _baseline_scale(base, is_squared=True)
        scale_lin = _baseline_scale(base, is_squared=False)
        assert torch.allclose(scale_sq, torch.tensor([12.5]))
        assert torch.allclose(scale_lin, torch.tensor([math.sqrt(12.5)]))
        pct = _error_pct(torch.tensor([1.25]), scale_sq)
        assert torch.allclose(pct, torch.tensor([10.0]))
        # zero baseline -> NaN pct
        pct_zero = _error_pct(torch.tensor([1.0]), torch.tensor([0.0]))
        assert torch.isnan(pct_zero).all()

    def test_resolve_error_metric(self):
        assert _resolve_error_metric("auto", probabilistic=False) == "mse"
        assert _resolve_error_metric("auto", probabilistic=True) == "w2"
        assert _resolve_error_metric("kl", probabilistic=True) == "kl"
        with pytest.raises(ValueError):
            _resolve_error_metric("w2", probabilistic=False)
        with pytest.raises(ValueError):
            _resolve_error_metric("nope", probabilistic=False)

    def test_is_probabilistic_config(self):
        assert not _is_probabilistic_config(_make_config("mse", 1))
        assert _is_probabilistic_config(_make_config("gaussian", 2))
        assert _is_probabilistic_config(_make_config("gaussian_censored", 2))
        assert not _is_probabilistic_config(_make_config("gaussian", 1))


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_candidate_mask(self):
        X_ctrl = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
        values = torch.tensor([1.0, 2.0])
        mask = _candidate_mask(X_ctrl, values)  # (2, 2, 3)
        assert mask.shape == (2, 2, 3)
        assert mask[0, 0].tolist() == [True, False, True]
        assert mask[0, 1].tolist() == [False, True, False]
        assert mask[1, 0].tolist() == [False, True, False]

    def test_apply_mask(self):
        X = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        mask = torch.tensor([[True, False, True]])
        out = _apply_mask(X, mask, intervention_feat_idx=2)
        assert torch.isnan(out[0, 0, 2]) and torch.isnan(out[0, 2, 2])
        assert not torch.isnan(out[0, 1, 2])
        assert not torch.isnan(X).any()  # original untouched

    def test_collect_candidate_values(self):
        inputs = np.array([[[0.0, 9.0], [1.0, 9.0], [2.0, 9.0], [np.nan, 9.0]]])
        values = _collect_candidate_values(inputs, control_feat_idx=0)
        assert values.tolist() == [1.0, 2.0]  # 0 (padding) and NaN excluded
        values = _collect_candidate_values(inputs, 0, exclude_values=None)
        assert values.tolist() == [0.0, 1.0, 2.0]

    def test_build_dataframe_schema(self):
        vals = np.array([[3.0, 2.0], [1.0, 4.0]])
        errs = np.array([[1.0, 2.0], [0.5, 0.7]])
        pcts = np.array([[10.0, 20.0], [5.0, 7.0]])
        df = _build_dataframe(vals, errs, pcts)
        assert list(df.columns) == [
            "sample_idx", "sens_1", "err_1", "err_pct_1",
            "sens_2", "err_2", "err_pct_2",
        ]
        assert len(df) == 2
        assert df["sens_1"].dtype.kind == "i"  # integral values cast to int

    def test_predict_test_from_ckpt_back_compatible(self):
        # The existing prediction entry point must keep its signature
        params = inspect.signature(predict_test_from_ckpt).parameters
        assert "input_conditioning_fn" in params
        assert "dataset_label" in params


# ---------------------------------------------------------------------------
# End-to-end: reference backend
# ---------------------------------------------------------------------------

class TestReferenceBackend:
    def test_greedy_ranking_recovers_weight_order(self):
        X, Y = _make_data(B=6)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(),
            datadir_path=None,
            checkpoint_path=None,
            n_most_sensitive=3,
            predict_fn=predict_fn,
            backend="reference",
        )
        assert len(df) == 6
        for k, expected in enumerate(EXPECTED_ORDER[:3], start=1):
            assert (df[f"sens_{k}"] == expected).all()

    def test_errors_are_cumulative_and_pct_consistent(self):
        X, Y = _make_data(B=6)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=2, predict_fn=predict_fn, backend="reference",
        )
        # err_2 >= err_1 (err_k = change with top-k removed, additive model)
        assert (df["err_2"] >= df["err_1"] - 1e-6).all()
        # err_1 = (w3 * 1)^2 (values are all 1)
        expected_err1 = WEIGHTS[3] ** 2
        assert math.isclose(df.loc[0, "err_1"], expected_err1, rel_tol=1e-5)
        # pct = 100 * err / mean(base^2)
        base0 = _weighted_sum(X[0:1], WEIGHTS).item()
        assert math.isclose(
            df.loc[0, "err_pct_1"], 100.0 * expected_err1 / base0 ** 2,
            rel_tol=1e-4,
        )

    def test_per_sample_selection_differs(self):
        X, Y = _make_data(B=4)
        # Sample-specific weights: even samples favor var 3, odd favor var 1
        w_table = torch.zeros(4, 6)
        for b in range(4):
            for v, w in WEIGHTS.items():
                w_table[b, v] = w
        w_table[1, 1] = 5.0  # sample 1: variable 1 dominates
        w_table[3, 1] = 5.0  # sample 3: variable 1 dominates
        predict_fn = _make_mock_predict_fn(X, Y, w_table)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=1, predict_fn=predict_fn, backend="reference",
        )
        assert df.loc[0, "sens_1"] == 3
        assert df.loc[1, "sens_1"] == 1
        assert df.loc[2, "sens_1"] == 3
        assert df.loc[3, "sens_1"] == 1

    def test_csv_output(self):
        # Note: workspace-local temp file (pytest tmp_path is blocked on this
        # Windows setup)
        X, Y = _make_data(B=3)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        out = Path("test/unit/_tmp_seq_sens/df_seq.csv")
        try:
            df = get_sequential_sensitivity_df(
                config=_make_config(), datadir_path=None, checkpoint_path=None,
                n_most_sensitive=2, predict_fn=predict_fn, backend="reference",
                output_path=out,
            )
            assert out.exists()
            df_loaded = pd.read_csv(out)
            assert df_loaded.shape == df.shape
            assert list(df_loaded.columns) == list(df.columns)
        finally:
            if out.exists():
                out.unlink()
            if out.parent.exists():
                out.parent.rmdir()

    def test_auto_backend_with_custom_predict_fn_uses_reference(self):
        X, Y = _make_data(B=2)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=1, predict_fn=predict_fn, backend="auto",
        )
        assert (df["sens_1"] == 3).all()

    def test_reference_rejects_train_split(self):
        X, Y = _make_data(B=2)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        with pytest.raises(ValueError, match="deterministic batch order"):
            get_sequential_sensitivity_df(
                config=_make_config(), datadir_path=None, checkpoint_path=None,
                n_most_sensitive=1, predict_fn=predict_fn, backend="reference",
                split="train",
            )

    def test_k_exceeds_candidates_raises(self):
        X, Y = _make_data(B=2)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        with pytest.raises(ValueError, match="exceeds the number of"):
            get_sequential_sensitivity_df(
                config=_make_config(), datadir_path=None, checkpoint_path=None,
                n_most_sensitive=10, predict_fn=predict_fn, backend="reference",
            )


# ---------------------------------------------------------------------------
# End-to-end: optimized backend (mock predictor)
# ---------------------------------------------------------------------------

class TestOptimizedBackend:
    def test_matches_reference_backend(self):
        X, Y = _make_data(B=6)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        df_ref = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=3, predict_fn=predict_fn, backend="reference",
        )
        predictor = _MockPredictor(X, Y, WEIGHTS)
        df_opt = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=3, backend="optimized", predictor=predictor,
            chunk_size=2,  # exercise candidate chunking
        )
        pd.testing.assert_frame_equal(df_ref, df_opt, check_dtype=False)

    def test_auto_candidate_values(self):
        X, Y = _make_data(B=4)
        predictor = _MockPredictor(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=2, backend="optimized", predictor=predictor,
            values_to_mask=None,  # auto-collect from control channel
        )
        assert (df["sens_1"] == 3).all()
        assert (df["sens_2"] == 2).all()

    def test_probabilistic_auto_w2(self):
        # Gaussian head: mu = weighted sum, raw_sigma constant -> w2 = |d_mu|
        X, Y = _make_data(B=4)

        class MockPredictorProba(_MockPredictor):
            def _forward(self, X, trg, **kwargs):
                mu = _weighted_sum(X, self._weights)
                raw_sigma = torch.full_like(mu, 0.5)
                return torch.stack([mu, raw_sigma], dim=-1)  # (B, 1, 2)

        predictor = MockPredictorProba(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(loss_fn="gaussian", out_dim=2),
            datadir_path=None, checkpoint_path=None,
            n_most_sensitive=2, backend="optimized", predictor=predictor,
            error_metric="auto",
        )
        assert (df["sens_1"] == 3).all()
        assert (df["sens_2"] == 2).all()
        # w2 with constant sigma = |d_mu| = w3 * 1 (values are all 1)
        assert math.isclose(df.loc[0, "err_1"], WEIGHTS[3], rel_tol=1e-4)

    def test_oom_backoff_recovers(self):
        # Mock predictor that OOMs when too many expanded rows are requested
        class MockPredictorOOM(_MockPredictor):
            def __init__(self, *args, max_rows=12, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_rows = max_rows
                self.n_oom = 0

            def _forward(self, X, trg, **kwargs):
                if X.shape[0] > self.max_rows:
                    self.n_oom += 1
                    raise torch.cuda.OutOfMemoryError("mock OOM")
                return super()._forward(X, trg, **kwargs)

        X, Y = _make_data(B=6)
        predictor = MockPredictorOOM(X, Y, WEIGHTS, max_rows=12)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=2, backend="optimized", predictor=predictor,
            chunk_size=4,  # 4*4=16 rows > 12 -> OOM -> halved to 2
        )
        assert predictor.n_oom > 0  # backoff was exercised
        assert (df["sens_1"] == 3).all()  # results still correct
        assert (df["sens_2"] == 2).all()

    def test_auto_chunk_size_from_memory_budget(self):
        # Tiny mock model: auto chunk_size is capped by n_cand, no error
        X, Y = _make_data(B=4)
        predictor = _MockPredictor(X, Y, WEIGHTS)
        df = get_sequential_sensitivity_df(
            config=_make_config(), datadir_path=None, checkpoint_path=None,
            n_most_sensitive=1, backend="optimized", predictor=predictor,
            chunk_size=None, max_chunk_memory_mb=2048,
        )
        assert (df["sens_1"] == 3).all()

    def test_k_exceeds_candidates_raises(self):
        X, Y = _make_data(B=2)
        predictor = _MockPredictor(X, Y, WEIGHTS)
        with pytest.raises(ValueError, match="exceeds the number of"):
            get_sequential_sensitivity_df(
                config=_make_config(), datadir_path=None, checkpoint_path=None,
                n_most_sensitive=10, backend="optimized", predictor=predictor,
            )

    def test_optimized_rejects_custom_predict_fn_without_predictor(self):
        X, Y = _make_data(B=2)
        predict_fn = _make_mock_predict_fn(X, Y, WEIGHTS)
        with pytest.raises(ValueError, match="does not use predict_fn"):
            get_sequential_sensitivity_df(
                config=_make_config(), datadir_path=None, checkpoint_path=None,
                n_most_sensitive=1, predict_fn=predict_fn, backend="optimized",
            )
