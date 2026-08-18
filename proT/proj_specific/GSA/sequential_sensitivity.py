"""
Sequential (greedy) per-sample input sensitivity analysis.

Algorithm
---------
1. Baseline: predict with the full input (nothing masked).
2. Step k = 1..K: for every remaining candidate input token value v, mask the
   already-selected tokens of each sample PLUS v (set the intervention feature
   to NaN at positions where the control feature equals the value; NAIM then
   blocks those tokens natively), predict, and measure the error between the
   baseline prediction and the masked prediction. For each sample, the
   candidate with the argmax error is the k-th most sensitive input; it is
   added to the sample's masked set and the process repeats.

Because the error is always measured against the baseline, err_k is the total
prediction change after removing the top-k inputs (errors are non-decreasing
in k for additive effects).

The number of most sensitive inputs K is a parameter, like n_components in
PCA.

Literature
----------
- First iteration = LOCO (leave-one-covariate-out; Lei et al. 2018); the
  sequential extension is a greedy/conditional LOCO.
- Greedy deterministic analog of SAGE (Covert, Lundberg & Lee, NeurIPS 2020):
  recorded errors are marginal contributions given the already-removed set.
- Sequential forward selection (Whitney 1971) applied to sensitivity ranking.
- Removal-based explanations (Sturmfels et al., JMLR 2021): NAIM's native
  NaN masking is a principled removal mechanism (no surrogate needed).
- Per-sample ranking follows erasure/occlusion attribution (Li et al. 2016;
  Dynamask, Crabbe & van der Schaar 2021).
- Probabilistic predictions are compared with distribution distances, in the
  spirit of moment-independent sensitivity measures (Borgonovo 2007). For
  Gaussian predictive distributions (mu, sigma) we use closed-form
  2-Wasserstein (default) or KL divergence.

Note on correlated inputs: individual removal effects can hide shared
information; the sequential conditional selection partially mitigates this
(once one of two correlated inputs is removed, the other's marginal effect
becomes visible at the next step).

Output
------
DataFrame (optionally written to CSV) with one row per test sample and
columns: sample_idx, sens_1, err_1, err_pct_1, ..., sens_K, err_K, err_pct_K.
err_pct_k = 100 * err_k / baseline_scale, where baseline_scale is the RMS of
the baseline prediction for that sample (mean of squares for squared metrics).
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from proT.evaluation.predict import predict_test_from_ckpt, create_predictor
from proT.training.experiment_control import update_config

# Default sigma floor, matches proT/training/losses.py (GaussianNLL /
# CensoredGaussianNLL): sigma = softplus(raw_sigma) + eps
DEFAULT_SIGMA_EPS = 1e-3

DETERMINISTIC_METRICS = ("mse", "rmse", "mae")
PROBABILISTIC_METRICS = ("w2", "kl")


# ---------------------------------------------------------------------------
# Error metrics (per-sample, vectorized)
# ---------------------------------------------------------------------------

def _to_bpc(t: torch.Tensor) -> torch.Tensor:
    """Bring a prediction tensor to canonical (B, P, C) form."""
    if t.ndim == 1:
        return t.unsqueeze(-1).unsqueeze(-1)  # (B,) -> (B, 1, 1)
    if t.ndim == 2:
        return t.unsqueeze(-1)                # (B, P) -> (B, P, 1)
    if t.ndim == 3:
        return t
    raise ValueError(f"Expected 1D-3D prediction tensor, got shape {t.shape}")


def _valid_mask(baseline_bpc: torch.Tensor) -> torch.Tensor:
    """Valid positions = non-NaN in the baseline mean channel. Returns (B, P)."""
    return ~torch.isnan(baseline_bpc[:, :, 0])


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of x (B, P) over valid positions, per sample. NaN if none valid."""
    x = torch.nan_to_num(x, nan=0.0)
    denom = valid.float().sum(dim=1)
    out = x.sum(dim=1) / denom.clamp_min(1.0)
    out[denom == 0] = float("nan")
    return out


def _mse_per_sample(base: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _masked_mean((base[:, :, 0] - mask[:, :, 0]) ** 2, valid)


def _rmse_per_sample(base: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_mse_per_sample(base, mask, valid).clamp_min(0.0))


def _mae_per_sample(base: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _masked_mean((base[:, :, 0] - mask[:, :, 0]).abs(), valid)


def _w2_per_sample(
    base: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> torch.Tensor:
    """
    Root-mean-square 2-Wasserstein distance between Gaussian predictions.

    Per position: W2^2 = (mu1 - mu2)^2 + (sigma1 - sigma2)^2 (closed form for
    univariate Gaussians). Aggregated as sqrt(mean(W2^2)) over valid positions.
    Same units as the target; accounts for both mean shift and spread change.
    """
    mu1, mu2 = base[:, :, 0], mask[:, :, 0]
    s1 = F.softplus(base[:, :, 1]) + sigma_eps
    s2 = F.softplus(mask[:, :, 1]) + sigma_eps
    w2_sq = (mu1 - mu2) ** 2 + (s1 - s2) ** 2
    return torch.sqrt(_masked_mean(w2_sq, valid).clamp_min(0.0))


def _kl_per_sample(
    base: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    sigma_eps: float = DEFAULT_SIGMA_EPS,
) -> torch.Tensor:
    """
    Mean KL divergence KL(N_base || N_masked) over valid positions (nats).

    Closed form for univariate Gaussians:
    KL = log(s2/s1) + (s1^2 + (mu1-mu2)^2) / (2 s2^2) - 1/2.
    """
    mu1, mu2 = base[:, :, 0], mask[:, :, 0]
    s1 = F.softplus(base[:, :, 1]) + sigma_eps
    s2 = F.softplus(mask[:, :, 1]) + sigma_eps
    kl = torch.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * s2 ** 2) - 0.5
    return _masked_mean(kl, valid)


def _is_probabilistic_config(config: dict) -> bool:
    """True if the model was trained with a probabilistic (mu, raw_sigma) head.

    For the lognormal losses, mu and sigma parameterize log(y + shift); the
    w2/kl metrics then measure distances in log-space (the Gaussian formulas
    apply to the underlying normal in the transformed space).
    """
    loss_fn = config.get("training", {}).get("loss_fn", "mse")
    out_dim = config.get("model", {}).get("kwargs", {}).get("out_dim", 1)
    return loss_fn in (
        "gaussian", "gaussian_censored", "lognormal", "lognormal_censored"
    ) and out_dim == 2


def _resolve_error_metric(error_metric: str, probabilistic: bool) -> str:
    """Resolve 'auto' and validate the metric against the prediction type."""
    if error_metric == "auto":
        return "w2" if probabilistic else "mse"
    if error_metric in PROBABILISTIC_METRICS and not probabilistic:
        raise ValueError(
            f"Metric '{error_metric}' requires a probabilistic (mu, raw_sigma) "
            "prediction, but the config does not indicate a Gaussian head. "
            f"Use one of {DETERMINISTIC_METRICS} or 'auto'."
        )
    if error_metric not in DETERMINISTIC_METRICS + PROBABILISTIC_METRICS:
        raise ValueError(
            f"Unknown error_metric '{error_metric}'. Available: "
            f"{DETERMINISTIC_METRICS + PROBABILISTIC_METRICS} or 'auto'."
        )
    return error_metric


def _get_error_fn(
    error_metric: str, sigma_eps: float = DEFAULT_SIGMA_EPS
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], bool]:
    """
    Return (error_fn, is_squared_metric).

    error_fn(base, masked) takes raw prediction tensors (any of (B,), (B,P),
    (B,P,C)) and returns per-sample errors (B,). NaN handling: positions with
    NaN in the baseline mean channel are excluded.
    """
    fn_map = {
        "mse": (_mse_per_sample, True),
        "rmse": (_rmse_per_sample, False),
        "mae": (_mae_per_sample, False),
        "w2": (lambda b, m, v: _w2_per_sample(b, m, v, sigma_eps), False),
        "kl": (lambda b, m, v: _kl_per_sample(b, m, v, sigma_eps), False),
    }
    fn, is_squared = fn_map[error_metric]

    def error_fn(base: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        base_bpc = _to_bpc(base)
        masked_bpc = _to_bpc(masked)
        if base_bpc.shape != masked_bpc.shape:
            raise ValueError(
                f"Baseline/masked prediction shape mismatch: "
                f"{base_bpc.shape} vs {masked_bpc.shape}"
            )
        valid = _valid_mask(base_bpc)
        return fn(base_bpc, masked_bpc, valid)

    return error_fn, is_squared


def _baseline_scale(base: torch.Tensor, is_squared: bool) -> torch.Tensor:
    """
    Per-sample interpretability scale: RMS of the baseline mean channel
    (mean of squares for squared metrics). Used for err_pct.
    """
    base_bpc = _to_bpc(base)
    valid = _valid_mask(base_bpc)
    mean_sq = _masked_mean(base_bpc[:, :, 0] ** 2, valid)
    return mean_sq if is_squared else torch.sqrt(mean_sq.clamp_min(0.0))


def _error_pct(err: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Percentage of the baseline scale; NaN where the scale is ~0 or err NaN."""
    tiny = 1e-12
    pct = 100.0 * err / scale.clamp_min(tiny)
    pct = torch.where(scale < tiny, torch.full_like(pct, float("nan")), pct)
    return pct


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _candidate_mask(
    X_ctrl: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Boolean mask (B, n_values, L): True where the control channel equals a
    candidate value. X_ctrl is (B, L), values is (n_values,).
    """
    return X_ctrl.unsqueeze(1) == values.view(1, -1, 1).to(X_ctrl.device)


def _apply_mask(
    X: torch.Tensor,
    mask_bl: torch.Tensor,
    intervention_feat_idx: int,
) -> torch.Tensor:
    """Set the intervention feature to NaN where mask_bl (B, L) is True."""
    X_masked = X.clone()
    X_masked[:, :, intervention_feat_idx][mask_bl] = float("nan")
    return X_masked


def _move_batch(batch, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Move a (X, trg) batch to device. Returns (X, trg) or (X, None)."""
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)
    return batch.to(device), None


class _PerSampleMaskingFn:
    """
    Stateful input_conditioning_fn for the reference backend.

    Applies, per sample of the current batch, the sample's accumulated
    selected values plus one shared candidate value. Relies on deterministic
    batch order (shuffle=False dataloaders: test/all). A fresh instance must
    be created for every full pass over the dataset.
    """

    def __init__(
        self,
        selected_per_sample: List[List[float]],
        candidate_value: float,
        control_feat_idx: int,
        intervention_feat_idx: int,
    ):
        self.selected_per_sample = selected_per_sample
        self.candidate_value = candidate_value
        self.control_feat_idx = control_feat_idx
        self.intervention_feat_idx = intervention_feat_idx
        self.cursor = 0

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        sel_slice = self.selected_per_sample[self.cursor:self.cursor + B]
        self.cursor += B

        X_ctrl = X[:, :, self.control_feat_idx]
        mask = X_ctrl == self.candidate_value  # (B, L)
        for i, sel_vals in enumerate(sel_slice):
            for v in sel_vals:
                mask[i] |= X_ctrl[i] == v

        return _apply_mask(X, mask, self.intervention_feat_idx)


# ---------------------------------------------------------------------------
# Candidate values
# ---------------------------------------------------------------------------

def _collect_candidate_values(
    inputs: np.ndarray,
    control_feat_idx: int,
    exclude_values: Optional[Tuple[float, ...]] = (0.0,),
) -> np.ndarray:
    """
    Unique sorted values of the control feature across the split, excluding
    NaN and exclude_values (default excludes 0.0 = padding convention).
    """
    ctrl = np.asarray(inputs)[..., control_feat_idx].ravel()
    ctrl = ctrl[~np.isnan(ctrl)]
    values = np.unique(ctrl)
    if exclude_values:
        values = values[~np.isin(values, np.asarray(list(exclude_values)))]
    if values.size == 0:
        raise ValueError("No candidate values found in the control feature.")
    return values


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def _build_dataframe(
    selected_vals: np.ndarray,   # (B_total, K)
    selected_errs: np.ndarray,   # (B_total, K)
    selected_pcts: np.ndarray,   # (B_total, K)
) -> pd.DataFrame:
    """Assemble the output DataFrame: one row per sample, K (sens, err, pct) triples."""
    B_total, K = selected_vals.shape
    data = {"sample_idx": np.arange(B_total)}
    integral = np.all(np.isclose(selected_vals, np.round(selected_vals)))
    for k in range(K):
        col = selected_vals[:, k]
        data[f"sens_{k + 1}"] = col.astype(int) if integral else col
        data[f"err_{k + 1}"] = selected_errs[:, k]
        data[f"err_pct_{k + 1}"] = selected_pcts[:, k]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Reference backend (injectable predict_fn, one full prediction per candidate)
# ---------------------------------------------------------------------------

def _sequential_sensitivity_reference(
    config: dict,
    datadir_path: Path,
    checkpoint_path: Path,
    n_most_sensitive: int,
    values_to_mask: Optional[np.ndarray],
    control_feat_idx: int,
    intervention_feat_idx: int,
    predict_fn: Callable,
    split: str,
    error_fn: Callable,
    is_squared: bool,
    cluster: bool,
    exclude_values: Tuple[float, ...],
) -> pd.DataFrame:
    """
    Reference backend: calls predict_fn once per candidate per step.

    Slow (each call reloads the model) but works with any existing prediction
    function with the predict_test_from_ckpt signature, e.g.
    predict_test_from_ckpt or predict_test_from_ckpt_adaptive.
    Requires deterministic batch order (split "test" or "all").
    """
    if split == "train":
        raise ValueError(
            "The reference backend requires deterministic batch order; "
            "split='train' uses shuffle=True. Use split='test'/'all' or "
            "backend='optimized'."
        )

    # Baseline prediction (no masking)
    print("[reference] Baseline prediction (no masking)...")
    res_base = predict_fn(
        config=config,
        datadir_path=datadir_path,
        checkpoint_path=checkpoint_path,
        dataset_label=split,
        cluster=cluster,
    )
    base = torch.from_numpy(np.asarray(res_base.outputs, dtype=np.float32))
    B_total = base.shape[0]
    scale = _baseline_scale(base, is_squared)

    if values_to_mask is None:
        values_to_mask = _collect_candidate_values(
            res_base.inputs, control_feat_idx, exclude_values
        )
    values_to_mask = np.asarray(values_to_mask)
    if n_most_sensitive > len(values_to_mask):
        raise ValueError(
            f"n_most_sensitive ({n_most_sensitive}) exceeds the number of "
            f"candidate values ({len(values_to_mask)})."
        )
    print(f"[reference] {B_total} samples, {len(values_to_mask)} candidate values.")

    selected_per_sample: List[List[float]] = [[] for _ in range(B_total)]
    selected_vals = np.full((B_total, n_most_sensitive), np.nan)
    selected_errs = np.full((B_total, n_most_sensitive), np.nan)
    selected_pcts = np.full((B_total, n_most_sensitive), np.nan)

    for k in range(n_most_sensitive):
        print(f"[reference] Step {k + 1}/{n_most_sensitive}: "
              f"{len(values_to_mask)} candidate predictions...")
        err_table = np.full((B_total, len(values_to_mask)), np.nan)

        for j, v in enumerate(tqdm(values_to_mask, desc=f"Step {k + 1}")):
            mask_fn = _PerSampleMaskingFn(
                selected_per_sample, v, control_feat_idx, intervention_feat_idx
            )
            res = predict_fn(
                config=config,
                datadir_path=datadir_path,
                checkpoint_path=checkpoint_path,
                dataset_label=split,
                cluster=cluster,
                input_conditioning_fn=mask_fn,
            )
            masked = torch.from_numpy(np.asarray(res.outputs, dtype=np.float32))
            err_table[:, j] = error_fn(base, masked).numpy()

        # Exclude already-selected values from the per-sample argmax
        for i in range(B_total):
            for v in selected_per_sample[i]:
                err_table[i, values_to_mask == v] = np.nan

        sel_idx = np.nanargmax(err_table, axis=1)
        for i in range(B_total):
            v = float(values_to_mask[sel_idx[i]])
            selected_per_sample[i].append(v)
            selected_vals[i, k] = v
            selected_errs[i, k] = err_table[i, sel_idx[i]]
        selected_pcts[:, k] = _error_pct(
            torch.from_numpy(selected_errs[:, k]), scale
        ).numpy()

    return _build_dataframe(selected_vals, selected_errs, selected_pcts)


# ---------------------------------------------------------------------------
# Optimized backend (single model load, batch expansion over candidates)
# ---------------------------------------------------------------------------

def _get_eval_loader(dm, split: str, batch_size: int):
    """
    Deterministic (shuffle=False) loader over the split's dataset.

    A fresh loader over the in-memory TensorDataset is built directly (also
    for 'train', whose default loader shuffles): per-sample mask tracking
    requires a fixed sample order, and num_workers=0 is fastest for
    in-memory data and safe on Windows.
    """
    ds = {"train": dm.train_ds, "test": dm.test_ds, "all": dm.all_ds}[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _safe_int(value, default: int) -> int:
    """int() that tolerates unresolved config interpolations (e.g. '${...}')."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _estimate_row_bytes(L: int, D: int, config: dict) -> int:
    """
    Rough per-expanded-row GPU memory estimate (bytes) for one forward pass.

    Dominated by the attention score tensors (n_heads * L^2 per attention
    layer) plus embedding/FFN activations (~12 * L * d_model). Used only for
    auto chunk sizing; the adaptive OOM backoff is the safety net.
    """
    kwargs = config.get("model", {}).get("kwargs", {})
    n_heads = _safe_int(kwargs.get("n_heads", 1), 1)
    e_layers = _safe_int(kwargs.get("e_layers", 1), 1)
    d_layers = _safe_int(kwargs.get("d_layers", 0), 0)
    d_model = _safe_int(
        config.get("model", {}).get("embed_dim", {}).get("d_model_set")
        or kwargs.get("d_model_enc"),
        128,
    )
    n_att = max(1, e_layers + d_layers)
    att_bytes = n_heads * L * L * 4 * n_att
    act_bytes = 12 * L * d_model * 4
    in_bytes = L * D * 4
    return in_bytes + att_bytes + act_bytes


def _chunk_errors(
    predictor,
    X: torch.Tensor,
    trg: Optional[torch.Tensor],
    base_b: torch.Tensor,
    acc_mask: torch.Tensor,
    cand_chunk: torch.Tensor,
    control_feat_idx: int,
    intervention_feat_idx: int,
    error_fn: Callable,
) -> torch.Tensor:
    """
    Forward-pass one candidate chunk via batch expansion, return errors (B, c).

    All large temporaries are function-local; the raw model output (which
    holds the attention weights of all layers) is deleted immediately after
    extracting the forecast, so peak memory is bounded by one chunk.
    """
    B, L, D = X.shape
    c = cand_chunk.shape[0]
    X_ctrl = X[:, :, control_feat_idx]

    # (B, c, L): candidate mask OR accumulated mask
    mask_exp = _candidate_mask(X_ctrl, cand_chunk) | acc_mask.unsqueeze(1)

    # Batch expansion: (B, c, L, D) -> (B*c, L, D)
    X_exp = X.unsqueeze(1).repeat(1, c, 1, 1)
    X_exp[:, :, :, intervention_feat_idx][mask_exp] = float("nan")
    X_exp = X_exp.reshape(B * c, L, D)
    del mask_exp
    trg_exp = None
    if trg is not None:
        trg_exp = trg.unsqueeze(1).repeat(1, c, 1, 1).reshape(
            B * c, trg.shape[1], trg.shape[2]
        )

    with torch.inference_mode():
        output = predictor._forward(X_exp, trg_exp)
        forecast = predictor._process_forward_output(output)["forecast"]
    del output, X_exp, trg_exp  # free attention weights and expanded inputs

    # (B*c, ...) -> (B, c, ...); error vs baseline per sample
    tail = forecast.shape[1:]
    err = error_fn(
        base_b.unsqueeze(1).expand(B, c, *base_b.shape[1:]).reshape(B * c, *base_b.shape[1:]),
        forecast.reshape(B * c, *tail),
    ).reshape(B, c)
    return err


def _sequential_sensitivity_optimized(
    config: dict,
    datadir_path: Path,
    checkpoint_path: Path,
    n_most_sensitive: int,
    values_to_mask: Optional[np.ndarray],
    control_feat_idx: int,
    intervention_feat_idx: int,
    split: str,
    error_fn: Callable,
    is_squared: bool,
    chunk_size: Optional[int],
    cluster: bool,
    exclude_values: Tuple[float, ...],
    predictor=None,
    max_chunk_memory_mb: float = 2048.0,
) -> pd.DataFrame:
    """
    Optimized backend.

    - Model loaded once (or reused via the `predictor` argument).
    - The dataset is iterated K+1 times total (1 baseline + K steps), not
      once per candidate.
    - Within each batch, all candidate masks are evaluated in parallel via
      batch expansion: (B, L, D) -> (B * n_chunk, L, D), chunked over
      candidates to bound GPU memory. Peak memory scales with
      batch_size * chunk_size * L^2 * n_heads (attention); if chunk_size
      is None it is derived from max_chunk_memory_mb, and on CUDA OOM the
      chunk is automatically halved and retried.
    - Only baseline outputs and per-step argmax results are kept in memory;
      expanded tensors and raw model outputs (attention weights of all
      layers) are deleted after each chunk.
    """
    seed_everything(config["training"]["seed"])
    torch.set_float32_matmul_precision("high")

    if predictor is None:
        config_updated = update_config(config)
        print("Loading model from checkpoint...")
        predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
    device = predictor.device

    dm = predictor.create_data_module(external_dataset=None, cluster=cluster)
    dm.prepare_data()
    dm.setup(stage=None)

    batch_size = config["training"]["batch_size"]

    def fresh_loader():
        return _get_eval_loader(dm, split, batch_size)

    # ------------------------------------------------------------------
    # Pass 1: baseline predictions (stored per batch on CPU)
    # ------------------------------------------------------------------
    print("[optimized] Pass 1: baseline predictions (no masking)...")
    base_list: List[torch.Tensor] = []
    ctrl_list: List[torch.Tensor] = []
    L_dim, D_dim = None, None
    for batch in tqdm(fresh_loader(), desc="Baseline"):
        X, trg = _move_batch(batch, device)
        if L_dim is None:
            L_dim, D_dim = X.shape[1], X.shape[2]
        with torch.inference_mode():
            output = predictor._forward(X, trg)
            forecast = predictor._process_forward_output(output)["forecast"]
        del output  # free attention weights of all layers immediately
        base_list.append(forecast.cpu())
        ctrl_list.append(X[:, :, control_feat_idx].cpu())

    B_total = sum(t.shape[0] for t in base_list)
    scale_list = [_baseline_scale(b, is_squared) for b in base_list]

    if values_to_mask is None:
        all_ctrl = torch.cat(ctrl_list, dim=0).numpy()
        values_to_mask = _collect_candidate_values(
            all_ctrl[..., None], 0, exclude_values
        )
    values_to_mask = np.asarray(values_to_mask, dtype=np.float32)
    n_cand = len(values_to_mask)
    if n_most_sensitive > n_cand:
        raise ValueError(
            f"n_most_sensitive ({n_most_sensitive}) exceeds the number of "
            f"candidate values ({n_cand})."
        )
    cand_tensor = torch.from_numpy(values_to_mask).to(device)
    val_to_col = {float(v): j for j, v in enumerate(values_to_mask)}

    if chunk_size is None:
        # Auto: derive from the memory budget. Attention memory scales with
        # expanded_rows * L^2 * n_heads, so a fixed row target is unsafe.
        row_bytes = _estimate_row_bytes(L_dim, D_dim, config)
        budget_bytes = max_chunk_memory_mb * 1024 ** 2
        chunk_size = max(1, int(budget_bytes // (batch_size * row_bytes)))
        print(f"[optimized] Auto chunk_size={chunk_size} "
              f"(~{row_bytes * batch_size * chunk_size / 1024 ** 2:.0f} MB "
              f"estimated peak per chunk, budget {max_chunk_memory_mb:.0f} MB)")
    chunk_size = min(chunk_size, n_cand)

    print(f"[optimized] {B_total} samples, {n_cand} candidates, "
          f"chunk_size={chunk_size} (expanded rows per pass: "
          f"~{batch_size * chunk_size}).")

    selected_vals = np.full((B_total, n_most_sensitive), np.nan)
    selected_errs = np.full((B_total, n_most_sensitive), np.nan)
    selected_pcts = np.full((B_total, n_most_sensitive), np.nan)
    # Per-sample accumulated selected values (python lists; B_total is small)
    selected_per_sample: List[List[float]] = [[] for _ in range(B_total)]

    # ------------------------------------------------------------------
    # Passes 2..K+1: greedy selection
    # ------------------------------------------------------------------
    for k in range(n_most_sensitive):
        print(f"[optimized] Step {k + 1}/{n_most_sensitive}...")
        sample_offset = 0

        for batch_idx, batch in enumerate(tqdm(fresh_loader(), desc=f"Step {k + 1}")):
            X, trg = _move_batch(batch, device)
            B, L, D = X.shape
            base_b = base_list[batch_idx].to(device)          # (B, ...)
            scale_b = scale_list[batch_idx].to(device)        # (B,)
            X_ctrl = X[:, :, control_feat_idx]                # (B, L)

            # Accumulated mask from previously selected values (B, L)
            acc_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            sel_slice = selected_per_sample[sample_offset:sample_offset + B]
            for i, sel_vals in enumerate(sel_slice):
                for v in sel_vals:
                    acc_mask[i] |= X_ctrl[i] == v

            err_all = torch.full((B, n_cand), float("nan"), device=device)

            # Chunk over candidates to bound memory; on CUDA OOM the chunk
            # is halved and retried (the reduction persists for the run)
            c0 = 0
            while c0 < n_cand:
                c = min(chunk_size, n_cand - c0)
                try:
                    err_all[:, c0:c0 + c] = _chunk_errors(
                        predictor, X, trg, base_b, acc_mask,
                        cand_tensor[c0:c0 + c],
                        control_feat_idx, intervention_feat_idx, error_fn,
                    )
                    c0 += c
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if chunk_size == 1:
                        raise
                    chunk_size = max(1, chunk_size // 2)
                    print(f"  [OOM] chunk_size reduced to {chunk_size}, retrying...")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Exclude already-selected values from the per-sample argmax
            for i, sel_vals in enumerate(sel_slice):
                for v in sel_vals:
                    err_all[i, val_to_col[v]] = float("nan")

            # torch has no nanargmax: map NaN (excluded candidates) to -inf
            sel_idx = torch.nan_to_num(err_all, nan=float("-inf")).argmax(dim=1)
            sel_err = err_all[torch.arange(B, device=device), sel_idx]
            sel_pct = _error_pct(sel_err, scale_b)

            for i in range(B):
                v = float(values_to_mask[sel_idx[i].item()])
                selected_per_sample[sample_offset + i].append(v)
                selected_vals[sample_offset + i, k] = v
            selected_errs[sample_offset:sample_offset + B, k] = sel_err.cpu().numpy()
            selected_pcts[sample_offset:sample_offset + B, k] = sel_pct.cpu().numpy()

            sample_offset += B

    return _build_dataframe(selected_vals, selected_errs, selected_pcts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sequential_sensitivity_df(
    config: dict,
    datadir_path: Path,
    checkpoint_path: Path,
    n_most_sensitive: int,
    values_to_mask: Optional[np.ndarray] = None,
    control_feat_idx: Optional[int] = None,
    intervention_feat_idx: Optional[int] = None,
    predict_fn: Callable = predict_test_from_ckpt,
    split: str = "test",
    error_metric: str = "auto",
    backend: str = "auto",
    chunk_size: Optional[int] = None,
    cluster: bool = False,
    exclude_values: Tuple[float, ...] = (0.0,),
    sigma_eps: float = DEFAULT_SIGMA_EPS,
    output_path: Optional[Path] = None,
    predictor=None,
    max_chunk_memory_mb: float = 2048.0,
) -> pd.DataFrame:
    """
    Greedy per-sample sequential input sensitivity analysis (like PCA's
    n_components, but for input tokens).

    At each step, the input token whose removal (NaN-masking, natively blocked
    by NAIM) maximizes the error against the full-input baseline prediction is
    selected per sample, kept masked, and the process repeats until
    n_most_sensitive inputs are selected.

    Args:
        config: Configuration dictionary.
        datadir_path: Path to data directory.
        checkpoint_path: Path to model checkpoint.
        n_most_sensitive: Number K of most sensitive inputs to find.
        values_to_mask: Candidate input token values (of the control feature).
            If None, unique non-NaN control values in the split are used
            (minus exclude_values).
        control_feat_idx: Index of the feature identifying the token
            (default: config.data.features.X.variable).
        intervention_feat_idx: Index of the feature set to NaN
            (default: config.data.features.X.value).
        predict_fn: Prediction function with the predict_test_from_ckpt
            signature (used by the reference backend). Keep back compatible.
        split: One of ["train", "test", "all"]. "train" is only supported by
            the optimized backend (a non-shuffled loader is built internally).
        error_metric: "auto" (w2 for probabilistic Gaussian heads, mse for
            deterministic), or one of "mse", "rmse", "mae", "w2", "kl".
            Deterministic metrics on a probabilistic model use the mu channel.
        backend: "auto" (optimized if predict_fn is predict_test_from_ckpt,
            else reference), "optimized" (single model load + batch
            expansion) or "reference" (one predict_fn call per candidate).
        chunk_size: Candidates evaluated simultaneously per batch in the
            optimized backend. None = auto, derived from max_chunk_memory_mb.
            Peak GPU memory scales with batch_size * chunk_size * L^2 *
            n_heads (attention scores dominate). On CUDA OOM the chunk is
            automatically halved and retried.
        max_chunk_memory_mb: Memory budget (MB) per expanded chunk used for
            auto chunk_size. Default 2048.
        cluster: Whether running on cluster.
        exclude_values: Control values excluded from the auto candidate set
            (default (0.0,) = padding convention).
        sigma_eps: Sigma floor for the Gaussian head, matching the training
            loss (sigma = softplus(raw_sigma) + sigma_eps). Default 1e-3.
        output_path: If given, the result DataFrame is written to this CSV.
        predictor: Advanced/testing: reuse an already-created predictor
            (optimized backend only), skipping model loading.

    Returns:
        DataFrame with one row per sample and columns:
        sample_idx, sens_1, err_1, err_pct_1, ..., sens_K, err_K, err_pct_K.
        err_k is the prediction change with the top-k inputs removed (vs the
        full-input baseline); err_pct_k is the same as a percentage of the
        sample's baseline RMS scale.

    Example:
        >>> df = get_sequential_sensitivity_df(
        ...     config, datadir_path, checkpoint_path,
        ...     n_most_sensitive=10,
        ...     output_path="experiments/evaluations/seq_sensitivity/df_seq.csv",
        ... )
    """
    assert split in ["train", "test", "all"], f"{split} is not a proper label!"
    if n_most_sensitive < 1:
        raise ValueError("n_most_sensitive must be >= 1")

    # Defaults from config (same convention as leave_one_out.py)
    if control_feat_idx is None:
        control_feat_idx = config["data"]["features"]["X"]["variable"]
    if intervention_feat_idx is None:
        intervention_feat_idx = config["data"]["features"]["X"]["value"]
    control_feat_idx = int(control_feat_idx)
    intervention_feat_idx = int(intervention_feat_idx)

    # Error metric
    probabilistic = _is_probabilistic_config(config)
    error_metric = _resolve_error_metric(error_metric, probabilistic)
    error_fn, is_squared = _get_error_fn(error_metric, sigma_eps)
    print(f"Error metric: {error_metric} (probabilistic={probabilistic})")

    # Backend resolution
    if backend == "auto":
        backend = "optimized" if predict_fn is predict_test_from_ckpt else "reference"
    if backend == "optimized" and predict_fn is not predict_test_from_ckpt and predictor is None:
        raise ValueError(
            "backend='optimized' does not use predict_fn (it loads the model "
            "once via create_predictor). Use backend='reference' with a custom "
            "predict_fn."
        )
    if backend not in ("optimized", "reference"):
        raise ValueError(f"Unknown backend '{backend}'. Use 'auto', 'optimized' or 'reference'.")

    print("=" * 80)
    print(f"SEQUENTIAL SENSITIVITY ANALYSIS (backend={backend}, K={n_most_sensitive})")
    print("=" * 80)

    if backend == "reference":
        df = _sequential_sensitivity_reference(
            config=config,
            datadir_path=datadir_path,
            checkpoint_path=checkpoint_path,
            n_most_sensitive=n_most_sensitive,
            values_to_mask=values_to_mask,
            control_feat_idx=control_feat_idx,
            intervention_feat_idx=intervention_feat_idx,
            predict_fn=predict_fn,
            split=split,
            error_fn=error_fn,
            is_squared=is_squared,
            cluster=cluster,
            exclude_values=exclude_values,
        )
    else:
        df = _sequential_sensitivity_optimized(
            config=config,
            datadir_path=datadir_path,
            checkpoint_path=checkpoint_path,
            n_most_sensitive=n_most_sensitive,
            values_to_mask=values_to_mask,
            control_feat_idx=control_feat_idx,
            intervention_feat_idx=intervention_feat_idx,
            split=split,
            error_fn=error_fn,
            is_squared=is_squared,
            chunk_size=chunk_size,
            cluster=cluster,
            exclude_values=exclude_values,
            predictor=predictor,
            max_chunk_memory_mb=max_chunk_memory_mb,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    print("\nSample results:")
    print(df.head())
    return df


if __name__ == "__main__":
    from os.path import dirname, abspath, join
    from omegaconf import OmegaConf
    from proT.proj_specific.GSA.leave_one_out import get_config_and_best_checkpoint

    ROOT_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))
    datadir_path = join(ROOT_DIR, r"data/input")
    experiment_path = join(ROOT_DIR, r"experiments/baseline_optuna/euler/baseline_proT_dyconex_sum_50908169")
    output_path = join(ROOT_DIR, r"experiments/evaluations/sequential_sensitivity/df_seq_sens.csv")

    config, checkpoint_path = get_config_and_best_checkpoint(experiment_path)

    df = get_sequential_sensitivity_df(
        config=config,
        datadir_path=datadir_path,
        checkpoint_path=checkpoint_path,
        n_most_sensitive=10,
        values_to_mask=np.arange(1, 11),  # e.g. variable ids 1..10
        split="test",
        error_metric="auto",
        backend="optimized",
        chunk_size=None,  # auto
        output_path=output_path,
    )
