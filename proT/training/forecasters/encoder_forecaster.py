"""
Encoder-only Forecaster for ProTEncoder (proT_enc).

Supports two head variants:
- "scalar": predicts a D-dimensional vector of scalar targets per sample
  (e.g. IST slope, fitting coefficients). The dataset provides the scalar
  targets directly; the model output is (B, D, out_dim).
- "gru": predicts the sequential target with a GRU whose initial hidden state
  is inferred from the encoder output via attention pooling. Supports the
  "real IST" inference mode (teacher forcing on the first k positions) and
  the scheduled curriculum used by OnlineTargetForecaster.
- "gru_per_var": like "gru", but with one independent GRU (and output head)
  per target variable; each variable's rollout starts from its own
  encoder-derived initial hidden state and no hidden state flows across
  variables. Requires model.kwargs n_dec_vars and dec_var_idx.

The probabilistic losses (PositionWeightedMSE / CensoredGaussianNLL) are kept
unchanged: the model outputs (mu, raw_sigma) channels.
"""

from typing import Any, Optional
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm

from proT.core.model_encoder import ProTEncoder
from proT.training.losses import (
    PositionWeightedMSE,
    CensoredGaussianNLL,
    GaussianNLL,
    LogNormalNLL,
    CensoredLogNormalNLL,
    StudentTNLL,
    CensoredLaplaceNLL,
    lognormal_crps,
    gaussian_crps,
    crossing_time_crps,
)
from proT.training.sample_weights import AdaptiveLossWeighter, summarize_target
from proT.training.hard_mining import HardMiningState
from proT.training.qk_schedule import configure_adamw_with_qk_ramp
from proT.training.curriculum import ForcingCurriculum
from proT.training.forecasters.target_mask import fraction_reveal_mask


class EncoderForecaster(pl.LightningModule):
    """
    Lightning module for the encoder-only ProTEncoder.

    Args:
        config: Configuration dictionary containing:
            - model.kwargs: ProTEncoder model parameters
            - training.loss_fn: "mse", "gaussian", "gaussian_censored",
              "lognormal" or "lognormal_censored"
            - training.loss_censor_threshold: threshold for the censored losses
            - training.loss_bce_on_observed: optional, default False
            - training.loss_bce_weight: weight of the bce_on_observed "alive"
              term (lognormal_censored only), default 1.0
            - training.loss_ttf_weight: weight of the crossing-time CRPS
              (lognormal_censored fail mode only), default 0.0 = off
            - training.loss_crps_weight: weight of the y-space CRPS added to
              the training loss (lognormal_censored), default 0.0 = off
            - training.loss_log_shift: shift s for the lognormal losses
              (log(y + s) is modeled; must satisfy min(target) + s > 0)
            - training.lr: learning rate for AdamW
            - training.weight_decay: optional, default 0.01
            - training.lam: BCE loss weight (optional, default 1.0)
            - training.gamma: entropy regularization weight (optional, default 0.1)
            - training.entropy_regularizer: enable entropy reg (optional, default False)
            - data.val_idx: index of the value feature in the target
            - data.pos_idx: index of the position feature in the target
            - training.target_show_mode: "fixed", "random" or "scheduled" (optional)
            - training.epoch_show_trg: epoch to start showing targets (optional)
            - training.show_trg_max_idx: max position to show (fixed mode)
            - training.show_trg_upper_bound_max: upper bound (random mode)
            - training.show_trg_max_idx_start: initial forcing bound (scheduled mode)
            - training.show_trg_max_idx_end: final forcing bound (scheduled mode, default 0)
            - training.show_trg_schedule: "linear" or "cosine" (scheduled mode, default "linear")
            - training.show_trg_schedule_end_frac: fraction of max_epochs at which the
              final bound is reached (scheduled mode, default 1.0)
            - training.show_trg_random_draw: if True (default), draw the threshold
              uniformly in [0, current bound] each epoch; if False, use the bound
              directly (scheduled mode)
            - training.show_trg_units: "index" (default) or "fraction". In
              "fraction" mode all show_trg bounds above are floats in [0, 1]
              interpreted as a fraction of each sample's own valid (non-NaN)
              target length, so short sequences (e.g. early IST failure) get
              the same fraction revealed as full-length ones.

    Note:
        In "scheduled" mode, validation and test are always evaluated without
        teacher forcing (show_trg_max_idx=0), so metrics are comparable across
        epochs and best-checkpoint selection is unbiased.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = ProTEncoder(**config["model"]["kwargs"])

        # Loss function (position-weighted, NaN-target masked)
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = PositionWeightedMSE(
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
            )
        elif config["training"]["loss_fn"] == "gaussian_censored":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "gaussian_censored loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = CensoredGaussianNLL(
                threshold=config["training"].get("loss_censor_threshold", 1.0),
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                bce_on_observed=config["training"].get("loss_bce_on_observed", False),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
            )
        elif config["training"]["loss_fn"] == "gaussian":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "gaussian loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = GaussianNLL(
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
            )
        elif config["training"]["loss_fn"] == "lognormal":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "lognormal loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = LogNormalNLL(
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
                shift=config["training"].get("loss_log_shift", 1e-3),
                nll_beta=config["training"].get("loss_nll_beta", 0.0),
                var_reg_weighted=config["training"].get("loss_var_reg_weighted", False),
                sw_mean_only=config["training"].get("loss_sw_mean_only", False),
            )
        elif config["training"]["loss_fn"] == "lognormal_censored":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "lognormal_censored loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = CensoredLogNormalNLL(
                threshold=config["training"].get("loss_censor_threshold", 1.0),
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                bce_on_observed=config["training"].get("loss_bce_on_observed", False),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
                shift=config["training"].get("loss_log_shift", 1e-3),
                var_reg_weighted=config["training"].get("loss_var_reg_weighted", False),
                expected_len=config["training"].get("loss_expected_len", None),
                fail_margin=config["training"].get("loss_fail_margin", 0.8),
                fail_anchor=config["training"].get("loss_fail_anchor", 0.0),
                bce_weight=config["training"].get("loss_bce_weight", 1.0),
                sigma_y_floor=config["training"].get("loss_sigma_y_floor", 0.0),
                sigma_y_floor_weight=config["training"].get(
                    "loss_sigma_y_floor_weight", 1.0),
            )
        elif config["training"]["loss_fn"] == "student":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "student loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = StudentTNLL(
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
                shift=config["training"].get("loss_log_shift", 1e-3),
                df=config["training"].get("loss_student_df", 4.0),
            )
        elif config["training"]["loss_fn"] == "laplace_censored":
            assert config["model"]["kwargs"].get("out_dim") == 2, \
                "laplace_censored loss requires out_dim=2 (channels: mu, raw_sigma)"
            self.loss_fn = CensoredLaplaceNLL(
                threshold=config["training"].get("loss_censor_threshold", 1.0),
                scheme=config["training"].get("loss_pos_weight", "none"),
                alpha=config["training"].get("loss_pos_weight_alpha", 2.0),
                bce_on_observed=config["training"].get("loss_bce_on_observed", False),
                var_reg=config["training"].get("loss_var_reg", 0.0),
                var_reg_form=config["training"].get("loss_var_reg_form", "raw_sigma"),
                shift=config["training"].get("loss_log_shift", 1e-3),
            )
        else:
            raise ValueError(f"Unsupported loss function: {config['training']['loss_fn']}")

        # Extra objectives (lognormal_censored): crossing-time CRPS (needs
        # fail mode) and y-space CRPS (bounded, sigma-honest). 0 = off.
        self.ttf_weight = float(config["training"].get("loss_ttf_weight", 0.0) or 0.0)
        self.crps_weight = float(config["training"].get("loss_crps_weight", 0.0) or 0.0)
        if self.ttf_weight > 0:
            assert getattr(self.loss_fn, "fail_mode", False), \
                "loss_ttf_weight requires lognormal_censored with loss_expected_len"

        # Flag for the censored-likelihood mode (derived mask, no BCE head)
        self.censored_loss = config["training"]["loss_fn"] in (
            "gaussian_censored", "lognormal_censored", "laplace_censored"
        )
        # Flag for log-space losses: mu parameterizes log(y + shift), so the
        # point prediction in y-space is the median exp(mu) - shift
        self.log_space_loss = config["training"]["loss_fn"] in (
            "lognormal", "lognormal_censored", "student", "laplace_censored"
        )
        self.log_shift = config["training"].get("loss_log_shift", 1e-3)
        # GRU heads: the autoregressive feedback must live in y-space, so the
        # model converts mu -> exp(mu) - shift when a log-space loss is used
        # (identity otherwise). See ProTEncoder._feedback_value.
        self.model.feedback_shift = self.log_shift if self.log_space_loss else None

        # Delta (incremental) output mode: the GRU head outputs per-step
        # increments in the native loss space, accumulated in-graph to the
        # cumulative distribution (see ProTEncoder). The main loss applies to
        # the cumulative channels; an additional Gaussian NLL supervises the
        # increments directly onto the decomposed target (see
        # _increment_targets). Only likelihoods whose increments sum in
        # closed form are supported (Gaussian/lognormal; not student/laplace).
        # input token dropout (train only, see _step); 0 = disabled
        self.input_token_dropout = float(
            config["training"].get("input_token_dropout", 0.0) or 0.0)
        self.enc_val_idx = int(config["data"]["features"]["X"]["value"])

        self.delta_mode = getattr(self.model, "rnn_output_mode", "absolute") == "delta"
        if self.delta_mode:
            assert self.model.head_type in ("gru", "gru_per_var"), \
                "rnn_output_mode='delta' requires a GRU head"
            assert config["training"]["loss_fn"] in (
                "gaussian", "gaussian_censored", "lognormal", "lognormal_censored"
            ), "rnn_output_mode='delta' supports only gaussian/lognormal losses " \
               "(increments must sum in closed form)"
            self.model.sigma_eps = getattr(self.loss_fn, "eps", 1e-3)
            self.delta_loss_fn = GaussianNLL(
                scheme="none", eps=getattr(self.loss_fn, "eps", 1e-3))
            self.delta_weight = config["training"].get("loss_delta_weight", 1.0)

        # Absorbing "failed" state (gru_per_var, eval only): give the model the
        # failure threshold and the y/log-space convention. See
        # ProTEncoder._apply_fail_stop.
        if getattr(self.model, "rnn_absorbing_fail", False):
            self.model.fail_threshold = config["training"].get(
                "loss_censor_threshold", 1.0)
            self.model.fail_shift = self.log_shift if self.log_space_loss else None

        # Autoregressive runaway guard (train + eval): clamp the GRU feedback
        # value at 1.5x the censor threshold (y-space). Deliberately
        # INDEPENDENT of the eval-only fail_threshold above: _step disables
        # fail_threshold during training, and coupling the clamp to it left
        # the guard off in training (root cause of the NaN cascade of run
        # 16140288). Censored losses only - they define a meaningful y-space
        # threshold.
        if config["training"]["loss_fn"] in (
                "gaussian_censored", "lognormal_censored", "laplace_censored") \
                and hasattr(self.model, "feedback_clamp_max"):
            self.model.feedback_clamp_max = 1.5 * config["training"].get(
                "loss_censor_threshold", 1.0)

        # Adaptive sample weighting (training only; val/test stay unweighted
        # so metrics remain comparable across loss variants)
        self.sample_weight_scheme = config["training"].get("loss_sample_weight", "none")
        assert self.sample_weight_scheme in ("none", "adaptive"), \
            f"Unknown loss_sample_weight: {self.sample_weight_scheme}"
        self.adaptive_weighter = None
        if self.sample_weight_scheme == "adaptive":
            self.adaptive_weighter = AdaptiveLossWeighter(
                gamma=config["training"].get("loss_sw_gamma", 1.0),
                form=config["training"].get("loss_sw_form", "linear"),
                momentum=config["training"].get("loss_sw_momentum", 0.9),
                w_min=config["training"].get("loss_sw_min", 0.1),
                w_max=config["training"].get("loss_sw_max", 5.0),
                start_epoch=config["training"].get("loss_sw_start_epoch", 0),
            )
        # Tail-MAE diagnostic: unweighted MAE on high-target (minority) samples
        self.tail_quantile = config["training"].get("loss_sw_tail_quantile", 0.95)
        self._tail_threshold = None

        # Hard-example mining: shared per-sample loss memory, injected per fold
        # by trainer.py (None = disabled). See proT/training/hard_mining.py.
        self.hard_mining_state: Optional[HardMiningState] = None

        # Loss weights
        self.lam = config["training"].get("lam", 1.0)
        self.gamma = config["training"].get("gamma", 0.1)
        self.entropy_regularizer = config["training"].get("entropy_regularizer", False)

        # Data indices
        self.dec_val_idx = config["data"]["val_idx"]
        self.dec_pos_idx = config["data"]["pos_idx"]
        # variable channel in Y (gru_per_var head; needed by the fail-mode loss)
        self.dec_var_idx = config["model"]["kwargs"].get("dec_var_idx", None)

        # Online target configuration: the whole teacher-forcing curriculum
        # (schedule, warmup, random draw, units, validation) lives in
        # proT/training/curriculum.py - one unit-testable place.
        self.curriculum = ForcingCurriculum.from_config(config)
        # synced copies used by forward()/_build_trg_pos_mask()/training_step
        self.target_show_mode = self.curriculum.mode
        self.epoch_show_trg = self.curriculum.epoch_show_trg
        self.show_trg_max_idx = self.curriculum.show_trg_max_idx
        self.show_trg_max_idx_upper_bound = self.curriculum.upper_bound_max
        self.show_trg_units = self.curriculum.units
        # fixed conditioning fractions for the val_crps_cond{pct} diagnostic
        # (fraction units; empty = off)
        self.val_cond_fracs = list(config["training"].get("val_cond_fracs", None) or [])
        if self.val_cond_fracs:
            assert self.show_trg_units == "fraction", \
                "val_cond_fracs requires show_trg_units='fraction'"
            assert self.model.head_type in ("gru", "gru_per_var"), \
                "val_cond_fracs requires a GRU head"

        # State tracking
        self.show_trg_active = False
        # rate limiter for the non-finite gradient warning (on_after_backward)
        self._nonfinite_grad_warns = 0

        # Save hyperparameters
        self.save_hyperparameters(config)

        # Metrics
        self.mae = tm.MeanAbsoluteError()
        self.rmse = tm.MeanSquaredError(squared=False)
        self.r2 = tm.R2Score()

        print(f"✓ EncoderForecaster initialized")
        print(f"  - Head type: {self.model.head_type}")
        print(f"  - Target show mode: {self.target_show_mode}")
        print(f"  - Optimizer: AdamW (lr={config['training']['lr']})")

    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor,
                show_trg_max_idx: float = None, predict_mode: bool = False) -> Any:
        # type: (torch.Tensor, torch.Tensor, float, bool) -> Any
        """
        Forward pass with optional target revealing.

        Args:
            data_input: Encoder input tensor (B, L_enc, D_enc)
            data_trg: Target tensor (B, L_dec, D_dec) for the GRU head;
                ignored by the scalar head (may be None).
            show_trg_max_idx: Maximum position index to reveal (or fraction of
                each sample's valid length when show_trg_units="fraction")
            predict_mode: If True, use show_trg_max_idx from config

        Returns:
            Tuple of (forecast_output, recon_output, attention_weights, masks, entropy)
        """
        encoder_in = data_input

        # Scalar head: no target tensor needed
        if self.model.head_type == "scalar":
            model_output, recon_output, (enc_self_att, _, pool_att), enc_mask, \
                (enc_self_ent, _, _) = self.model.forward(
                    input_tensor=encoder_in,
                    target_tensor=None,
                    trg_pos_mask=None
                )
            return model_output, recon_output, (enc_self_att, [], pool_att), enc_mask, \
                (enc_self_ent, [], [])

        # GRU head: build the target tensor with the given indicator
        dec_input = data_trg.clone()

        # Determine how much target to show
        if show_trg_max_idx is None:
            if self.show_trg_max_idx is not None and (self.show_trg_active or predict_mode):
                show_trg_max_idx = self.show_trg_max_idx
            else:
                show_trg_max_idx = 0.0

        # Create mask for target positions. The bound is a scalar or, in
        # random_per_sample mode, a (B,) tensor of per-sample fractions
        if torch.is_tensor(show_trg_max_idx) and show_trg_max_idx.dim() > 0:
            show_any = bool((show_trg_max_idx > 0).any())
        else:
            show_any = show_trg_max_idx is not None and float(show_trg_max_idx) > 0
        if show_any:
            # Mask along sequence dimension (index or per-sample fraction mode)
            trg_pos_mask = self._build_trg_pos_mask(dec_input, show_trg_max_idx)

            # Create feature mask - only zero out value feature
            feature_mask = torch.zeros(dec_input.size(-1), dtype=torch.bool, device=dec_input.device)
            feature_mask[self.dec_val_idx] = True

            # Apply combined mask
            combined_mask = trg_pos_mask & feature_mask.unsqueeze(0).unsqueeze(0)
            dec_input[combined_mask] = 0.0

            # Create given indicator: 1.0 where targets are given, 0.0 where not given
            given_indicator = (~trg_pos_mask).float().squeeze(-1).unsqueeze(-1)  # B x L x 1
        else:
            # Zero out all values
            dec_input[:, :, self.dec_val_idx] = 0.0
            trg_pos_mask = None

            # Nothing is given
            given_indicator = torch.zeros(
                dec_input.size(0), dec_input.size(1), 1,
                device=dec_input.device, dtype=dec_input.dtype
            )  # B x L x 1

        # Append given indicator to decoder input (becomes last feature)
        dec_input = torch.cat([dec_input, given_indicator], dim=-1)  # B x L x (D+1)

        # Forward pass
        model_output, recon_output, (enc_self_att, _, pool_att), enc_mask, \
            (enc_self_ent, _, _) = self.model.forward(
                input_tensor=encoder_in,
                target_tensor=dec_input,
                trg_pos_mask=trg_pos_mask
            )

        return model_output, recon_output, (enc_self_att, [], pool_att), enc_mask, \
            (enc_self_ent, [], [])

    def _increment_targets(self, Y: torch.Tensor) -> torch.Tensor:
        """Decompose the absolute target into increments (delta mode).

        The increment target is d_t = z_t - z_{t-1} in the native loss space
        (z = log(y + shift) for log-space losses, z = y otherwise). The base
        z_{-1} is the native-space value of y = 0: log(shift) in log-space
        (was 0, which in log-space corresponds to y ~ 1 and made the first
        increment of every curve ~ -6.9, unmatchable for a monotone head),
        0 in y-space. For the gru_per_var head the previous position is taken
        within the same variable only (base z_0 at each variable's first
        position), mirroring the per-variable rollout state. NaN propagates:
        censored/missing positions (and increments across a missing previous
        value) are excluded by the NLL.
        """
        y = Y[:, :, self.dec_val_idx]                                # (B, L)
        # the clamp is a numerical last resort: the data can dip slightly
        # below -shift
        if self.log_space_loss:
            z = torch.log((y + self.log_shift).clamp_min(1e-8))
            z0 = math.log(self.log_shift)
        else:
            z = y
            z0 = 0.0
        base = torch.full_like(z, z0)
        base[:, 1:] = z[:, :-1]
        if self.model.head_type == "gru_per_var":
            var_ids = Y[:, :, self.dec_var_idx]
            # NaN != NaN, so padded tails get base z0 (excluded via NaN z anyway)
            same_var = var_ids[:, 1:] == var_ids[:, :-1]
            base[:, 1:] = torch.where(same_var, z[:, :-1],
                                      torch.full_like(z[:, 1:], z0))
        return z - base

    def _step(self, batch, stage: str = None):
        # type: (tuple, Optional[str]) -> tuple
        """
        Common step for train/val/test.

        Args:
            batch: Tuple of (input_data, target_data)
            stage: One of "train", "val", "test"

        Returns:
            Tuple of (loss, predicted_values, target_data)
        """
        # With hard-example mining the train loader yields (X, Y, sample_idx)
        if len(batch) == 3:
            X, Y, sample_idx = batch
        else:
            X, Y = batch
            sample_idx = None
        # Input token dropout (train only, anti-memorization): hide a random
        # fraction of the OBSERVED encoder values by setting them to NaN; the
        # existing "value_missing" embedding treats them as missing. The
        # process/step/variable ids stay, so the model knows WHAT is hidden.
        # A fresh subset per step breaks exact input fingerprints.
        if stage == "train" and self.input_token_dropout > 0:
            obs = ~torch.isnan(X[:, :, self.enc_val_idx])
            drop = obs & (torch.rand_like(X[:, :, self.enc_val_idx])
                          < self.input_token_dropout)
            X = X.clone()
            X[:, :, self.enc_val_idx] = X[:, :, self.enc_val_idx] \
                .masked_fill(drop, float("nan"))
        trg_val = Y[:, :, self.dec_val_idx]
        trg_nan = trg_val.isnan()
        mse_mask = None

        # Forward pass. In scheduled mode, val/test are always evaluated without
        # teacher forcing (the real prediction task), so metrics are comparable
        # across epochs and best-checkpoint selection is unbiased.
        # The absorbing fail-stop override is disabled inside _step: the NLL
        # must score the underlying predictive distribution, not the hard stop
        # decision (a false-positive crossing would zero an observed tail with
        # sigma -> 0 and explode the val/test NLL). The override stays active
        # for prediction/reporting (proT.evaluation.predict calls forward
        # directly).
        _ft = getattr(self.model, "fail_threshold", None)
        if _ft is not None:
            self.model.fail_threshold = None
        try:
            if stage in ("val", "test") and self.target_show_mode in (
                    "scheduled", "random_per_sample"):
                forecast_output, _, _, _, (enc_self_ent, _, _) = self.forward(
                    data_input=X, data_trg=Y, show_trg_max_idx=0.0
                )
            else:
                forecast_output, _, _, _, (enc_self_ent, _, _) = self.forward(
                    data_input=X, data_trg=Y
                )
        finally:
            if _ft is not None:
                self.model.fail_threshold = _ft

        # Runaway diagnostic: count non-finite model outputs. A non-finite
        # forecast no longer reaches the weights (loss-side nan_to_num +
        # on_after_backward gradient zeroing), but it must stay visible in
        # the logs instead of passing silently.
        if stage == "train":
            self.log("train_nonfinite_out",
                     (~torch.isfinite(forecast_output)).sum().float(),
                     on_step=False, on_epoch=True)

        # Entropy statistics (encoder only)
        enc_self = torch.concat(enc_self_ent, dim=0).mean()

        # Entropy regularization
        if self.entropy_regularizer:
            ent_regularizer = 1.0 / enc_self
        else:
            ent_regularizer = 0.0

        # MSE loss masks
        if self.show_trg_max_idx is not None:
            mse_mask = torch.logical_not(trg_nan)

        # (mu, raw_sigma, target) of the predictive distribution, set by the
        # probabilistic branches below and consumed by the CRPS logging
        crps_params = None
        # base (main-loss) NLL WITHOUT the delta increment term, set by the
        # delta-mode branches below: {stage}_nll must reflect the fit quality
        # of the cumulative distribution only, so it stays usable for
        # best-checkpoint selection even if the increment NLL explodes
        nll_metric = None

        # Gaussian / lognormal / student NLL: channels are (mu, raw_sigma)
        if self.config["training"]["loss_fn"] in ("gaussian", "lognormal", "student"):
            mu = forecast_output[:, :, 0]
            raw_sigma = forecast_output[:, :, 1]
            # scalar head: trg_val is (B, L_dec) with the scalar targets in
            # the first output_dim columns -> (B, D). GRU heads predict the
            # full target sequence: keep all positions (NaN = missing,
            # excluded by the loss).
            if self.model.head_type == "scalar":
                trg_scalar = trg_val[:, :self.model.output_dim]
            else:
                trg_scalar = trg_val
            sw = self._train_sample_weights(stage, mu, raw_sigma, trg_scalar,
                                            sample_idx=sample_idx)
            loss = self.loss_fn(mu, raw_sigma, trg_scalar, sample_weight=sw)
            # delta mode: supervise the increments on the decomposed target
            if self.delta_mode:
                nll_metric = loss
                d = self.model.last_step_deltas
                inc_nll = self.delta_loss_fn(d[:, :, 0], d[:, :, 1],
                                             self._increment_targets(Y))
                self.log(f"{stage}_inc_nll", inc_nll, on_step=False,
                         on_epoch=True)
                loss = loss + self.delta_weight * inc_nll
            # log-space losses: report the median in y-space as point forecast
            predicted_value = torch.exp(mu) - self.log_shift if self.log_space_loss else mu
            trg = torch.nan_to_num(trg_scalar)
            mse_mask = torch.logical_not(torch.isnan(trg_scalar))
            # keep the predictive distribution for the CRPS diagnostic
            crps_params = (mu, raw_sigma, trg_scalar)
        # Censored NLL: channels are (mu, raw_sigma), the failure mask
        # is derived from the value distribution (no independent BCE head)
        elif self.censored_loss:
            mu = forecast_output[:, :, 0]
            raw_sigma = forecast_output[:, :, 1]
            # log-space losses: report the median in y-space as point forecast
            point = torch.exp(mu) - self.log_shift if self.log_space_loss else mu

            # scalar head: the target is a D-dimensional vector of scalars
            if self.model.head_type == "scalar":
                # trg_val is (B, L_dec) with the scalar targets in the first
                # output_dim columns; reshape to (B, D) for the loss
                trg_scalar = trg_val[:, :self.model.output_dim]
                sw = self._train_sample_weights(stage, mu, raw_sigma, trg_scalar,
                                                sample_idx=sample_idx)
                loss = self.loss_fn(mu, raw_sigma, trg_scalar, sample_weight=sw)
                predicted_value = point
                trg = torch.nan_to_num(trg_scalar)
                mse_mask = torch.logical_not(torch.isnan(trg_scalar))
                crps_params = (mu, raw_sigma, trg_scalar)
            else:
                # fail mode: termination-aware masks (anchor + boundary
                # survival, interrupted/early tails masked). See
                # CensoredLogNormalNLL.classify_termination.
                masks = None
                if getattr(self.loss_fn, "fail_mode", False):
                    masks = self.loss_fn.classify_termination(
                        trg_val, Y[:, :, self.dec_var_idx], Y[:, :, self.dec_pos_idx],
                        self.loss_fn.expected_len, self.loss_fn.threshold,
                        self.loss_fn.fail_margin)
                    # fail-mode diagnostics: classification balance per
                    # variable sequence (sample, variable) and tail
                    # supervision volume (visible in metrics.csv)
                    self.log(f"{stage}_fail_honest_frac",
                             masks["honest"].nanmean(), on_step=False,
                             on_epoch=True)
                    self.log(f"{stage}_fail_interrupted_frac",
                             masks["interrupted"].nanmean(), on_step=False,
                             on_epoch=True)
                    self.log(f"{stage}_fail_survival_pos",
                             masks["survival"].float().sum(dim=1).mean(),
                             on_step=False, on_epoch=True)
                sw = self._train_sample_weights(stage, mu, raw_sigma, trg_val,
                                                sample_idx=sample_idx, masks=masks)
                # masks are supported only by the fail-mode censored losses
                if masks is not None:
                    loss = self.loss_fn(mu, raw_sigma, trg_val,
                                        sample_weight=sw, masks=masks)
                else:
                    loss = self.loss_fn(mu, raw_sigma, trg_val,
                                        sample_weight=sw)
                # false-failure pressure diagnostic: mean unweighted
                # -log P(y < threshold) on observed positions
                if getattr(self.loss_fn, "bce_on_observed", False) \
                        and hasattr(self.loss_fn, "bce_term"):
                    self.log(f"{stage}_bce",
                             self.loss_fn.bce_term(mu, raw_sigma, trg_val, masks),
                             on_step=False, on_epoch=True)
                # unweighted loss decomposition (cum_*): which term moves
                if hasattr(self.loss_fn, "components"):
                    for k, v in self.loss_fn.components(
                            mu, raw_sigma, trg_val, masks).items():
                        self.log(f"{stage}_cum_{k}", v, on_step=False, on_epoch=True)
                # base NLL (cumulative, incl. penalty/anchor) before the
                # extra objectives below, for checkpoint selection
                nll_metric = loss
                # crossing-time CRPS (distance to failure, see losses.py)
                if self.ttf_weight > 0 and masks is not None:
                    ttf = crossing_time_crps(
                        mu, raw_sigma, masks, self.loss_fn.log_threshold,
                        self.loss_fn.expected_len, eps=self.loss_fn.eps,
                        sample_weight=sw)
                    self.log(f"{stage}_ttf_crps", ttf, on_step=False, on_epoch=True)
                    loss = loss + self.ttf_weight * ttf
                # y-space CRPS on observed positions (bounded, discourages
                # overconfident sigma without the NLL's unbounded error term)
                if self.crps_weight > 0 and self.log_space_loss:
                    sig_c = F.softplus(raw_sigma) + self.loss_fn.eps
                    trg_obs = trg_val if masks is None else torch.where(
                        masks["observed"], trg_val, torch.full_like(trg_val, float("nan")))
                    crps_tr = lognormal_crps(mu, sig_c, trg_obs, shift=self.log_shift)
                    loss = loss + self.crps_weight * crps_tr
                # delta mode: supervise the increments on the decomposed target
                if self.delta_mode:
                    d = self.model.last_step_deltas
                    inc_trg = self._increment_targets(Y)
                    inc_nll = self.delta_loss_fn(d[:, :, 0], d[:, :, 1], inc_trg)
                    self.log(f"{stage}_inc_nll", inc_nll, on_step=False,
                             on_epoch=True)
                    for k, v in self.delta_loss_fn.components(
                            d[:, :, 0], d[:, :, 1], inc_trg).items():
                        self.log(f"{stage}_inc_{k}", v, on_step=False, on_epoch=True)
                    var_min = getattr(self.model, "rnn_delta_var_min", None)
                    if var_min:
                        # fraction of increments whose std sits within 10%
                        # of the floor sqrt(var_min) (raw follows the
                        # sigma = softplus(raw) + eps convention)
                        dsig = F.softplus(d[:, :, 1]) + self.loss_fn.eps
                        ok = ~torch.isnan(inc_trg)
                        if ok.any():
                            self.log(f"{stage}_inc_at_floor",
                                     (dsig[ok] <= 1.1 * var_min ** 0.5).float().mean(),
                                     on_step=False, on_epoch=True)
                    loss = loss + self.delta_weight * inc_nll
                # train total without sample weights / hard mining, comparable
                # to val_loss (detached diagnostic; val/test are unweighted)
                if stage == "train" and sw is not None:
                    with torch.no_grad():
                        self.log("train_nll_unweighted",
                                 self.loss_fn(mu, raw_sigma, trg_val, masks=masks)
                                 if masks is not None else
                                 self.loss_fn(mu, raw_sigma, trg_val),
                                 on_step=False, on_epoch=True)
                predicted_value = point
                # decision-consistent metrics at eval: report the point
                # forecast of the STOPPED sequence (the deployed predictor:
                # absorbing fail-stop + zeros), while the NLL above stays on
                # the raw predictive distribution
                if stage in ("val", "test") and _ft is not None:
                    stopped = self.model._apply_fail_stop(forecast_output, Y)
                    mu_s = stopped[:, :, 0]
                    predicted_value = torch.exp(mu_s) - self.log_shift \
                        if self.log_space_loss else mu_s
                    # fail-stop diagnostics per variable sequence (sample,
                    # variable): fraction stopped by the absorbing "failed"
                    # state (mass-FAILED shows up as -> 1), stopped length
                    # fraction, and predicted-failure vs honest-failure
                    # confusion counts (stopped <=> median crossed)
                    dead_pos = stopped[:, :, 0] != forecast_output[:, :, 0]
                    var_y = torch.nan_to_num(Y[:, :, self.dec_var_idx], nan=0.0).long()
                    valid_y = ~torch.isnan(Y[:, :, self.dec_pos_idx]) & (var_y > 0)
                    n_seq, n_stop, len_frac = 0, 0, []
                    tp = fp = fn = 0
                    for j, v in enumerate(sorted(
                            int(u) for u in torch.unique(var_y[valid_y]).tolist())):
                        m_v = valid_y & (var_y == v)
                        has_v = m_v.any(dim=1)
                        stop_v = (dead_pos & m_v).any(dim=1) & has_v
                        n_seq += int(has_v.sum())
                        n_stop += int(stop_v.sum())
                        len_frac.append(((dead_pos & m_v).sum(dim=1).float()
                                         / m_v.sum(dim=1).clamp_min(1))[has_v])
                        if masks is not None and j < masks["honest"].shape[1]:
                            hon = masks["honest"][:, j] == 1.0
                            tp += int((stop_v & hon).sum())
                            fp += int((stop_v & ~hon).sum())
                            fn += int((~stop_v & hon & has_v).sum())
                    self.log(f"{stage}_failstop_frac", n_stop / max(n_seq, 1),
                             on_step=False, on_epoch=True)
                    self.log(f"{stage}_failstop_len_frac",
                             torch.cat(len_frac).mean() if len_frac
                             else torch.tensor(0.0),
                             on_step=False, on_epoch=True)
                    if masks is not None:
                        for name, val in (("tp", tp), ("fp", fp), ("fn", fn)):
                            self.log(f"{stage}_failstop_{name}", float(val),
                                     on_step=False, on_epoch=True,
                                     reduce_fx="sum")
                trg = torch.nan_to_num(trg_val)
                # metrics on observed (non-censored) positions only
                mse_mask = torch.logical_not(trg_nan)
                # CRPS of the RAW predictive distribution (no fail-stop) on
                # observed positions (NaN targets are skipped by the CRPS);
                # bounded in y-units, usable for best-checkpoint selection
                crps_params = (mu, raw_sigma, trg_val)
        # Handle BCE loss if output has 2 channels (value + breaking region indicator)
        elif forecast_output.size(-1) == 2:
            Y1_hat = forecast_output[:, :, 0]  # Predicted value
            Y2_hat = forecast_output[:, :, 1]  # Breaking region logits

            bce_loss_fn = torch.nn.BCEWithLogitsLoss()
            bce = bce_loss_fn(Y2_hat, trg_nan.float())

            # only valid, non-NaN targets contribute; optional position weighting
            sw = self._train_sample_weights(stage, Y1_hat, trg_val,
                                            sample_idx=sample_idx)
            mse = self.loss_fn(Y1_hat, trg_val, sample_weight=sw)

            loss = mse + self.lam * bce
            predicted_value = Y1_hat
            trg = torch.nan_to_num(trg_val)
        else:
            # Standard MSE loss (only valid, non-NaN targets contribute)
            predicted_value = forecast_output.squeeze(-1)
            trg = torch.nan_to_num(trg_val)

            sw = self._train_sample_weights(stage, predicted_value, trg_val,
                                            sample_idx=sample_idx)
            loss = self.loss_fn(predicted_value, trg_val, sample_weight=sw)

        # Log metrics
        # NOTE: torchmetrics R2Score (<=1.0.x) has a hardcoded atol=1e-4 on
        # rss/tss; for O(1e-3) targets (IST slopes) both fall below it and the
        # metric silently returns its sentinel 1.0. R2 is scale-invariant, so
        # rescale to O(1) before evaluation. mae/rmse are unaffected.
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            if mse_mask is not None:
                p_met, t_met = predicted_value[mse_mask], trg[mse_mask]
            else:
                p_met, t_met = predicted_value.reshape(-1), trg.reshape(-1)
            if name == "r2":
                p_met, t_met = p_met * 1e3, t_met * 1e3
            metric_eval = metric(p_met, t_met)

            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True,
                    prog_bar=(stage == "val"))

        # Log the NLL as a metric for best-checkpoint selection (probabilistic losses)
        if self.config["training"]["loss_fn"] in (
            "gaussian", "gaussian_censored", "lognormal", "lognormal_censored",
            "student", "laplace_censored"
        ):
            self.log(f"{stage}_nll", nll_metric if nll_metric is not None else loss,
                     on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))

        # Log the CRPS: a strictly proper scoring rule (like the NLL) but
        # measured in y-units and bounded as sigma -> 0, where it reduces to
        # the MAE. Unlike val_nll it stays monotone in fit quality when a
        # variance penalty (loss_var_reg) shrinks sigma, which makes it the
        # sound choice for best-checkpoint selection with these losses.
        # See lognormal_crps() in proT/training/losses.py.
        if crps_params is not None:
            mu_c, raw_sigma_c, trg_c = crps_params
            sigma_c = F.softplus(raw_sigma_c) + getattr(self.loss_fn, "eps", 1e-3)
            with torch.no_grad():
                if self.log_space_loss:
                    crps = lognormal_crps(mu_c, sigma_c, trg_c, shift=self.log_shift)
                else:
                    crps = gaussian_crps(mu_c, sigma_c, trg_c)
            self.log(f"{stage}_crps", crps, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))
            # conditioned training: CRPS split into the given prefix
            # (one-step priors before each update) and the free tail
            bound = self.show_trg_max_idx
            if stage == "train" and torch.is_tensor(bound) and bound.dim() > 0:
                self._log_region_crps(stage, mu_c, sigma_c, trg_c, Y, bound)

        # Log entropy statistics
        self.log(f"{stage}_enc_self_entropy", enc_self, on_step=False, on_epoch=True, prog_bar=False)

        # Tail-MAE diagnostic (unweighted, minority high-target samples)
        self._log_tail_mae(stage, predicted_value, trg_val)

        # Add entropy regularization to loss
        loss = loss + self.gamma * ent_regularizer

        return loss, predicted_value, Y

    def on_train_start(self):
        """Compute the tail threshold on the training fold (in-memory tensors)."""
        if self.config["training"].get("loss_hard_mining", False) \
                and self.hard_mining_state is None:
            print("Warning: loss_hard_mining is enabled but no HardMiningState was "
                  "injected (trainer.py does this per fold). Hard mining is OFF.")
        dm = getattr(self.trainer, "datamodule", None)
        train_ds = getattr(dm, "train_ds", None) if dm is not None else None
        if train_ds is None or not hasattr(train_ds, "tensors"):
            return
        targets = train_ds.tensors[1][:, :, self.dec_val_idx]
        s = summarize_target(targets, "max")
        s = s[torch.isfinite(s)]
        if s.numel() > 0:
            self._tail_threshold = torch.quantile(s.float(), self.tail_quantile).item()

    def on_train_epoch_end(self):
        """Advance the hard-mining epoch counter (read by the batch sampler
        when the train dataloader is rebuilt for the next epoch)."""
        if self.hard_mining_state is not None:
            self.hard_mining_state.epoch = self.current_epoch + 1

    def _train_sample_weights(self, stage, *loss_args, sample_idx=None, masks=None):
        """Per-sample loss weights (B, 1) for training; None otherwise.

        Also feeds the hard-mining loss memory (if active) from the per-sample
        losses, so the batch sampler can refresh its hard pool next epoch.
        """
        if stage != "train":
            return None
        hm = self.hard_mining_state
        if self.adaptive_weighter is None and (hm is None or sample_idx is None):
            return None
        per_sample = (self.loss_fn.per_sample(*loss_args, masks=masks)
                      if masks is not None else self.loss_fn.per_sample(*loss_args))
        if hm is not None and sample_idx is not None:
            hm.update(sample_idx.detach().cpu().numpy(),
                      per_sample.detach().cpu().numpy())
        if self.adaptive_weighter is None:
            return None
        w = self.adaptive_weighter(per_sample, self.current_epoch)
        self.adaptive_weighter.update(per_sample)
        self.log("train_sw_mean", w.mean(), on_step=False, on_epoch=True)
        self.log("train_sw_frac_boosted", (w > 1.5).float().mean(),
                 on_step=False, on_epoch=True)
        return w

    def _log_tail_mae(self, stage, predicted_value, target_raw):
        """Unweighted MAE on minority (high-target) samples; train-fold threshold."""
        if stage not in ("val", "test") or self._tail_threshold is None:
            return
        s = summarize_target(target_raw, "max")  # (B,)
        tail = (s >= self._tail_threshold) & torch.isfinite(s)
        valid = ~torch.isnan(target_raw)
        m = valid & tail.unsqueeze(1)
        if not m.any():
            return
        mae_tail = (predicted_value[m] - torch.nan_to_num(target_raw)[m]).abs().mean()
        self.log(f"{stage}_mae_tail", mae_tail, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        """
        Training step with target curriculum.
        Uses automatic PyTorch Lightning optimization.
        """
        # The scalar head has no sequential target to reveal: the adaptive
        # curriculum is only active for the GRU heads.
        if self.model.head_type in ("gru", "gru_per_var"):
            # Check if we should start showing targets
            if self.current_epoch == self.epoch_show_trg:
                self.show_trg_active = True

            # Update target upper bound if in random mode
            if self.current_epoch > self.epoch_show_trg and self.target_show_mode == "random":
                self.show_trg_max_idx = self.curriculum.draw_random_mode()

            # One conditioning fraction per sample, redrawn every step:
            # free-run (f = 0) and conditioned samples share one rollout
            # (>= also re-activates the curriculum after a checkpoint resume)
            if self.target_show_mode == "random_per_sample" \
                    and self.current_epoch >= self.epoch_show_trg:
                self.show_trg_active = True
                Y = batch[1]
                self.show_trg_max_idx = self.curriculum.draw_per_sample(
                    Y.size(0), device=Y.device)
                self.log("show_trg_frac_mean", self.show_trg_max_idx.mean(),
                         on_step=False, on_epoch=True)

            # Update forcing threshold from the annealing schedule
            if self.show_trg_active and self.target_show_mode == "scheduled":
                self.show_trg_max_idx = self.curriculum.draw(self.current_epoch)
                self.log("show_trg_max_idx", float(self.show_trg_max_idx),
                         on_step=False, on_epoch=True, prog_bar=False)

        # Forward step
        loss, _, _ = self._step(batch=batch, stage="train")
        # safety net: a non-finite loss would poison AdamW (NaN grad -> NaN
        # clip coefficient -> all weights NaN). Skip the pathological batch
        # instead (Lightning skips the optimizer step when None is returned).
        if not torch.isfinite(loss):
            print(f"WARNING: non-finite train loss at epoch {self.current_epoch}, "
                  f"batch {batch_idx}; batch skipped.")
            return None
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_after_backward(self):
        """Last line of defense against non-finite gradients.

        Zeroes non-finite gradient ELEMENTS (not the whole step): with a
        single NaN/Inf element the total grad norm is NaN, the clip
        coefficient becomes NaN and AdamW writes NaN into EVERY weight (the
        failure mode that killed run 16140288). This hook runs after backward
        and BEFORE gradient clipping (Lightning order), so the surviving
        finite gradients are still norm-clipped normally.
        """
        n_bad = 0
        for p in self.parameters():
            if p.grad is None:
                continue
            bad = ~torch.isfinite(p.grad)
            n = int(bad.sum())
            if n:
                p.grad[bad] = 0.0
                n_bad += n
        if n_bad:
            if self._trainer is not None:
                self.log("train_grad_nonfinite", float(n_bad),
                         on_step=False, on_epoch=True)
            if self._nonfinite_grad_warns < 20:
                self._nonfinite_grad_warns += 1
                epoch = self._trainer.current_epoch \
                    if self._trainer is not None else "?"
                print(f"WARNING: {n_bad} non-finite gradient elements zeroed "
                      f"(epoch {epoch}); metric: train_grad_nonfinite.")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # conditioned forecasting quality at fixed fractions (diagnostic;
        # best-checkpoint selection keeps using the free-run metrics)
        for frac in self.val_cond_fracs:
            self._log_conditioned_crps("val", batch, frac)
        return loss

    def _region_crps(self, mu, sigma, trg, keep):
        """y-space CRPS restricted to the positions in keep (NaN elsewhere);
        None when the region is empty."""
        if not bool((keep & ~torch.isnan(trg)).any()):
            return None
        t = torch.where(keep, trg, torch.full_like(trg, float("nan")))
        with torch.no_grad():
            if self.log_space_loss:
                return lognormal_crps(mu, sigma, t, shift=self.log_shift)
            return gaussian_crps(mu, sigma, t)

    def _log_region_crps(self, stage, mu, sigma, trg, Y, bound):
        """Log {stage}_crps_prefix / {stage}_crps_tail for a conditioned pass:
        prefix = given positions, tail = hidden positions."""
        hidden = self._build_trg_pos_mask(Y, bound).squeeze(-1)
        for name, keep in (("prefix", ~hidden), ("tail", hidden)):
            c = self._region_crps(mu, sigma, trg, keep)
            if c is not None:
                self.log(f"{stage}_crps_{name}", c, on_step=False, on_epoch=True)

    def _log_conditioned_crps(self, stage, batch, frac):
        """Forward with a fixed conditioning fraction and log the CRPS of the
        free tail as {stage}_crps_cond{pct} (fail-stop disabled, as in _step)."""
        X, Y = batch[0], batch[1]
        _ft = getattr(self.model, "fail_threshold", None)
        self.model.fail_threshold = None
        try:
            out = self.forward(data_input=X, data_trg=Y,
                               show_trg_max_idx=float(frac))[0]
        finally:
            self.model.fail_threshold = _ft
        mu = out[:, :, 0]
        sigma = F.softplus(out[:, :, 1]) + getattr(self.loss_fn, "eps", 1e-3)
        hidden = self._build_trg_pos_mask(Y, float(frac)).squeeze(-1)
        c = self._region_crps(mu, sigma, Y[:, :, self.dec_val_idx], hidden)
        if c is not None:
            self.log(f"{stage}_crps_cond{int(round(100 * frac))}", c,
                     on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, _, _ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure simple AdamW optimizer.
        Optionally with a linear LR ramp on Q/K projections (qk_lr_ramp).

        Returns:
            AdamW optimizer, or ([optimizer], [scheduler]) if qk_lr_ramp is on
        """
        if self.config["training"].get("qk_lr_ramp", False):
            return configure_adamw_with_qk_ramp(
                forecaster=self,
                base_lr=self.config["training"]["lr"],
                weight_decay=self.config["training"].get("weight_decay", 0.01),
                scope=self.config["training"].get("qk_freeze_scope", "all"),
                max_epochs=self.config["training"]["max_epochs"],
                end_frac=self.config["training"].get("qk_lr_ramp_end_frac", 0.95),
                start_factor=self.config["training"].get("qk_lr_ramp_start_factor", 1e-4),
            )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"].get("weight_decay", 0.01)
        )
        return optimizer

    def _build_trg_pos_mask(self, dec_input: torch.Tensor, bound: float) -> torch.Tensor:
        """
        Build the (B, L, 1) bool mask of target positions to hide from the GRU
        (True = value must be predicted, not shown).

        - "index" mode: hide positions with pos > bound (scalar comparison
          against the position channel, original behavior).
        - "fraction" mode: per sample and per target variable, the first
          round(bound * L_v) valid positions are shown, L_v counting observed
          AND censored positions (leak-free, see fraction_reveal_mask);
          bound may be a per-sample (B,) tensor.
        """
        if self.show_trg_units == "fraction":
            return fraction_reveal_mask(dec_input, bound, self.dec_pos_idx,
                                        self.dec_val_idx, self.dec_var_idx)
        return (dec_input[:, :, self.dec_pos_idx] > bound).unsqueeze(-1)
