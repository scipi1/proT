"""
Online Target Forecaster with random target feeding and simple AdamW optimization.
Combines curriculum learning with clean, maintainable code.
"""

from typing import Any
import math
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm

from proT.core.model import ProT
from proT.training.losses import (
    PositionWeightedMSE,
    CensoredGaussianNLL,
    CensoredLogNormalNLL,
    CensoredLaplaceNLL,
)
from proT.training.sample_weights import AdaptiveLossWeighter, summarize_target
from proT.training.qk_schedule import configure_adamw_with_qk_ramp
from proT.training.forecasters.target_mask import fraction_reveal_mask


class OnlineTargetForecaster(pl.LightningModule):
    """
    Forecaster with progressive target revelation during training.
    
    This forecaster implements curriculum learning where the decoder is shown
    an increasing portion of the target sequence as training progresses.
    
    Features:
    - Simple AdamW optimizer
    - Progressive target revelation (curriculum learning)
    - BCE loss handling for NaN regions
    - Entropy regularization
    - MSE loss masking
    
    Args:
        config: Configuration dictionary containing:
            - model.kwargs: ProT model parameters
            - training.loss_fn: "mse" (value MSE + BCE mask head),
              "gaussian_censored" (Tobit NLL; head channels are mu and raw
              sigma, the failure mask is derived as ndtr((mu-threshold)/sigma))
              or "lognormal_censored" (lognormal Tobit; mu parameterizes
              log(y + loss_log_shift), the failure mask is derived as
              ndtr((mu - log(threshold + shift))/sigma))
            - training.loss_censor_threshold: Censoring threshold for
              the censored losses (optional, default: 1.0)
            - training.loss_bce_on_observed: Add the threshold-side term on
              observed positions for the censored losses (optional, default: False)
            - training.loss_log_shift: Shift s for "lognormal_censored"
              (optional, default: 1e-3; must satisfy min(target) + s > 0)
            - training.lr: Learning rate for AdamW
            - training.weight_decay: Weight decay (optional, default: 0.01)
            - training.lam: BCE loss weight (optional, default: 1.0)
            - training.gamma: Entropy regularization weight (optional, default: 0.1)
            - training.entropy_regularizer: Enable entropy reg (optional, default: False)
            - data.val_idx: Index of value feature in target
            - data.pos_idx: Index of position feature in target
            - training.target_show_mode: "fixed", "random" or "scheduled" (optional)
            - training.epoch_show_trg: Epoch to start showing targets (optional)
            - training.show_trg_max_idx: Max position to show (fixed mode)
            - training.show_trg_upper_bound_max: Upper bound (random mode)
            - training.show_trg_max_idx_start: Initial forcing bound (scheduled mode)
            - training.show_trg_max_idx_end: Final forcing bound (scheduled mode, default: 0)
            - training.show_trg_schedule: "linear" or "cosine" (scheduled mode, default: "linear")
            - training.show_trg_schedule_end_frac: Fraction of max_epochs at which the
              final bound is reached (scheduled mode, default: 1.0)
            - training.show_trg_random_draw: If True (default), draw the threshold
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
        self.model = ProT(**config["model"]["kwargs"])
        
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
        
        # Flag for the censored-likelihood mode (derived mask, no BCE head)
        self.censored_loss = config["training"]["loss_fn"] in (
            "gaussian_censored", "lognormal_censored", "laplace_censored"
        )
        # Flag for log-space losses: mu parameterizes log(y + shift), so the
        # point prediction in y-space is the median exp(mu) - shift
        self.log_space_loss = config["training"]["loss_fn"] in (
            "lognormal_censored", "laplace_censored"
        )
        self.log_shift = config["training"].get("loss_log_shift", 1e-3)
        
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
        
        # Loss weights
        self.lam = config["training"].get("lam", 1.0)
        self.gamma = config["training"].get("gamma", 0.1)
        self.entropy_regularizer = config["training"].get("entropy_regularizer", False)
        
        # Data indices
        self.dec_val_idx = config["data"]["val_idx"]
        self.dec_pos_idx = config["data"]["pos_idx"]
        
        # Online target configuration
        self.target_show_mode = config["training"].get("target_show_mode", None)
        self.epoch_show_trg = config["training"].get("epoch_show_trg", 0)
        self.show_trg_max_idx = config["training"].get("show_trg_max_idx", None)
        self.show_trg_max_idx_upper_bound = config["training"].get("show_trg_upper_bound_max", None)

        # Scheduled mode configuration (annealed forcing bound)
        self.max_epochs = config["training"]["max_epochs"]
        self.show_trg_max_idx_start = config["training"].get("show_trg_max_idx_start", None)
        self.show_trg_max_idx_end = config["training"].get("show_trg_max_idx_end", 0)
        self.show_trg_schedule = config["training"].get("show_trg_schedule", "linear")
        self.show_trg_schedule_end_frac = config["training"].get("show_trg_schedule_end_frac", 1.0)
        self.show_trg_random_draw = config["training"].get("show_trg_random_draw", True)

        # Units of the forcing bounds: absolute position index (default) or
        # fraction of each sample's valid (non-NaN) target length
        self.show_trg_units = config["training"].get("show_trg_units", "index")

        # Validation
        assert self.show_trg_units in ("index", "fraction"), \
            f"show_trg_units must be 'index' or 'fraction', got {self.show_trg_units}"
        if self.show_trg_units == "fraction":
            for name, val in (
                ("show_trg_max_idx", self.show_trg_max_idx),
                ("show_trg_upper_bound_max", self.show_trg_max_idx_upper_bound),
                ("show_trg_max_idx_start", self.show_trg_max_idx_start),
                ("show_trg_max_idx_end", self.show_trg_max_idx_end),
            ):
                if val is not None:
                    assert 0.0 <= val <= 1.0, \
                        f"{name} must be in [0, 1] when show_trg_units='fraction', got {val}"
        
        # Validation
        if self.target_show_mode == "random":
            assert self.show_trg_max_idx_upper_bound is not None, \
                "show_trg_upper_bound_max must be set for random target mode"
        
        if self.target_show_mode == "scheduled":
            assert self.show_trg_max_idx_start is not None, \
                "show_trg_max_idx_start must be set for scheduled target mode"
            assert self.show_trg_schedule in ("linear", "cosine"), \
                f"show_trg_schedule must be 'linear' or 'cosine', got {self.show_trg_schedule}"
            assert 0.0 < self.show_trg_schedule_end_frac <= 1.0, \
                f"show_trg_schedule_end_frac must be in (0, 1], got {self.show_trg_schedule_end_frac}"
        
        # State tracking
        self.show_trg_active = False
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Metrics
        self.mae = tm.MeanAbsoluteError()
        self.rmse = tm.MeanSquaredError(squared=False)
        self.r2 = tm.R2Score()
        
        print(f"✓ OnlineTargetForecaster initialized")
        print(f"  - Target show mode: {self.target_show_mode}")
        print(f"  - Optimizer: AdamW (lr={config['training']['lr']})")
    
    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor, 
                show_trg_max_idx: float = None, predict_mode: bool = False) -> Any:
        """
        Forward pass with optional target revealing.
        
        Args:
            data_input: Encoder input tensor (B, L_enc, D_enc)
            data_trg: Decoder target tensor (B, L_dec, D_dec)
            show_trg_max_idx: Maximum position index to reveal (or fraction of
                each sample's valid length when show_trg_units="fraction")
            predict_mode: If True, use show_trg_max_idx from config
            
        Returns:
            Tuple of (forecast_output, recon_output, attention_weights, masks, entropy)
        """
        encoder_in = data_input
        dec_input = data_trg.clone()
        
        # Determine how much target to show
        if show_trg_max_idx is None:
            if self.show_trg_max_idx is not None and (self.show_trg_active or predict_mode):
                show_trg_max_idx = self.show_trg_max_idx
            else:
                show_trg_max_idx = 0.0
        
        # Create mask for target positions
        if show_trg_max_idx is not None and show_trg_max_idx > 0:
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
        # This allows the embedding system to embed it based on config
        dec_input = torch.cat([dec_input, given_indicator], dim=-1)  # B x L x (D+1)
        
        # Forward pass
        model_output, recon_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, \
            (enc_self_ent, dec_self_ent, dec_cross_ent) = self.model.forward(
                input_tensor=encoder_in,
                target_tensor=dec_input,
                trg_pos_mask=trg_pos_mask
            )
        
        return model_output, recon_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, \
            (enc_self_ent, dec_self_ent, dec_cross_ent)
    
    def _step(self, batch, stage: str = None):
        """
        Common step for train/val/test.
        
        Args:
            batch: Tuple of (input_data, target_data)
            stage: One of "train", "val", "test"
            
        Returns:
            Tuple of (loss, predicted_values, target_data)
        """
        X, Y = batch
        trg_val = Y[:, :, self.dec_val_idx]
        trg_nan = trg_val.isnan()
        mse_mask = None
        
        # Forward pass. In scheduled mode, val/test are always evaluated without
        # teacher forcing (the real prediction task), so metrics are comparable
        # across epochs and best-checkpoint selection is unbiased.
        if stage in ("val", "test") and self.target_show_mode == "scheduled":
            forecast_output, _, _, _, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(
                data_input=X, data_trg=Y, show_trg_max_idx=0.0
            )
        else:
            forecast_output, _, _, _, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(
                data_input=X, data_trg=Y
            )
        
        # Entropy statistics
        enc_self = torch.concat(enc_self_ent, dim=0).mean()
        dec_self = torch.concat(dec_self_ent, dim=0).mean()
        dec_cross = torch.concat(dec_cross_ent, dim=0).mean()
        
        # Entropy regularization
        if self.entropy_regularizer:
            ent_regularizer = 1.0 / enc_self + 1.0 / dec_self + 1.0 / dec_cross
        else:
            ent_regularizer = 0.0
        
        # MSE loss masks
        if self.show_trg_max_idx is not None:
            mse_mask = torch.logical_not(trg_nan)
        
        # Censored NLL: channels are (mu, raw_sigma), the failure mask
        # is derived from the value distribution (no independent BCE head)
        if self.censored_loss:
            mu = forecast_output[:, :, 0]
            raw_sigma = forecast_output[:, :, 1]
            
            sw = self._train_sample_weights(stage, mu, raw_sigma, trg_val)
            loss = self.loss_fn(mu, raw_sigma, trg_val, sample_weight=sw)
            # log-space losses: report the median in y-space as point forecast
            predicted_value = torch.exp(mu) - self.log_shift if self.log_space_loss else mu
            trg = torch.nan_to_num(trg_val)
            # metrics on observed (non-censored) positions only
            mse_mask = torch.logical_not(trg_nan)
        # Handle BCE loss if output has 2 channels (value + breaking region indicator)
        elif forecast_output.size(-1) == 2:
            Y1_hat = forecast_output[:, :, 0]  # Predicted value
            Y2_hat = forecast_output[:, :, 1]  # Breaking region logits
            
            bce_loss_fn = torch.nn.BCEWithLogitsLoss()
            bce = bce_loss_fn(Y2_hat, trg_nan.float())
            
            # only valid, non-NaN targets contribute; optional position weighting
            sw = self._train_sample_weights(stage, Y1_hat, trg_val)
            mse = self.loss_fn(Y1_hat, trg_val, sample_weight=sw)
            
            loss = mse + self.lam * bce
            predicted_value = Y1_hat
            trg = torch.nan_to_num(trg_val)
        else:
            # Standard MSE loss (only valid, non-NaN targets contribute)
            predicted_value = forecast_output.squeeze(-1)
            trg = torch.nan_to_num(trg_val)
            
            sw = self._train_sample_weights(stage, predicted_value, trg_val)
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
        
        # Log entropy statistics
        for name, value in [("enc_self_entropy", enc_self), ("dec_self_entropy", dec_self), 
                           ("dec_cross_entropy", dec_cross)]:
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=False)
        
        # Tail-MAE diagnostic (unweighted, minority high-target samples)
        self._log_tail_mae(stage, predicted_value, trg_val)

        # Add entropy regularization to loss
        loss = loss + self.gamma * ent_regularizer
        
        return loss, predicted_value, Y
    
    def on_train_start(self):
        """Compute the tail threshold on the training fold (in-memory tensors)."""
        dm = getattr(self.trainer, "datamodule", None)
        train_ds = getattr(dm, "train_ds", None) if dm is not None else None
        if train_ds is None or not hasattr(train_ds, "tensors"):
            return
        targets = train_ds.tensors[1][:, :, self.dec_val_idx]
        s = summarize_target(targets, "max")
        s = s[torch.isfinite(s)]
        if s.numel() > 0:
            self._tail_threshold = torch.quantile(s.float(), self.tail_quantile).item()

    def _train_sample_weights(self, stage, *loss_args):
        """Per-sample loss weights (B, 1) for training; None otherwise."""
        if stage != "train" or self.adaptive_weighter is None:
            return None
        per_sample = self.loss_fn.per_sample(*loss_args)
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
        # Check if we should start showing targets
        if self.current_epoch == self.epoch_show_trg:
            self.show_trg_active = True
        
        # Update target upper bound if in random mode
        if self.current_epoch > self.epoch_show_trg and self.target_show_mode == "random":
            self._update_target_upper_bound()
        
        # Update forcing threshold from the annealing schedule
        if self.show_trg_active and self.target_show_mode == "scheduled":
            self._update_scheduled_bound()
            self.log("show_trg_max_idx", float(self.show_trg_max_idx),
                     on_step=False, on_epoch=True, prog_bar=False)
        
        # Forward step
        loss, _, _ = self._step(batch=batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
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
        Build the (B, L, 1) bool mask of target positions to hide from the
        decoder (True = value must be predicted, not shown).

        - "index" mode: hide positions with pos > bound (scalar comparison
          against the position channel, original behavior).
        - "fraction" mode: per sample and per target variable, the first
          round(bound * L_v) valid positions are shown, L_v counting observed
          AND censored positions (leak-free, see fraction_reveal_mask).
        """
        if self.show_trg_units == "fraction":
            return fraction_reveal_mask(
                dec_input, bound, self.dec_pos_idx, self.dec_val_idx,
                self.config["model"]["kwargs"].get("dec_var_idx", None))
        return (dec_input[:, :, self.dec_pos_idx] > bound).unsqueeze(-1)

    def _cast_bound(self, bound: float):
        """Fraction mode keeps float bounds in [0, 1]; index mode rounds to int."""
        if self.show_trg_units == "fraction":
            return float(bound)
        return int(round(bound))

    def _update_target_upper_bound(self):
        """Update target upper bound for random mode."""
        if self.target_show_mode == "random":
            if self.show_trg_units == "fraction":
                self.show_trg_max_idx = random.uniform(0.0, self.show_trg_max_idx_upper_bound)
            else:
                self.show_trg_max_idx = random.randint(0, self.show_trg_max_idx_upper_bound)

    def _scheduled_bound(self, epoch: int) -> int:
        """
        Forcing upper bound at a given epoch under the annealing schedule.

        The bound stays at show_trg_max_idx_start until epoch_show_trg, then
        anneals (linear or cosine) to show_trg_max_idx_end, reached at
        epoch = show_trg_schedule_end_frac * max_epochs.
        """
        start_ep = self.epoch_show_trg
        end_ep = min(
            self.max_epochs - 1,
            int(round(self.show_trg_schedule_end_frac * self.max_epochs))
        )

        if epoch <= start_ep:
            return self._cast_bound(self.show_trg_max_idx_start)
        if epoch >= end_ep:
            return self._cast_bound(self.show_trg_max_idx_end)

        frac = (epoch - start_ep) / max(1, end_ep - start_ep)
        if self.show_trg_schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0
        else:  # linear
            w = 1.0 - frac

        return self._cast_bound(
            self.show_trg_max_idx_end
            + (self.show_trg_max_idx_start - self.show_trg_max_idx_end) * w
        )

    def _update_scheduled_bound(self):
        """Update the forcing threshold for scheduled mode (optionally random)."""
        bound = self._scheduled_bound(self.current_epoch)
        if self.show_trg_random_draw and bound > 0:
            if self.show_trg_units == "fraction":
                bound = random.uniform(0.0, bound)
            else:
                bound = random.randint(0, bound)
        self.show_trg_max_idx = bound
