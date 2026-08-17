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
from proT.training.losses import PositionWeightedMSE, CensoredGaussianNLL
from proT.training.qk_schedule import configure_adamw_with_qk_ramp


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
            - training.loss_fn: "mse" (value MSE + BCE mask head) or
              "gaussian_censored" (Tobit NLL; head channels are mu and raw
              sigma, the failure mask is derived as ndtr((mu-threshold)/sigma))
            - training.loss_censor_threshold: Censoring threshold for
              "gaussian_censored" (optional, default: 1.0)
            - training.loss_bce_on_observed: Add the threshold-side term on
              observed positions for "gaussian_censored" (optional, default: False)
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
            )
        else:
            raise ValueError(f"Unsupported loss function: {config['training']['loss_fn']}")
        
        # Flag for the censored-likelihood mode (derived mask, no BCE head)
        self.censored_loss = config["training"]["loss_fn"] == "gaussian_censored"
        
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
            show_trg_max_idx: Maximum position index to reveal
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
            # Mask along sequence dimension based on position
            trg_pos_mask = (dec_input[:, :, self.dec_pos_idx] > show_trg_max_idx).unsqueeze(-1)
            
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
        
        # Censored Gaussian NLL: channels are (mu, raw_sigma), the failure mask
        # is derived from the value distribution (no independent BCE head)
        if self.censored_loss:
            mu = forecast_output[:, :, 0]
            raw_sigma = forecast_output[:, :, 1]
            
            loss = self.loss_fn(mu, raw_sigma, trg_val)
            predicted_value = mu
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
            mse = self.loss_fn(Y1_hat, trg_val)
            
            loss = mse + self.lam * bce
            predicted_value = Y1_hat
            trg = torch.nan_to_num(trg_val)
        else:
            # Standard MSE loss (only valid, non-NaN targets contribute)
            predicted_value = forecast_output.squeeze(-1)
            trg = torch.nan_to_num(trg_val)
            
            loss = self.loss_fn(predicted_value, trg_val)
        
        # Log metrics
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            if mse_mask is not None:
                metric_eval = metric(predicted_value[mse_mask], trg[mse_mask])
            else:
                metric_eval = metric(predicted_value.reshape(-1), trg.reshape(-1))
            
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, 
                    prog_bar=(stage == "val"))
        
        # Log entropy statistics
        for name, value in [("enc_self_entropy", enc_self), ("dec_self_entropy", dec_self), 
                           ("dec_cross_entropy", dec_cross)]:
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=False)
        
        # Add entropy regularization to loss
        loss = loss + self.gamma * ent_regularizer
        
        return loss, predicted_value, Y
    
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
    
    def _update_target_upper_bound(self):
        """Update target upper bound for random mode."""
        if self.target_show_mode == "random":
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
            return int(self.show_trg_max_idx_start)
        if epoch >= end_ep:
            return int(self.show_trg_max_idx_end)

        frac = (epoch - start_ep) / max(1, end_ep - start_ep)
        if self.show_trg_schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0
        else:  # linear
            w = 1.0 - frac

        return int(round(
            self.show_trg_max_idx_end
            + (self.show_trg_max_idx_start - self.show_trg_max_idx_end) * w
        ))

    def _update_scheduled_bound(self):
        """Update the forcing threshold for scheduled mode (optionally random)."""
        bound = self._scheduled_bound(self.current_epoch)
        if self.show_trg_random_draw and bound > 0:
            bound = random.randint(0, bound)
        self.show_trg_max_idx = bound
