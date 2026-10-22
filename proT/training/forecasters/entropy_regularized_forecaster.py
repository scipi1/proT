"""
Simple Transformer Forecaster with Entropy Regularization.
This forecaster extends the SimpleForecaster by adding entropy regularization to the loss function.
"""

from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm

from proT.core.model import ProT
from proT.training.losses import PositionWeightedMSE
from proT.training.qk_schedule import configure_adamw_with_qk_ramp


class EntropyRegularizedForecaster(pl.LightningModule):
    """
    Simple transformer forecaster with entropy regularization.
    Extends SimpleForecaster by adding an entropy regularization term to encourage
    higher entropy in attention distributions.
    
    The entropy regularization term is: gamma * (1/enc_self_entropy + 1/dec_self_entropy + 1/dec_cross_entropy)
    This penalizes low entropy (peaked attention) and encourages more distributed attention patterns.
    
    Args:
        config: Configuration dictionary containing:
            - model.kwargs: ProT model parameters
            - training.loss_fn: Loss function name (e.g., "mse")
            - training.lr: Learning rate for AdamW
            - training.gamma: Weight for entropy regularization (default: 1E-3)
            - training.entropy_regularizer: Boolean flag to enable/disable entropy regularization (default: False)
            - data.val_idx: Index of value feature in target tensor
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
        else:
            raise ValueError(f"Unsupported loss function: {config['training']['loss_fn']}")
        
        # Data indices
        self.dec_val_idx = config["data"]["val_idx"]
        
        # Entropy regularization parameters
        self.gamma = config["training"].get("gamma", 1E-3)
        self.entropy_regularizer = config["training"].get("entropy_regularizer", False)
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Metrics
        self.mae = tm.MeanAbsoluteError()
        self.rmse = tm.MeanSquaredError(squared=False)
        self.r2 = tm.R2Score()
    
    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor) -> Any:
        """
        Forward pass through the model.
        
        Args:
            data_input: Encoder input tensor (B, L_enc, D_enc)
            data_trg: Decoder target tensor (B, L_dec, D_dec)
            
        Returns:
            Tuple of (forecast_output, recon_output, attention_weights, masks, entropy)
        """
        # Prepare decoder input: zero out target values
        dec_input = data_trg.clone()
        dec_input[:, :, self.dec_val_idx] = 0.0
        
        # Forward pass
        model_output, recon_output, attn_weights, enc_mask, entropy = self.model.forward(
            input_tensor=data_input,
            target_tensor=dec_input,
            trg_pos_mask=None
        )
        
        return model_output, recon_output, attn_weights, enc_mask, entropy
    
    def _step(self, batch, stage: str = None):
        """
        Common step for train/val/test with entropy regularization.
        
        Args:
            batch: Tuple of (input_data, target_data)
            stage: One of "train", "val", "test"
            
        Returns:
            Tuple of (loss, predicted_values, target_data)
        """
        X, Y = batch
        trg_val = Y[:, :, self.dec_val_idx]
        
        # Forward pass
        forecast_output, _, _, _, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(X, Y)
        
        # Calculate loss (only valid, non-NaN targets contribute; optional position weighting)
        predicted_value = forecast_output.squeeze(-1)
        trg = torch.nan_to_num(trg_val)
        
        loss = self.loss_fn(predicted_value, trg_val)
        
        # Calculate entropy statistics
        enc_self = torch.concat(enc_self_ent, dim=0).mean()
        dec_self = torch.concat(dec_self_ent, dim=0).mean()
        dec_cross = torch.concat(dec_cross_ent, dim=0).mean()
        
        # Apply entropy regularization if enabled
        if self.entropy_regularizer:
            # Protect against NaN and division by zero
            ent_regularizer = 0.0
            if not torch.isnan(enc_self) and enc_self > 1e-8:
                ent_regularizer += 1.0/enc_self
            if not torch.isnan(dec_self) and dec_self > 1e-8:
                ent_regularizer += 1.0/dec_self
            if not torch.isnan(dec_cross) and dec_cross > 1e-8:
                ent_regularizer += 1.0/dec_cross
            
            loss = loss + self.gamma * ent_regularizer
        
        # Log metrics
        # NOTE: torchmetrics R2Score (<=1.0.x) has a hardcoded atol=1e-4 on
        # rss/tss; for O(1e-3) targets (IST slopes) both fall below it and the
        # metric silently returns its sentinel 1.0. R2 is scale-invariant, so
        # rescale to O(1) before evaluation. mae/rmse are unaffected.
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            p_met, t_met = predicted_value.reshape(-1), trg.reshape(-1)
            if name == "r2":
                p_met, t_met = p_met * 1e3, t_met * 1e3
            metric_eval = metric(p_met, t_met)
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log entropy statistics
        for name, value in [("enc_self_entropy", enc_self), ("dec_self_entropy", dec_self), ("dec_cross_entropy", dec_cross)]:
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss, predicted_value, Y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
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
        Configure AdamW optimizer - simple and effective.
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
