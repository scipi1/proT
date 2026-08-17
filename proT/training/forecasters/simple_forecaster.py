"""
Simple Transformer Forecaster with clean AdamW optimizer.
This is the base forecaster with minimal complexity - suitable for most use cases.
"""

from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm

from proT.core.model import ProT
from proT.training.losses import PositionWeightedMSE
from proT.training.qk_schedule import configure_adamw_with_qk_ramp


class SimpleForecaster(pl.LightningModule):
    """
    Simple transformer forecaster with standard PyTorch Lightning training.
    Uses AdamW optimizer with no complex optimization schemes.
    
    This is the recommended forecaster for most use cases.
    
    Args:
        config: Configuration dictionary containing:
            - model.kwargs: ProT model parameters
            - training.loss_fn: Loss function name (e.g., "mse")
            - training.lr: Learning rate for AdamW
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
        Common step for train/val/test.
        
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
        
        # Log metrics
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            metric_eval = metric(predicted_value.reshape(-1), trg.reshape(-1))
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log entropy statistics
        enc_self = torch.concat(enc_self_ent, dim=0).mean()
        dec_self = torch.concat(dec_self_ent, dim=0).mean()
        dec_cross = torch.concat(dec_cross_ent, dim=0).mean()
        
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
