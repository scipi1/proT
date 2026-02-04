import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
from typing import Tuple
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.baseline.baseline_models import RNN, TCN, MLP, S6


class RNNForecaster(pl.LightningModule):
    """
    PyTorch Lightning module for baseline sequence models (LSTM, GRU, TCN, MLP, S6).
    
    Wraps various recurrent and sequential architectures for time series forecasting
    with automatic training, validation, and testing loops.
    
    Supports:
        - LSTM: Long Short-Term Memory network
        - GRU: Gated Recurrent Unit network
        - TCN: Temporal Convolutional Network
        - MLP: Multi-Layer Perceptron
        - S6: State Space Sequence model (Mamba-style)
    
    Args:
        config: Configuration dictionary containing model architecture and training settings
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self.val_idx = config["data"]["val_idx"]  # value feature index
        
        # import model
        if config["model"]["model_object"] in ["GRU", "LSTM"]:
            self.model = RNN(**config["model"]["kwargs"])
            
        elif config["model"]["model_object"] == "TCN":
            self.model = TCN(**config["model"]["kwargs"])
            
        elif config["model"]["model_object"] == "MLP":
            self.model = MLP(**config["model"]["kwargs"])
        
        elif config["model"]["model_object"] == "S6":
            self.model = S6(**config["model"]["kwargs"])
        
        
        # define loss
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        
        self.save_hyperparameters(config)        # logs entire resolved config
        
        # metrics
        self.mae   = tm.MeanAbsoluteError()
        self.rmse  = tm.MeanSquaredError(squared=False)
        self.r2    = tm.R2Score() 
    
    
    def forward(self, x, y, **kwargs):
        return self.model(x, y, **kwargs)
    
    
    def _step(self, batch: Tuple[torch.Tensor], stage: str=None):
        
        X, Y = batch
        
        input_tensor = X
        
        # extract target features and zero the value
        masked_target = Y.clone()
        masked_target[:,:,self.val_idx] = 0
        predict_out = self.forward(x=input_tensor, y=masked_target)
        
        # Check for NaN in predictions
        if torch.isnan(predict_out).any():
            self.log(f"{stage}_nan_detected", 1.0, on_step=True, on_epoch=False)
            # Replace NaN predictions with zeros to prevent loss explosion
            predict_out = torch.nan_to_num(predict_out, nan=0.0)
        
        trg = torch.nan_to_num(Y[:,:,self.val_idx])
        
        # Additional safety: clip extremely large values that might lead to NaN
        predict_out = torch.clamp(predict_out, min=-1e6, max=1e6)
        
        mse_per_elem = self.loss_fn(predict_out, trg)
        
        # Check for NaN in loss
        if torch.isnan(mse_per_elem).any():
            self.log(f"{stage}_loss_nan_detected", 1.0, on_step=True, on_epoch=False)
            mse_per_elem = torch.nan_to_num(mse_per_elem, nan=1e6)  # Large penalty for NaN
        
        loss = mse_per_elem.mean()
        
        # Final check: if loss itself is NaN, replace with large value
        if torch.isnan(loss):
            self.log(f"{stage}_final_loss_nan", 1.0, on_step=True, on_epoch=False)
            loss = torch.tensor(1e6, device=loss.device, requires_grad=True)
        
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            metric_eval = metric(predict_out.reshape(-1)  , trg.reshape(-1)  )                         
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        return loss, predict_out, trg
    
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch,stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss,_,_ = self._step(batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss,_,_ = self._step(batch,stage="test")
        self.log("test_loss", loss)
        return loss
    
    
    def on_before_optimizer_step(self, optimizer):
        """
        Optional: Monitor gradients for NaN/Inf values before optimizer step.
        This helps identify which parameters are causing numerical issues.
        """
        # Check for NaN or Inf in gradients
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    valid_gradients = False
                    # Log which parameter has invalid gradients (useful for debugging)
                    grad_norm = param.grad.norm().item()
                    self.log(f"grad_norm_{name.replace('.', '_')}", grad_norm, on_step=True, on_epoch=False)
        
        if not valid_gradients:
            self.log("invalid_gradients_detected", 1.0, on_step=True, on_epoch=False)
    
    
    def configure_optimizers(self):
        optim_cfg = self.hparams.get("training", {})
        opt = torch.optim.AdamW(
            self.parameters(), # applied to all model parameters
            lr           = optim_cfg.get("lr", 1e-4),
            weight_decay = optim_cfg.get("weight_decay", 0.0),
            betas        = optim_cfg.get("betas", (0.9, 0.999)),
        )

        # Add gradient clipping configuration
        optimizer_config = {
            "optimizer": opt,
            "gradient_clip_val": optim_cfg.get("gradient_clip_val", 1.0),  # default max_norm=1.0
            "gradient_clip_algorithm": "norm"
        }

        sched_cfg = optim_cfg.get("scheduler")
        if sched_cfg:
            sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size = sched_cfg.get("step_size", 10),
                gamma     = sched_cfg.get("gamma", 0.1),
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": sched,
                "interval":  "epoch",
                "monitor":   "val_loss",
            }

        return optimizer_config
