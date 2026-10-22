"""
Lightning module for the encoder benchmark (see baseline_models.py).

The target is a vector of n_out scalars per sample, read from the target
tensor as Y[:, :n_out, val_idx] (Dyconex seq2slope: slope of SenseA/SenseB;
Ishigami last-value dataset: Y holds one row). Loss: MSE over the observed
(non-NaN) scalars. Training conditions are identical for all encoders.
"""

import pytorch_lightning as pl
import torch
import torchmetrics as tm
from typing import Tuple

from proT.baseline.baseline_models import BenchmarkModel


class BenchmarkForecaster(pl.LightningModule):
    """
    Lightning wrapper for BenchmarkModel (scalar targets, MSE loss).

    Config keys:
        model.kwargs: BenchmarkModel arguments (d_model, comps_embed,
            ds_embed_in, encoder, head)
        data.val_idx: value channel of the target tensor
        training.lr, training.weight_decay (opt.), training.scheduler (opt.)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)        # logs entire resolved config

        self.val_idx = config["data"]["val_idx"]
        self.model = BenchmarkModel(**config["model"]["kwargs"])
        self.n_out = self.model.n_out

        assert config["training"]["loss_fn"] == "mse", "Benchmark supports only 'mse'"

        # metrics (pooled over the n_out scalars)
        self.mae  = tm.MeanAbsoluteError()
        self.rmse = tm.MeanSquaredError(squared=False)
        self.r2   = tm.R2Score()

    def forward(self, x, *args, **kwargs):
        """x: (B, L, D) -> (B, n_out). Extra arguments are ignored."""
        return self.model(x)

    def get_target(self, Y: torch.Tensor) -> torch.Tensor:
        """Scalar targets (B, n_out) from the target tensor."""
        return Y[:, :self.n_out, self.val_idx]

    def _step(self, batch: Tuple[torch.Tensor], stage: str = None):
        X, Y = batch[0], batch[1]
        pred = self.forward(X)                                    # (B, n_out)
        trg = self.get_target(Y)                                  # (B, n_out)
        obs = ~torch.isnan(trg)

        loss = ((pred - torch.nan_to_num(trg)) ** 2)[obs].mean()

        p, t = pred[obs], trg[obs]
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            self.log(f"{stage}_{name}", metric(p, t), on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))
        # per-output MAE (e.g. SenseA / SenseB)
        if self.n_out > 1:
            for k in range(self.n_out):
                m = obs[:, k]
                if m.any():
                    self.log(f"{stage}_mae_out{k}", (pred[m, k] - trg[m, k]).abs().mean(),
                             on_step=False, on_epoch=True)
        return loss, pred, trg
    
    
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

        # gradient clipping is set on the pl.Trainer (training.gradient_clip_val)
        optimizer_config = {"optimizer": opt}

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


# backward-compatible alias (old name used by trainer/predictor)
RNNForecaster = BenchmarkForecaster
