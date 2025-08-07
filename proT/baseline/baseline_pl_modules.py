import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.baseline.baseline_models import RNN, TCN, MLP


class RNNForecaster(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.val_idx = config["data"]["val_idx"]  # value feature index
        
        # import model
        if config["model"]["model_object"] in ["GRU", "LSTM"]:
            self.model = RNN(**config["model"]["kwargs"])
            
        elif config["model"]["model_object"] == "TCN":
            self.model = TCN(**config["model"]["kwargs"])
            
        elif config["model"]["model_object"] == "MLP":
            self.model = MLP(**config["model"]["kwargs"])
        
        
        
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
    
    
    def _step(self, batch, stage: str=None):
        
        X, Y = batch
        
        input_tensor = X
        
        # extract target features and zero the value
        masked_target = Y.clone()
        masked_target[:,:,self.val_idx] = 0
        
        predict_out = self.forward(x=input_tensor, y=masked_target)
        
        trg = torch.nan_to_num(Y[:,:,self.val_idx])
        mse_per_elem = self.loss_fn(predict_out, trg)
        loss = mse_per_elem.mean()
        
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
    
    
    def configure_optimizers(self):
        optim_cfg = self.hparams.get("training", {})
        opt = torch.optim.AdamW(
            self.parameters(), # applied to all model parameters
            lr           = optim_cfg.get("lr", 1e-4),
            weight_decay = optim_cfg.get("weight_decay", 0.0),
            betas        = optim_cfg.get("betas", (0.9, 0.999)),
        )

        sched_cfg = optim_cfg.get("scheduler")
        if sched_cfg:
            sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size = sched_cfg.get("step_size", 10),
                gamma     = sched_cfg.get("gamma", 0.1),
            )
            return {
                "optimizer":   opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval":  "epoch",
                    "monitor":   "val_loss",
                },
            }

        return opt

        
