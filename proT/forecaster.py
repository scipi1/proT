# Standard library imports
import sys
from os.path import dirname, abspath
import random
from typing import Any


# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm

# Local imports
ROOT_DIR = dirname(abspath(__file__))
sys.path.append(ROOT_DIR)
from proT.proT_model import ProT
from proT.simulator.trajectory_simulator import ISTSimulator



class TransformerForecaster(pl.LightningModule):
    """
    Assembles a model according to the config file
    into a LightningModule for training

    Args:
        config: configuration file
    """
    def __init__(self, config):
        super().__init__()
        
        self.automatic_optimization = False
        
        self.config = config
        self.model = ProT(**config["model"]["kwargs"])
        
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
            
        # for BCE loss
        self.lam = config["training"].get("lam", 1.0)
            
        self.dec_val_idx = config["data"]["val_idx"]
        self.dec_pos_idx = config["data"]["pos_idx"]
        self.switch_epoch = config["training"]["switch_epoch"]
        self.switch_step = config["training"]["switch_step"]
        
        # show target
        self.target_show_mode = config["training"].get("target_show_mode", None)
        self.epoch_show_trg = config["training"].get("epoch_show_trg", None)
        self.show_trg_max_idx = config["training"].get("show_trg_max_idx", None)
        self.show_trg_max_idx_upper_bound = config["training"].get("show_trg_max_idx_upper_bound", None)
        
        if self.target_show_mode == "random":
            assert self.show_trg_max_idx_upper_bound is not None, AssertionError("show_trg_upper_bound_max must be set for random target mode") 
        
        # entropy regularizer
        self.gamma = self.lam = config["training"].get("gamma", 0.1)
        self.entropy_regularizer = config["training"].get("entropy_regularizer", False)
        
        # simulator
        if config["training"].get("pinn",False):
            self.decoder_input_module = ISTSimulator(model="F").get_decoder_input
            self.trajectory_simulator = ISTSimulator(model="F").forward
        else:
            self.decoder_input_module = None 
            self.trajectory_simulator = None

        
        # keep false
        self._switched    = False
        self.schedulers_flag = False
        self.show_trg_ = False 
            
        self.save_hyperparameters(config)
        
        # metrics
        self.mae   = tm.MeanAbsoluteError()
        self.rmse  = tm.MeanSquaredError(squared=False)
        self.r2    = tm.R2Score() 
        
        
    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor, show_trg_max_idx: float=0.0, predict_mode=False) -> Any:
        
        encoder_in = data_input
                
        if self.decoder_input_module is not None:
            dec_input = self.decoder_input_module(batch_size=data_trg.size(0), device=data_input.get_device())
        
        else:
            # set values of target sequence to zero
            dec_input = data_trg.clone()
            
            # get the upper bound to show the target sequence to the decoder
            if show_trg_max_idx is None:
                if (self.show_trg_max_idx is not None and (self.show_trg_ == True or predict_mode == True)):
                    show_trg_max_idx = self.show_trg_max_idx
                else:
                    show_trg_max_idx = 0.0
                
                
            if show_trg_max_idx is not None:
                # create a mask along L dimension for values that are above the upper bound
                trg_pos_mask = (dec_input[:,:,3] > show_trg_max_idx).unsqueeze(-1) # B x L x 1 #TODO remove hardcoded index
                
                
                # create a feature mask for val_idx: only the values must be set to zero
                feature_mask = torch.zeros(dec_input.size(-1), dtype=torch.bool, device=dec_input.device)
                feature_mask[self.dec_val_idx] = True
                
                # apply combined mask
                combined_mask = trg_pos_mask & feature_mask.unsqueeze(0).unsqueeze(0)
                dec_input[combined_mask] = 0

            else:
                dec_input[:,:, self.dec_val_idx] = 0.0
                trg_pos_mask = None
            
        
        # TODO recon_output still needed?
        model_output, recon_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.model.forward(
            input_tensor=encoder_in,
            target_tensor=dec_input,
            trg_pos_mask=trg_pos_mask)
        
        if self.trajectory_simulator is not None:
            forecast_output = self.trajectory_simulator(model_output)
        else:
            forecast_output = model_output
        
        return forecast_output, recon_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent)
    
    
    
    def _step(self, batch, stage: str=None):
        
        X, Y = batch
        trg_val = Y[:,:,self.dec_val_idx]
        trg_nan = trg_val.isnan()
        mse_mask = None
        
        forecast_output,_,_,_, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(data_input=X, data_trg=Y)
        
        # Entropy regularization
        enc_self = torch.concat(enc_self_ent, dim=0).mean()
        dec_self = torch.concat(dec_self_ent, dim=0).mean()
        dec_cross = torch.concat(dec_cross_ent, dim=0).mean()
            
        if self.entropy_regularizer:
            ent_regularizer = 1.0/enc_self + 1.0/dec_self + 1.0/dec_cross
        else:
            ent_regularizer = 0.0
            
        # MSE loss masks: calculate only on valid values
        if self.show_trg_max_idx is not None:
            trg_pos_mask = (Y[:,:,self.dec_pos_idx] > self.show_trg_max_idx)
            mse_mask = torch.logical_not(trg_nan) #torch.logical_and(torch.logical_not(trg_nan)),trg_pos_mask) # B x L
        
        if forecast_output.size(-1) == 2:
            # values and logits
            Y1_hat = forecast_output[:,:,0]  # IST value
            Y2_hat = forecast_output[:,:,1] # breaking region
            
            bce_loss_fn = torch.nn.BCEWithLogitsLoss()
            bce = bce_loss_fn(Y2_hat,trg_nan.float())
            mse = self.loss_fn(Y1_hat[mse_mask], trg_val[mse_mask]).mean()
            
            loss = mse+self.lam*bce
            
            predicted_value = Y1_hat
            trg = torch.nan_to_num(trg_val)
        else:
            predicted_value = forecast_output
            trg = torch.nan_to_num(trg_val)
            
            mse_per_elem  = self.loss_fn(predicted_value.squeeze(), trg.squeeze())
        
            loss = mse_per_elem.mean()
        
        
        # save metrics
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            
            if mse_mask is not None:
                metric_eval = metric(predicted_value[mse_mask], trg[mse_mask])
            else:
                metric_eval = metric(predicted_value.reshape(-1)  , trg.reshape(-1)  )    
            
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
            
        for name, value in [("enc_self_entropy", enc_self), ("dec_self_entropy", dec_self), ("dec_cross_entropy", dec_cross)]:
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
            
            
        # Log entropy regularization
        loss = loss + self.gamma*ent_regularizer
            
        return loss, predicted_value, Y
    
    
    def training_step(self, batch, batch_idx):
        
        # check if epoch is epoch_switch
        if self.current_epoch == self.epoch_show_trg:
            self.show_trg_ = True
        
        if self.current_epoch > self.epoch_show_trg:
            self._update_target_upper_bound()
        
        # forward step
        loss,_,_ = self._step(batch=batch, stage="train")
        
        # one backward pass for all parameters
        self.manual_backward(loss)
        
        # gradient clipping
        if self.config["training"]["optimization"] in [1,2]:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        opt_emb_now, opt_model_now, opt_emb_switch, opt_model_switch = self.optimizers()
        
        
        if self.lr_schedulers() is not None:
            model_scheduler_now, model_scheduler_switch = self.lr_schedulers()
            self.schedulers_flag = True
        
        if (not self._switched) and (self.current_epoch >= self.config["training"]["switch_epoch"]):
            
            if (not self._switched) and (self.current_epoch >= self.switch_epoch):
                
                # switch to p2 optimizer
                self.optimizers()[0] = opt_emb_switch
                self.optimizers()[1] = opt_model_switch
                
                if self.schedulers_flag:
                    self.lr_schedulers()[0] = model_scheduler_switch
                
                self._switched = True
            
        # optimizer steps & zero_grad
        opt_emb_now.step()            
        opt_model_now.step()
        
        if self.schedulers_flag:
            model_scheduler_now.step()
            
        opt_emb_now.zero_grad(set_to_none=True)
        opt_model_now.zero_grad(set_to_none=True)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss,_,_ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss,_,_ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss)
        return loss
    
    
    
    def split_params(self):
        enc_emb_params = list(self.model.enc_embedding.embed_modules_list.parameters())
        dec_emb_params = list(self.model.dec_embedding.embed_modules_list.parameters())
        emb_param_ids = {id(p) for p in enc_emb_params + dec_emb_params}
        other_params = [p for p in self.model.parameters() if id(p) not in emb_param_ids]
        
        group_1 = enc_emb_params[:2]
        group_2 = other_params + dec_emb_params + enc_emb_params[2:]
        
        return group_1, group_2
    
    
    
    def configure_optimizers(self):
        
        assert self.config["training"]["optimization"] in [1,2,3,4,5,6,7], AssertionError("Invalid optimization method")
        
        model_scheduler_p1 = lambda opt: torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=5),
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1E-4, total_iters=self.switch_step),
                    ],
            milestones=[5]
            )
            
        model_scheduler_p2 = lambda opt: torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1E-4, total_iters=self.switch_step),
                    torch.optim.lr_scheduler.LinearLR(opt, start_factor=1E-4, total_iters=self.config["training"]["warmup_steps"]),
                    ],
            milestones=[self.switch_step]
            )
        
        
        # optimizers kwargs
        SGD_kwargs = {"momentum":0.0, "weight_decay":0.0}
        
        
        if self.config["training"]["optimization"] == 1:
            """
            Same Adam optimizer for embeddings and model parameters.
            """
            opt_emb_p1 = torch.optim.AdamW(self.parameters(),lr=self.config["training"]["base_lr"])
            opt_emb_p2 = torch.optim.AdamW(self.parameters(),lr=self.config["training"]["base_lr"])
            opt_model_p1   = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            opt_model_p2   = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        
        
        elif self.config["training"]["optimization"] == 2:
            """
            Adam optimizer for embeddings and model parameters with different learning rates.
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1   = torch.optim.Adam(group_1, lr=self.config["training"]["emb_lr"])
            opt_emb_p2   = torch.optim.Adam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        
        
        
        if self.config["training"]["optimization"] == 3:
            """
            Model Optimizer | Enc. Emb. Opt. 
            ----------------|---------------
            Adam(base_lr)   | SparseAdam(emb_lr)
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        
        
        
        if self.config["training"]["optimization"] == 4:
            """
            Model Optimizer | Enc. Emb. Opt. 
            ----------------|---------------
            Adam(base_lr)   | Adagrad(emb_lr)
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1 = torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_start_lr"])
            opt_model_p1 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        
        
        
        
        if self.config["training"]["optimization"] == 5:
            """
            Embedding optimizer runs in two phases, switching at 'switch_epoch': 
            Phase  |  Objective        | Model Optimizer | Embedding Optimizer
            ------ | ----------------- | ----------------| ----------------
            1      | learn embedding   | SGD             | SparseAdam
            2      | learn model       | Adam            | SparseAdam
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1 =torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1   = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            # schedulers
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
                }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
                }

            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
        
        
        if self.config["training"]["optimization"] == 6:
            """
            Embedding optimizer runs in two phases, switching at 'switch_epoch': 
            Phase  | Model Optimizer | Embedding Optimizer
            ------ | ----------------| ----------------
            1      | SGD(sched.1)    | SparseAdam(emb_start_lr)
            2      | Adam(sched.2)   | SparseAdam(emb_lr)
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1 =torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 =torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_lr"])
            opt_model_p1   = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2   = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            # schedulers
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
                }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
                }

            return [opt_emb_p1, opt_emb_p2, opt_model_p1, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
        
        
        if self.config["training"]["optimization"] == 7:
            """
            Embedding optimizer runs in two phases, switching at 'switch_epoch': 
            Phase  |  Objective        | Model Optimizer | Embedding Optimizer
            ------ | ----------------- | ----------------| ----------------
            1      | learn embedding   | SGD             | Adagrad
            2      | learn model       | Adam            | Adagrad
            """
            
            # get parameters
            group_1, group_2 = self.split_params()
            
            # optimizers
            opt_emb_p1 = torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.Adagrad(group_1,  lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            # schedulers
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
                }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
                }

            return [opt_emb_p1, opt_emb_p2, opt_model_p1, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
        
    def _update_target_upper_bound(self):
        if self.target_show_mode == "random":
            self.show_trg_max_idx = random.randint(0, self.show_trg_max_idx_upper_bound)