from typing import Any
import pytorch_lightning as pl
# from lightning.pytorch.core import LightningModule
import torch
import torch.nn as nn
from prochain_transformer.model import Spacetimeformer
import torchmetrics



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
        self.dynamic_kwargs = {
            "enc_mask"              : None,
            "dec_self_mask"         : None,
            "dec_cross_mask"        : None}
        
        self.val_idx = config["data"]["val_idx"]
        
        self.model = Spacetimeformer(
            
            # embeddings
            ds_embed_enc =             config["model"]["ds_embed_enc"],
            ds_embed_dec =             config["model"]["ds_embed_dec"],
            comps_embed_enc =          config["model"]["comps_embed_enc"],
            comps_embed_dec =          config["model"]["comps_embed_dec"],

            # attention
            enc_attention_type =       config["model"]["enc_attention_type"],
            dec_self_attention_type =  config["model"]["dec_self_attention_type"],
            dec_cross_attention_type = config["model"]["dec_cross_attention_type"],
            enc_mask_type =            config["model"]["enc_mask_type"],
            dec_self_mask_type =       config["model"]["dec_self_mask_type"],
            dec_cross_mask_type =      config["model"]["dec_cross_mask_type"],
            n_heads =                  config["model"]["n_heads"],
            #attn_factor: int = 5, #TODO understand, DO NOT DEL for now!
            
            # options
            e_layers =       config["model"]["e_layers"],
            d_layers =       config["model"]["d_layers"],
            activation =     config["model"]["activation"],
            norm=            config["model"]["norm"],
            use_final_norm = config["model"]["use_final_norm"],
            device =         torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            out_dim =        config["model"]["out_dim"],
            d_ff =           config["model"]["d_ff"],
            d_model_enc =    config["model"]["d_model_enc"],
            d_model_dec =    config["model"]["d_model_dec"],
            d_queries_keys = config["model"]["d_qk"],
            
            # dropout
            dropout_emb =                 config["model"]["dropout_emb"],
            dropout_data =                config["model"]["dropout_data"], # from old embeddings DEPRECATED
            dropout_attn_out =            config["model"]["dropout_attn_out"],
            dropout_ff =                  config["model"]["dropout_ff"],
            enc_dropout_qkv =             config["model"]["enc_dropout_qkv"],
            enc_attention_dropout =       config["model"]["enc_attention_dropout"],
            dec_self_dropout_qkv =        config["model"]["dec_self_dropout_qkv"],
            dec_self_attention_dropout =  config["model"]["dec_self_attention_dropout"],
            dec_cross_dropout_qkv =       config["model"]["dec_cross_dropout_qkv"],
            dec_cross_attention_dropout = config["model"]["dec_cross_attention_dropout"],
            )
        
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
            
        self.switch_epoch = config["training"]["switch_epoch"]
        self._switched    = False 
            
        self.save_hyperparameters(ignore=['loss_fn'])
        
        
    def forward(
        self, 
        data_input, 
        data_trg,
        kwargs):
        
        encoder_in = data_input
        self.dynamic_kwargs=kwargs
        
        
        # set values of target sequence to zero
        dec_input = data_trg.clone()
        dec_input[:, :, self.val_idx] = 0
        
        forecast_output, recon_output, (enc_self_attns, dec_cross_attns), enc_mask = self.model.forward(
            input_tensor=encoder_in,
            target_tensor=dec_input)
            #**self.dynamic_kwargs)
        
        
        return forecast_output, recon_output, (enc_self_attns, dec_cross_attns), enc_mask
    
    def set_kwargs(self, kwargs):
        self.dynamic_kwargs = kwargs
    
    def _step(self, batch, **kwargs):
        kwargs.update(self.dynamic_kwargs)
        X, Y = batch
        
        
        predict_out,_,_, enc_mask = self.forward(data_input=X, data_trg=Y, kwargs=self.dynamic_kwargs)
        
        trg = torch.nan_to_num(Y[:,:,self.val_idx])
                
        mse_per_elem  = self.loss_fn(predict_out.squeeze(), trg.squeeze())
        
        loss = mse_per_elem.mean()
        return loss, predict_out, Y
        
    
    def training_step(self, batch, batch_idx):
        
        # forward step
        loss,_,_ = self._step(batch=batch)
        
        if loss.isnan().any():
            breakpoint()
        
        # one backward pass for all parameters
        self.manual_backward(loss)
        
        # gradient clipping
        if self.config["training"]["optimization"] in [1,2,3]:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        
        
        if self.config["training"]["optimization"] == 1:
            """
            Same Adam optimizer for embeddings and model parameters.
            """
            
            # get optimizer
            opt = self.optimizers()
            
            # optimizer steps & zero_grad
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        
        
        elif self.config["training"]["optimization"] in [2,5]:
            """
            Adam optimizer for embeddings and model parameters with different learning rates.
            """
            
            # get optimizers
            opt_emb, opt_model = self.optimizers()
            
            # optimizer steps & zero_grad
            opt_emb.step()
            opt_model.step()
            opt_emb.zero_grad(set_to_none=True)
            opt_model.zero_grad(set_to_none=True)
        
        
        
        elif self.config["training"]["optimization"] in [3,4]:
            """
            Embedding optimizer runs in two phases: a first phase with Adagrad, and a second with Adam.
            """
            
            # get optimizers
            opt_emb, opt_model, opt_emb_steady = self.optimizers()

            
            # switch to Adam for embeddings
            if (not self._switched) and (self.current_epoch >= self.switch_epoch):
                
                if self.config["training"]["optimization"] in [4,5]:
                    
                    for p in opt_emb.param_groups[0]['params']:

                        # ensure Adam buffers exist
                        adam_state = opt_emb_steady.state[p]
                        
                        # reset Adam state
                        adam_state['exp_avg']     = torch.zeros_like(p)
                        adam_state['exp_avg_sq']  = torch.zeros_like(p)
                        adam_state['step']        = torch.tensor(0., dtype=torch.float32)
                
                # switch to steady optimizer
                self.optimizers()[0] = opt_emb_steady          
                self._switched = True
            
            # optimizer steps & zero_grad
            opt_emb.step()            
            opt_model.step()          
            opt_emb.zero_grad(set_to_none=True)
            opt_model.zero_grad(set_to_none=True)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        loss,_,_ = self._step(batch=batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    
    def test_step(self,batch,batch_idx):
        loss,_,_ = self._step(batch=batch)
        self.log("test_loss", loss)
        return loss
    
    
    def predict_step(self,batch,batch_idx):
        _,predict_out,y_val = self._step(batch=batch)
        return predict_out,y_val
    
    
    def split_params(self):
        enc_emb_params = list(self.model.enc_embedding.embed_modules_list.parameters())
        dec_emb_params = list(self.model.dec_embedding.embed_modules_list.parameters())
        emb_param_ids = {id(p) for p in enc_emb_params + dec_emb_params}
        other_params = [p for p in self.model.parameters() if id(p) not in emb_param_ids]
        
        return enc_emb_params,dec_emb_params, other_params
    
    
    
    def configure_optimizers(self):
        
        assert self.config["training"]["optimization"] in [1,2,3,4,5], AssertionError("Invalid optimization method")
        
        if self.config["training"]["optimization"] == 1:
            """
            Same Adam optimizer for embeddings and model parameters.
            """
            optimizer = torch.optim.Adam(self.parameters(),lr=self.config["training"]["base_lr"])
            
            return optimizer
        
            
        elif self.config["training"]["optimization"] == 2:
            """
            Adam optimizer for embeddings and model parameters with different learning rates.
            Scheduler for the model parameters to kick in after a few epochs.
            """
            
            # get parameters
            enc_emb_params, dec_emb_params, other_params = self.split_params()
            
            # optimizers
            opt_emb   = torch.optim.Adam(enc_emb_params+dec_emb_params, lr=self.config["training"]["emb_lr"], betas=(0.9, 0.95))
            opt_model   = torch.optim.Adam(other_params, lr=self.config["training"]["base_lr"], betas=(0.9, 0.95))
            
            # schedulers
            sched = torch.optim.lr_scheduler.StepLR(opt_model, step_size=self.config["training"]["warmup_steps"], gamma=0.5)
            
            scheduler_model = {
                "scheduler": sched,
                "interval": "epoch",        # update every batch
            }
            
            return [opt_emb,opt_model], [scheduler_model]
        
        
        if self.config["training"]["optimization"] == 3:
            """
            Embedding optimizer runs in two phases: a first phase with Adagrad, and a second with Adam.
            Scheduler for the model parameters to kick in after a few epochs.
            """
            
            # get parameters
            enc_emb_params, dec_emb_params, other_params = self.split_params()
            
            # optimizers
            opt_emb_init =torch.optim.Adagrad(enc_emb_params,  lr=self.config["training"]["emb_start_lr"], lr_decay=0.0, eps=1e-10)
            opt_emb_steady = torch.optim.Adam(enc_emb_params,  lr=self.config["training"]["emb_lr"], betas=(0.9, 0.95),weight_decay=0.0)
            opt_model   = torch.optim.Adam(other_params+dec_emb_params, lr=self.config["training"]["base_lr"], betas=(0.9, 0.95))
            
            # schedulers
            scheduler_model = {
                "scheduler": torch.optim.lr_scheduler.StepLR(opt_model, step_size=self.config["training"]["warmup_epochs"], gamma=0.5),
                "interval": "epoch",
                }
            
            return [opt_emb_init, opt_model, opt_emb_steady], [scheduler_model]
        
        
        if self.config["training"]["optimization"] == 4 :
            """
            Embedding optimizer runs in two phases: a first phase with Adagrad, and a second with AdamW.
            Scheduler for the model parameters to kick in after a few epochs.
            """
            
            # get parameters
            enc_emb_params, dec_emb_params, other_params = self.split_params()
            
            
            other_params += dec_emb_params + enc_emb_params[2:]
                        
            # optimizers
            opt_emb_init =torch.optim.Adagrad(enc_emb_params[:2],  lr=self.config["training"]["emb_start_lr"], lr_decay=0.0, eps=1e-10)
            opt_emb_steady = torch.optim.SparseAdam(enc_emb_params[:2], lr=self.config["training"]["emb_lr"], betas=(0.9, 0.95))
            opt_model   = torch.optim.Adam(other_params, lr=self.config["training"]["base_lr"], betas=(0.9, 0.95))
            
            # schedulers
            scheduler_model = {
                "scheduler": torch.optim.lr_scheduler.StepLR(opt_model, step_size=self.config["training"]["warmup_epochs"], gamma=0.5),
                "interval": "epoch",
                }
            
            return [opt_emb_init, opt_model, opt_emb_steady], [scheduler_model]
        
        
        if self.config["training"]["optimization"] == 5:
            """
            Embedding optimizer runs in two phases: a first phase with Adagrad, and a second with AdamW.
            Scheduler for the model parameters to kick in after a few epochs.
            """
            
            # get parameters
            enc_emb_params, dec_emb_params, other_params = self.split_params()
            other_params += dec_emb_params + enc_emb_params[2:]
                        
            # optimizers
            opt_emb = torch.optim.SparseAdam(enc_emb_params[:2], lr=self.config["training"]["emb_lr"], betas=(0.9, 0.95))            
            opt_model   = torch.optim.Adam(other_params, lr=self.config["training"]["base_lr"], betas=(0.9, 0.95))
            
            # schedulers
            scheduler_model = {
                "scheduler": torch.optim.lr_scheduler.StepLR(opt_model, step_size=self.config["training"]["warmup_epochs"], gamma=0.5),
                "interval": "epoch",
                }
            
            return [opt_emb, opt_model], [scheduler_model]
