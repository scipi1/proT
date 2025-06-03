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
        self.switch_step = config["training"]["switch_step"]
        
        # keep false
        self._switched    = False
        self.schedulers_flag = False 
            
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
        
        group_1 = enc_emb_params[:2]
        group_2 = other_params + dec_emb_params + enc_emb_params[2:]
        
        return group_1, group_2
    
    
    
    def configure_optimizers(self):
        
        assert self.config["training"]["optimization"] in [1,2,3,4,5,6,7], AssertionError("Invalid optimization method")
        
        
        # def split_params(self):
        #     enc_emb_params = list(self.model.enc_embedding.embed_modules_list.parameters())
        #     dec_emb_params = list(self.model.dec_embedding.embed_modules_list.parameters())
        #     emb_param_ids = {id(p) for p in enc_emb_params + dec_emb_params}
        #     other_params = [p for p in self.model.parameters() if id(p) not in emb_param_ids]

        #     group_1 = enc_emb_params[:2]
        #     group_2 = other_params + dec_emb_params + enc_emb_params[2:]

        #     return group_1, group_2
        
        
        
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
            opt_emb_p1 = torch.optim.Adam(self.parameters(),lr=self.config["training"]["base_lr"])
            opt_emb_p2 = torch.optim.Adam(self.parameters(),lr=self.config["training"]["base_lr"])
            opt_model_p1   = torch.optim.Adam(self.parameters(), lr=self.config["training"]["base_lr"])
            opt_model_p2   = torch.optim.Adam(self.parameters(), lr=self.config["training"]["base_lr"])
            
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



                
