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
        
        self.config = config
        self.dynamic_kwargs = {
            "enc_mask"              : None,
            "dec_self_mask"         : None,
            "dec_cross_mask"        : None}
        
        self.val_idx = config["data"]["val_idx"]
        # TODO: incorporate ds_embed_enc/dec into config
        
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
            self.loss_fn = nn.MSELoss()
            
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
        
        forecast_output, recon_output, (enc_self_attns, dec_cross_attns) = self.model.forward(
            input_tensor=encoder_in,
            target_tensor=dec_input)
            #**self.dynamic_kwargs)
        
        
        return forecast_output, recon_output, (enc_self_attns, dec_cross_attns)
    
    def set_kwargs(self, kwargs):
        self.dynamic_kwargs = kwargs
    
    def _step(self, batch, **kwargs):
        kwargs.update(self.dynamic_kwargs)
        X, Y = batch
        
        
        predict_out,_,_ = self.forward(data_input=X, data_trg=Y, kwargs=self.dynamic_kwargs)
        
        trg = torch.nan_to_num(Y[:,:,self.val_idx])
        
        
        
        loss = self.loss_fn(predict_out.squeeze(), trg.squeeze())
        return loss, predict_out, Y
        
    def training_step(self,batch):
        loss,_, _ = self._step(batch=batch)
        self.log("train_loss",loss,prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss,_,_ = self._step(batch=batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        loss,predict_out, y_val = self._step(batch=batch)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self,batch,batch_idx):
        _,predict_out,y_val = self._step(batch=batch)
        return predict_out,y_val
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.config["training"]["learning_rate"])
