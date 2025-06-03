from functools import partial
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
# from einops import rearrange, repeat TO DELETE

from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(__file__)))
from modules.extra_layers import Normalization
from modules.encoder import Encoder, EncoderLayer
from modules.decoder import Decoder, DecoderLayer
from modules.attention import ScaledDotAttention,KernelAttention,AttentionLayer
    
from modules.embedding import ModularEmbedding
from modules.extra_layers import UniformAttentionMask


class Spacetimeformer(nn.Module):
    """
    Simplified Spacetimeformer (https://github.com/QData/spacetimeformer)
    Required data shape: (BATCH_SIZE, sequence_length, variables)
    """
    def __init__(
        self,
        
        # embeddings
        ds_embed_enc,
        comps_embed_enc,
        ds_embed_dec,
        comps_embed_dec,
        
        # attention
        enc_attention_type,
        enc_mask_type,
        dec_self_attention_type,
        dec_self_mask_type,
        dec_cross_attention_type,
        dec_cross_mask_type,
        n_heads: int,
        #attn_factor: int = 5, #TODO understand, DO NOT DEL for now!
        
        
        # dropout
        dropout_emb: float,
        dropout_data: float, # from old embeddings
        dropout_attn_out: float,
        dropout_ff: float ,
        enc_dropout_qkv: float,
        enc_attention_dropout: float,
        dec_self_dropout_qkv: float,
        dec_self_attention_dropout: float,
        dec_cross_dropout_qkv: float,
        dec_cross_attention_dropout: float,
        
        # options
        e_layers: int,
        d_layers: int,
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        out_dim: int,
        d_ff: int,
        d_model_enc: int,
        d_model_dec: int,
        d_queries_keys: int,
        ):
        super().__init__()
        
        # embeddings. separate enc/dec in case the variable indices are not aligned
        self.enc_embedding = ModularEmbedding(
            ds_embed=ds_embed_enc,
            comps=comps_embed_enc,
            device=device)
        
        self.dec_embedding = ModularEmbedding(
            ds_embed=ds_embed_dec,
            comps=comps_embed_dec,
            device=device)
        
        # Select Attention Options
        attn_shared_kwargs = {
            "n_heads"           : n_heads,
            "d_queries_keys"    : d_queries_keys,
        }

        attn_enc_kwargs = {
            "d_model_queries"   : d_model_enc,
            "d_model_keys"      : d_model_enc,
            "d_model_values"    : d_model_enc,
            "attention_type"    : enc_attention_type,
            "mask_type"         : enc_mask_type,
            "dropout_qkv"       : enc_dropout_qkv,
            "attention_dropout" : enc_attention_dropout,
        }
        
        attn_dec_self_kwargs = {
            "d_model_queries"   : d_model_dec,
            "d_model_keys"      : d_model_dec,
            "d_model_values"    : d_model_dec,
            "attention_type"    : dec_self_attention_type,
            "mask_type"         : dec_self_mask_type,
            "dropout_qkv"       : dec_self_dropout_qkv,
            "attention_dropout" : dec_self_attention_dropout,
        }
        
        attn_dec_cross_kwargs = {
            "d_model_queries"   : d_model_dec,
            "d_model_keys"      : d_model_enc,
            "d_model_values"    : d_model_enc,
            "attention_type"    : dec_cross_attention_type,
            "mask_type"         : dec_cross_mask_type,
            "dropout_qkv"       : dec_cross_dropout_qkv,
            "attention_dropout" : dec_cross_attention_dropout,
        }
        

        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    global_attention=self._attn(**(attn_shared_kwargs | attn_enc_kwargs)),
                    d_model_enc=d_model_enc,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm
                    ) for _ in range(e_layers)],
            
            #conv_layers = [ConvBlock(split_length_into=split_length_into, d_model=d_model) for l in range(intermediate_downsample_convs)],
            norm_layer = Normalization(norm, d_model=d_model_enc) if use_final_norm else None,
            emb_dropout = dropout_emb
            )


        self.decoder = Decoder(
            decoder_layers=[
                DecoderLayer(
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec_self_kwargs)),
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec_cross_kwargs)),
                    d_model_dec=d_model_dec,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                    ) for _ in range(d_layers)],

            norm_layer=Normalization(norm, d_model=d_model_dec) if use_final_norm else None,
            emb_dropout=dropout_emb
            )
        
        out_dim = 1
        recon_dim = 1
        d_yc = 1
        
        # final linear layers turn Transformer output into predictions
        self.forecaster = nn.Linear(d_model_dec, out_dim, bias=True)
        self.reconstructor = nn.Linear(d_model_enc, recon_dim, bias=True)
        # self.classifier = nn.Linear(d_model, d_yc, bias=True)
        
        
    def forward(
        self,
        input_tensor,
        target_tensor,
        ):
        
        # embed input and get mask for missing values
        enc_input = self.enc_embedding(X=input_tensor)
        enc_input_pos = self.enc_embedding.pass_var(X=input_tensor) #TODO still relevant?
        enc_mask = self.enc_embedding.get_mask(X=input_tensor)
        
        # scale embedding with missing data
        # scale = enc_mask.shape[1]/torch.sum(~enc_mask.squeeze(),axis=-1, keepdim=True)
        # enc_input *= scale.unsqueeze(-1)
        # enc_input*= ~enc_mask
        
        # pass embedded input to encoder
        enc_out, enc_self_attns = self.encoder(
            X=enc_input,
            mask_miss_k=enc_mask,
            mask_miss_q=enc_mask,
            enc_input_pos=enc_input_pos
            )
        
        # embed target
        dec_input = self.dec_embedding(X=target_tensor)
        dec_input_pos = self.dec_embedding.pass_var(X=target_tensor)
        dec_self_mask=self.dec_embedding.get_mask(X=target_tensor)
        
        
        # pass embedded target and encoder output to decoder
        dec_out, dec_cross_attns = self.decoder(
            X=dec_input,
            enc_out=enc_out,
            self_mask_miss_k=dec_self_mask, 
            self_mask_miss_q=dec_self_mask,
            cross_mask_miss_k=enc_mask, 
            cross_mask_miss_q=dec_self_mask,
            dec_input_pos = dec_input_pos
            )
        # forecasting predictions
        forecast_out = self.forecaster(dec_out)
        # reconstruction predictions
        recon_out = self.reconstructor(enc_out)
        
        return forecast_out, recon_out, (enc_self_attns, dec_cross_attns), enc_mask #TODO maybe add dec_self_att to be fair
    
    
    def _attn(
        self,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        n_heads: int,
        d_queries_keys: int,
        attention_type: str,
        mask_type: str,
        dropout_qkv: float,
        attention_dropout: float,
        ):

        # choose attention type
        assert attention_type in ["Kernel","ScaledDotProduct"]
        
        if attention_type == "Kernel":
            attention_module = KernelAttention
            
        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention
            
        # choose mask type (currently only uniform)
        mask_layer = None # init
        
        if mask_type is not None:
            if mask_type == "Uniform":
                mask_layer = UniformAttentionMask()

            
        att = AttentionLayer(
            attention=attention_module,
            d_model_queries=d_model_queries,
            d_model_keys=d_model_keys,
            d_model_values=d_model_values,
            d_queries_keys=d_queries_keys,
            n_heads=n_heads,
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv)
            
        return att