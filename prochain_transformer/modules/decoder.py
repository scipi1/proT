import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath, join
import sys
sys.path.append(dirname(abspath(__file__)))
from extra_layers import (
    Normalization,
    UniformAttentionMask
)

# TODO move somewhere else
from datetime import datetime

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DUMP_DIR = join(ROOT_DIR,"dump")

# Get the current date and time
def time_name(filename):
    now = datetime.now()

    # Format it as YYYYMMDD_HHMMSS (or adjust as needed)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Create a filename with the timestamp
    return filename+f"_{timestamp}"


class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        global_cross_attention,
        d_model_dec,
        # d_yt, #(??) #TODO don't need them?
        # d_yc, #(??)
        activation,
        norm,
        d_ff,
        dropout_ff,
        dropout_attn_out,
        ):
        super(DecoderLayer, self).__init__()
        
        # global attention is initialized in the `model.py` module
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        
        
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)

        # output convolutions or linear
        self.conv1 = nn.Conv1d(in_channels=d_model_dec, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model_dec, kernel_size=1)
        self.linear1 = nn.Linear(in_features=d_model_dec, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_dec, bias=True)
        
        
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        
    def forward(
        self, X: torch.Tensor, 
        enc_out: torch.Tensor, 
        self_mask_miss_k: torch.Tensor, 
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor, 
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool
        ):
        
        X1 = self.norm1(X, ~self_mask_miss_q)
        
        X1, dec_self_att = self.global_self_attention(
            query=X1,
            key=X1,
            value=X1,
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask
            )
        
        X = X + self.dropout_attn_out(X1)
        
        X1 = self.norm2(X, ~self_mask_miss_q)
        
        X1, dec_cross_att = self.global_cross_attention(
            query=X1,
            key=enc_out,
            value=enc_out,
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos = None,
            causal_mask = False
            )
        
        X = X + self.dropout_attn_out(X1)

        X1 = self.norm3(X, ~self_mask_miss_q)
        
        # TODO: give options
        # feedforward layers as 1x1 convs
        # X1 = self.dropout_ff(self.activation(self.conv1(X1.transpose(-1, 1))))
        # X1 = self.dropout_ff(self.conv2(X1).transpose(-1, 1))
        
        # feedforward layers (linear)
        X1 = self.dropout_ff(self.activation(self.linear1(X1)))
        X1 = self.dropout_ff(self.linear2(X1))
        
        # final res connection
        decoder_out = X + X1

        return decoder_out, dec_self_att, dec_cross_att
    
    
class Decoder(nn.Module):
    def __init__(
        self, 
        decoder_layers: int, 
        norm_layer: nn.Module, 
        emb_dropout: float):
        
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self, X: torch.Tensor, 
        enc_out: torch.Tensor, 
        self_mask_miss_k: torch.Tensor, 
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor, 
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool
        ):
        
        X = self.emb_dropout(X)

        dec_self_att_list = []
        dec_cross_att_list = []
        
        for _, decoder_layer in enumerate(self.layers):
            
            X, dec_self_att, dec_cross_att = decoder_layer(
                X=X, 
                enc_out=enc_out, 
                self_mask_miss_k=self_mask_miss_k, 
                self_mask_miss_q=self_mask_miss_q,
                cross_mask_miss_k=cross_mask_miss_k, 
                cross_mask_miss_q=cross_mask_miss_q,
                dec_input_pos=dec_input_pos,
                causal_mask=causal_mask
                )
            
            dec_self_att_list.append(dec_self_att)
            dec_cross_att_list.append(dec_cross_att)

        if self.norm_layer is not None:
            X = self.norm_layer(X, ~self_mask_miss_q)

        return X, dec_self_att_list, dec_cross_att_list