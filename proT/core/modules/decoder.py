import warnings
from typing import List, Tuple, Optional
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

def time_name(filename: str) -> str:
    """
    Generate a timestamped filename.
    
    Args:
        filename: Base filename to append timestamp to
        
    Returns:
        Filename with timestamp suffix in format 'filename_YYYYMMDD_HHMMSS'
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return filename + f"_{timestamp}"


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feedforward network.
    
    Implements a pre-norm Transformer decoder layer architecture:
    1. Self-attention over decoder inputs
    2. Cross-attention between decoder and encoder outputs
    3. Position-wise feedforward network
    
    Each sub-layer includes residual connections and dropout.
    
    Args:
        global_self_attention: Attention module for self-attention
        global_cross_attention: Attention module for cross-attention with encoder
        d_model_dec: Decoder model dimension
        activation: Activation function type ('relu' or 'gelu')
        norm: Normalization method for layer normalization
        d_ff: Feedforward network hidden dimension
        dropout_ff: Dropout rate for feedforward layers
        dropout_attn_out: Dropout rate for attention output
    """
    def __init__(
        self,
        global_self_attention: nn.Module,
        global_cross_attention: nn.Module,
        d_model_dec: int,
        activation: str,
        norm: str,
        d_ff: int,
        dropout_ff: float,
        dropout_attn_out: float,
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
        self, 
        X: torch.Tensor, 
        enc_out: torch.Tensor, 
        self_mask_miss_k: Optional[torch.Tensor], 
        self_mask_miss_q: Optional[torch.Tensor],
        cross_mask_miss_k: Optional[torch.Tensor], 
        cross_mask_miss_q: Optional[torch.Tensor],
        dec_input_pos: Optional[torch.Tensor],
        causal_mask: bool
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder layer.
        
        Args:
            X: Decoder input tensor (B, L, d_model_dec)
            enc_out: Encoder output tensor (B, S, d_model_enc)
            self_mask_miss_k: Missing value mask for self-attention keys (B, L, 1)
            self_mask_miss_q: Missing value mask for self-attention queries (B, L, 1)
            cross_mask_miss_k: Missing value mask for cross-attention keys (B, S, 1)
            cross_mask_miss_q: Missing value mask for cross-attention queries (B, L, 1)
            dec_input_pos: Position tensor for causal masking (B, L, 1)
            causal_mask: Whether to apply causal masking in self-attention
            
        Returns:
            Tuple containing:
                - decoder_out: Decoder layer output (B, L, d_model_dec)
                - self_att: Self-attention weights
                - cross_att: Cross-attention weights
                - self_ent: Self-attention entropy
                - cross_ent: Cross-attention entropy
        """
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        X1 = self.norm1(X, not_self_mask_miss_q)
        
        X1, self_att, self_ent = self.global_self_attention(
            query=X1,
            key=X1,
            value=X1,
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask
            )
        
        X2 = X + self.dropout_attn_out(X1)
        
        X3 = self.norm2(X2, not_self_mask_miss_q)
        
        # Cross-attention is never causal - decoder queries attend to all encoder outputs
        X3, cross_att, cross_ent = self.global_cross_attention(
            query=X3,
            key=enc_out,
            value=enc_out,
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos = None,
            causal_mask = False
            )
        
        X4 = X2 + self.dropout_attn_out(X3)

        X5 = self.norm3(X4, not_self_mask_miss_q)
        
        # TODO: give options
        # feedforward layers as 1x1 convs
        # X1 = self.dropout_ff(self.activation(self.conv1(X1.transpose(-1, 1))))
        # X1 = self.dropout_ff(self.conv2(X1).transpose(-1, 1))
        
        # feedforward layers (linear)
        X5 = self.dropout_ff(self.activation(self.linear1(X5)))
        X5 = self.dropout_ff(self.linear2(X5))
        
        # final res connection
        decoder_out = X4 + X5

        return decoder_out, self_att, cross_att, self_ent, cross_ent
    
    
class Decoder(nn.Module):
    """
    Transformer decoder consisting of stacked decoder layers.
    
    Processes target sequences by attending to both the target sequence (self-attention)
    and the encoder output (cross-attention). Supports optional final layer normalization
    and embedding dropout.
    
    Args:
        decoder_layers: List of DecoderLayer modules
        norm_layer: Optional final normalization layer
        emb_dropout: Dropout rate applied to input embeddings
    """
    def __init__(
        self, 
        decoder_layers: List[DecoderLayer], 
        norm_layer: Optional[nn.Module], 
        emb_dropout: float):
        
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self, 
        X: torch.Tensor, 
        enc_out: torch.Tensor, 
        self_mask_miss_k: Optional[torch.Tensor], 
        self_mask_miss_q: Optional[torch.Tensor],
        cross_mask_miss_k: Optional[torch.Tensor], 
        cross_mask_miss_q: Optional[torch.Tensor],
        dec_input_pos: Optional[torch.Tensor],
        causal_mask: bool
        ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all decoder layers.
        
        Args:
            X: Decoder input tensor (B, L, d_model_dec)
            enc_out: Encoder output tensor (B, S, d_model_enc)
            self_mask_miss_k: Missing value mask for self-attention keys
            self_mask_miss_q: Missing value mask for self-attention queries
            cross_mask_miss_k: Missing value mask for cross-attention keys
            cross_mask_miss_q: Missing value mask for cross-attention queries
            dec_input_pos: Position tensor for causal masking
            causal_mask: Whether to apply causal masking
            
        Returns:
            Tuple containing:
                - X: Final decoder output (B, L, d_model_dec)
                - self_att_list: List of self-attention weights from each layer
                - cross_att_list: List of cross-attention weights from each layer
                - self_enc_list: List of self-attention entropies from each layer
                - cross_enc_list: List of cross-attention entropies from each layer
        """
        X = self.emb_dropout(X)

        self_att_list, cross_att_list = [], []
        self_enc_list, cross_enc_list = [], []
        
        for _, decoder_layer in enumerate(self.layers):
            
            X, self_att, cross_att, self_enc, cross_enc = decoder_layer(
                X=X, 
                enc_out=enc_out, 
                self_mask_miss_k=self_mask_miss_k, 
                self_mask_miss_q=self_mask_miss_q,
                cross_mask_miss_k=cross_mask_miss_k, 
                cross_mask_miss_q=cross_mask_miss_q,
                dec_input_pos=dec_input_pos,
                causal_mask=causal_mask
                )
            
            self_att_list.append(self_att)
            cross_att_list.append(cross_att)
            self_enc_list.append(self_enc)
            cross_enc_list.append(cross_enc)

        if self.norm_layer is not None:
            not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
            X = self.norm_layer(X, not_self_mask_miss_q)

        return X, self_att_list, cross_att_list, self_enc_list, cross_enc_list
