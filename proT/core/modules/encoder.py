from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(__file__)))
from extra_layers import Normalization



class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feedforward network.
    
    Implements a pre-norm Transformer encoder layer architecture:
    1. Self-attention over input sequence
    2. Position-wise feedforward network
    
    Each sub-layer includes residual connections and dropout.
    
    Args:
        global_attention: Attention module for self-attention
        d_model_enc: Encoder model dimension
        activation: Activation function type ('relu' or 'gelu')
        norm: Normalization method for layer normalization
        d_ff: Feedforward network hidden dimension
        dropout_ff: Dropout rate for feedforward layers
        dropout_attn_out: Dropout rate for attention output
    """
    def __init__(
        self,
        global_attention: nn.Module,
        d_model_enc: int,  
        activation: str,         
        norm: str, 
        d_ff: int,                 
        dropout_ff: float,            
        dropout_attn_out: float
        ):
        super().__init__()
        
        # global attention is initialized in the `model.py` module
        self.global_attention = global_attention
        
        # normalization
        self.norm1 = Normalization(method=norm, d_model=d_model_enc)
        self.norm2 = Normalization(method=norm, d_model=d_model_enc)
        
        # output convolutions or linear
        self.conv1 = nn.Conv1d(in_channels=d_model_enc, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model_enc, kernel_size=1)
        self.linear1 = nn.Linear(in_features=d_model_enc, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_enc, bias=True)
        
        # dropouts and activation
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        #self.time_windows = time_windows                #??
        #self.time_window_offset = time_window_offset    #??
        #self.d_yc = d_yc                                # for local attention

    def forward(
        self, 
        X: torch.Tensor, 
        mask_miss_k: Optional[torch.Tensor], 
        mask_miss_q: Optional[torch.Tensor], 
        enc_input_pos: Optional[torch.Tensor],
        causal_mask: bool
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder layer.
        
        Args:
            X: Input tensor (B, L, d_model_enc)
            mask_miss_k: Missing value mask for keys (B, L, 1)
            mask_miss_q: Missing value mask for queries (B, L, 1)
            enc_input_pos: Position tensor for causal masking (B, L, 1)
            causal_mask: Whether to apply causal masking
            
        Returns:
            Tuple containing:
                - encoder_out: Encoder layer output (B, L, d_model_enc)
                - attn: Self-attention weights
                - ent: Self-attention entropy
        """
        # uses pre-norm Transformer architecture
        
        not_mask_miss_q = ~mask_miss_q if mask_miss_q is not None else None
        
        X1 = self.norm1(X, not_mask_miss_q)
        
        
        # self-attention queries=keys=values=X
        X1, attn, ent = self.global_attention(
            query=X1,
            key=X1,
            value=X1,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos = enc_input_pos,
            causal_mask=causal_mask,
            )                    
        
        # resnet
        X = X + self.dropout_attn_out(X1)
        
        X1 = self.norm2(X, not_mask_miss_q)
        
        
        # feedforward layers (done here as 1x1 convs)
        # X1 = self.dropout_ff(self.activation(self.conv1(X1.transpose(-1, 1))))
        # X1 = self.dropout_ff(self.conv2(X1).transpose(-1, 1))
        
        
        # feedforward layers (linear)
        X1 = self.dropout_ff(self.activation(self.linear1(X1)))
        X1 = self.dropout_ff(self.linear2(X1))
        
        # final res connection
        encoder_out = X + X1
        
        return encoder_out, attn, ent
    
    
class Encoder(nn.Module):
    """
    Transformer encoder consisting of stacked encoder layers.
    
    Processes input sequences through self-attention and feedforward networks.
    Supports optional final layer normalization and embedding dropout.
    
    Args:
        encoder_layers: List of EncoderLayer modules
        norm_layer: Optional final normalization layer
        emb_dropout: Dropout rate applied to input embeddings
    """
    def __init__(
        self,
        encoder_layers: List[EncoderLayer],
        norm_layer: Optional[nn.Module],
        emb_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self, 
        X: torch.Tensor, 
        mask_miss_k: Optional[torch.Tensor], 
        mask_miss_q: Optional[torch.Tensor],
        enc_input_pos: Optional[torch.Tensor],
        causal_mask: bool
        ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all encoder layers.
        
        Args:
            X: Input tensor (B, L, d_model_enc)
            mask_miss_k: Missing value mask for keys
            mask_miss_q: Missing value mask for queries
            enc_input_pos: Position tensor for causal masking
            causal_mask: Whether to apply causal masking
            
        Returns:
            Tuple containing:
                - X: Final encoder output (B, L, d_model_enc)
                - attn_list: List of attention weights from each layer
                - ent_list: List of attention entropies from each layer
        """
        X = self.emb_dropout(X)

        attn_list, ent_list = [], []
        
        for _, encoder_layer in enumerate(self.layers):
            X, attn, ent = encoder_layer(
                X=X, 
                mask_miss_k=mask_miss_k, 
                mask_miss_q=mask_miss_q, 
                enc_input_pos=enc_input_pos,
                causal_mask=causal_mask) 
            
            attn_list.append(attn)
            ent_list.append(ent)
            
        if self.norm_layer is not None:
            not_mask_miss_q = ~mask_miss_q if mask_miss_q is not None else None
            X = self.norm_layer(X, not_mask_miss_q)

        return X, attn_list, ent_list
