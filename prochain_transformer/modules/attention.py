from os.path import dirname, abspath
import sys
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from math import sqrt, log
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from prochain_transformer.modules.extra_layers import UniformAttentionMask
from typing import List

class ScaledDotAttention(nn.Module):
    def __init__(self, mask_layer: nn.Module, attention_dropout: float):
        
        super(ScaledDotAttention, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        ):
        
        #B,L,H,E = query.shape
        E = query.shape[-1]
        #_,S,_,D = value.shape
        
        scale =1.0 / sqrt(E)
        
        # scores = torch.einsum("blhe,bshe->bhls", query, key)
        scores = torch.einsum("ble,bse->bls", query, key)
        
        # if pos is not None:
        #     """
        #     Attention with Linear Bias for positions
        #     (reference https://arxiv.org/abs/2108.12409)
        #     """
        #     pos = pos.expand(B,L,L).transpose(-1,-2)-pos.expand(B,L,L)
        #     m = .5
        #     scores += m*pos.nan_to_num()
        
        # causal mask
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos)
            scores = scores + M_causal
        
        # missing data
        if mask_miss_k is not None:
            """
            masking missing value with -inf to force the softmax to zero
            (reference https://arxiv.org/abs/2407.11540)
            """
            key_size = scores.size(-1)                                                         
            query_size = scores.size(-2)                                                      
            mask_miss_k_expanded = mask_miss_k.expand(-1,key_size,query_size).transpose(-1,-2)
            mask_miss_q_expanded = mask_miss_q.expand(-1,query_size,key_size)
            M_k = torch.zeros_like(scores).masked_fill_(mask_miss_k_expanded,-torch.inf)
            M_q = torch.zeros_like(scores).masked_fill_(mask_miss_q_expanded,-torch.inf)
            att = torch.relu(torch.softmax(scale * (scores + M_k), dim=-1) + M_q)

        else:
            att = torch.softmax(scale * scores, dim=-1)
            
        A = torch.nan_to_num(self.dropout(att))
        V = torch.einsum("bsl,bld->bsd", A, value)
        
        return (V.contiguous(), A) #A


def build_causal_mask(p: torch.Tensor) -> torch.Tensor:
    """
    Args:
        p: (B, L, 1) tensor with the position of every token in the sequence.

    Returns:
        (B, L, L) mask M with
            M[b, i, j] = -inf if p[b, j] > p[b, i]
                        0    otherwise.
    """
    p_flat = p.squeeze(-1)      # shape (B, L)

    p_i = p_flat.unsqueeze(-1)  # shape (B, L, 1)
    p_j = p_flat.unsqueeze(-2)  # shape (B, 1, L)

    # Build the additive mask (same dtype/device as `p`)
    M = torch.zeros_like(p_i.expand(-1, -1, p_flat.size(-1)))
    M.masked_fill_(p_j > p_i, float("-inf"))
    return M

def k_exp(Q,K,d_k):
    return torch.exp(torch.einsum("blhe,bkhe->bhlk", Q, K)/sqrt(d_k))

# TODO delete
# class KernelAttention(nn.Module):
#     def __init__(self, mask_layer: nn.Module=None, attention_dropout: float=0) -> None:
        
#         super(KernelAttention, self).__init__()
        
#         self.mask_layer = mask_layer
#         self.dropout = nn.Dropout(attention_dropout)
#         self.kernel = self.k_exp 
        
#     def k_exp(self,Q,K,d_k):
#         return torch.exp(torch.einsum("ble,bke->blk", Q, K)/sqrt(d_k))
        
#     def forward(
#         self, 
#         query:torch.Tensor, 
#         key:torch.Tensor, 
#         value:torch.Tensor, 
#         mask=None,
#         output_att:bool=False):
        
#         B,L,E = query.shape
#         _,K,_ = key.shape
#         _,S,D = value.shape
        
#         ker = self.dropout(self.kernel(query,key,E))
        
#         # normalization constant
#         Z = ker.sum(axis=2).unsqueeze(-1).expand(-1,-1,K)
        
#         attention_scores = ker/Z
        
#         # masking
#         if self.mask_layer is not None and mask is not None:
#             attention_scores = self.mask_layer(attention_scores, mask)
        
#         return (attention_scores@value, attention_scores)
        
        
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        d_queries_keys: int,
        n_heads: int,
        mask_layer: nn.Module,
        attention_dropout: float,
        dropout_qkv: float):
        
        super(AttentionLayer, self).__init__()
        
        self.inner_attention = attention(
            mask_layer=mask_layer,
            attention_dropout=attention_dropout
            )
        self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model_values, d_model_queries * n_heads)
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_queries)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        ):
        
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        
        # Move it eventually somewhere else
        # if pos is not None:
        #     pos = pos.expand(B,L,L).transpose(-1,-2)-pos.expand(B,L,L)
            
        
        
        if H >1:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, H, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, H, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)
            
        out, attn = self.inner_attention(
            query=query,
            key=key,
            value=value,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos=pos,
            causal_mask=causal_mask,
            )
        
        out = out.view(B, L, -1)
        
        if self.n_heads>1:
            out = self.out_projection(out)
        
        return out, attn
    
    
    
def main():
    """Quick test

    Returns:
        _tuple(torch.Tensor): attention output and score
    """
    
    bs = 1
    seq_len = 5
    d_model = 12
    d_queries_keys = 8
    mask = [False, True, True, False, False]
    x = torch.ones(bs,seq_len,d_model)
    x[0,0,0] = torch.nan
    
    attention = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)
    
    out, score = attention.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None, 
        )
    
    
    print(f"score shape: {score.shape} {out.shape}")
    
    
if __name__ == "__main__":
    main()
    
