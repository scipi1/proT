import os
import warnings
from math import sqrt, log
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from proT.core.modules.extra_layers import UniformAttentionMask
from proT.utils.entropy_utils import register_attention_entropy, calculate_attention_entropy
from typing import List

class ScaledDotAttention(nn.Module):
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttention, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        group_ids: torch.Tensor = None,
        group_mask: bool = False,
        ):
        
        # Handle both single-head (3D) and multi-head (4D) tensors
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Apply causal mask
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            
            scores = scores + M_causal
        
        # Apply group mask (tokens attend only within the same group)
        if group_ids is not None and group_mask:
            M_group = build_group_mask(group_ids, n_heads=H)
            
            scores = scores + M_group
        
        # Apply missing data masks
        # masking missing value with -inf to force the softmax to zero
        # (reference https://arxiv.org/abs/2407.11540)
        
        if is_multihead:
            # For multi-head: scores shape is (B, H, L, S)
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
            # Expand masks to (B, H, L, S)
            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, query_size).transpose(-1, -2)
            
            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, key_size)
        else:
            # For single-head: scores shape is (B, L, S)
            
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.expand(-1, -1, query_size).transpose(-1, -2)
            
            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.expand(-1, -1, key_size)
        
        if mask_miss_k is not None:
            M_k = torch.zeros_like(scores).masked_fill_(mask_miss_k_expanded, -torch.inf)
        else:
            M_k = torch.zeros_like(scores)
            
        if mask_miss_q is not None:
            M_q = torch.zeros_like(scores).masked_fill_(mask_miss_q_expanded, -torch.inf)
        else:
            M_q = torch.zeros_like(scores)
        
        # Scale the attention scores
        scaled_scores = scale * (scores + M_k)

        # NaN-score guard: a NaN score means poisoned Q/K (e.g. non-finite
        # weights after a bad update) - mask those keys to -inf so softmax
        # stays finite; the all-(-inf)-row guard below then zeroes rows that
        # lost every key this way. A NaN score would otherwise NaN the whole
        # row through softmax (isinf(NaN) is False, so the -inf row guard
        # alone does NOT catch it - this was the crash of run 16140288).
        nan_scores = torch.isnan(scaled_scores)
        n_nan_scores = int(nan_scores.sum())
        if n_nan_scores:
            scaled_scores = scaled_scores.masked_fill(nan_scores, -torch.inf)
        
        # Detect rows where ALL values are -inf (no valid attention targets)
        # This happens when causal mask + missing data masks block all positions
        all_neginf = torch.isinf(scaled_scores) & (scaled_scores < 0)
        all_neginf_rows = all_neginf.all(dim=-1, keepdim=True)
        
        # Replace all-inf rows with zeros to prevent NaN in softmax
        # This is mathematically correct: no valid keys = uniform zero attention
        scaled_scores = torch.where(
            all_neginf_rows.expand_as(scaled_scores), 
            torch.zeros_like(scaled_scores), 
            scaled_scores
        )
        
        # Compute softmax (now safe from NaN)
        att = torch.softmax(scaled_scores, dim=-1)
        
        # Zero out the attention for rows that were all -inf
        att = torch.where(
            all_neginf_rows.expand_as(att), 
            torch.zeros_like(att), 
            att
        )
        
        # Robust NaN guard: warn with diagnostics and zero the NaN entries
        # instead of crashing the run (see sanitize_nan; PROT_NAN_GUARD=raise
        # restores the old hard-failure behavior for debugging)
        att = sanitize_nan(
            att,
            name="att_after_softmax",
            layer_name=self.layer_name,
            n_nan_scores=n_nan_scores,
            all_neginf_rows=all_neginf_rows,
        )
        
        # Apply query mask and relu
        att = torch.relu(att + M_q)
        
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
            
            
        A = torch.nan_to_num(self.dropout(att))
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy


def build_causal_mask(p: torch.Tensor, n_heads: int = 1) -> torch.Tensor:
    """
    Args:
        p: (B, L, 1) tensor with the position of every token in the sequence.
        n_heads: number of attention heads

    Returns:
        For single head (n_heads=1): (B, L, L) mask M
        For multi-head (n_heads>1): (B, H, L, L) mask M
        with M[b, (h,) i, j] = -inf if p[b, j] > p[b, i], 0 otherwise.
    """
    p_flat = p.squeeze(-1)      # shape (B, L)

    p_i = p_flat.unsqueeze(-1)  # shape (B, L, 1)
    p_j = p_flat.unsqueeze(-2)  # shape (B, 1, L)

    # Build the additive mask (same dtype/device as `p`)
    M = torch.zeros_like(p_i.expand(-1, -1, p_flat.size(-1)))
    M.masked_fill_(p_j > p_i, float("-inf"))
    
    # Expand for multi-head if needed
    if n_heads > 1:
        M = M.unsqueeze(1).expand(-1, n_heads, -1, -1)  # (B, H, L, L)
    
    return M


def build_group_mask(g: torch.Tensor, n_heads: int = 1) -> torch.Tensor:
    """
    Build an additive attention mask that blocks attention between tokens
    belonging to different groups (e.g., different PCB layers).

    Args:
        g: (B, L) or (B, L, 1) tensor with the group id of every token.
           NaN group ids are mapped to a sentinel (-1) so that all
           unassigned tokens form their own group.
        n_heads: number of attention heads

    Returns:
        For single head (n_heads=1): (B, L, L) mask M
        For multi-head (n_heads>1): (B, H, L, L) mask M
        with M[b, (h,) i, j] = -inf if g[b, i] != g[b, j], 0 otherwise.
    """
    g_flat = torch.nan_to_num(g.squeeze(-1) if g.dim() == 3 else g, nan=-1.0)  # (B, L)

    g_i = g_flat.unsqueeze(-1)  # (B, L, 1)
    g_j = g_flat.unsqueeze(-2)  # (B, 1, L)

    # Build the additive mask (same dtype/device as `g`)
    M = torch.zeros_like(g_i.expand(-1, -1, g_flat.size(-1)))
    M.masked_fill_(g_j != g_i, float("-inf"))

    # Expand for multi-head if needed
    if n_heads > 1:
        M = M.unsqueeze(1).expand(-1, n_heads, -1, -1)  # (B, H, L, L)

    return M


def calculate_attention_entropy(att_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate entropy of attention weights.
    
    Args:
        att_weights: Attention weights tensor
                    - Multi-head: (B, H, L, S) 
                    - Single-head: (B, L, S)
        eps: Small value to avoid log(0)
    
    Returns:
        Entropy tensor:
        - Multi-head: (B, H, L) - entropy for each query position in each head
        - Single-head: (B, L) - entropy for each query position
    """
    # Clamp to avoid log(0)
    att_clamped = torch.clamp(att_weights, min=eps)
    
    # Calculate entropy: -sum(p * log(p)) along the key dimension (last dimension)
    log_att = torch.log(att_clamped)
    entropy = -torch.sum(att_clamped * log_att, dim=-1)
    
    # Handle NaN values that might arise from 0 * log(0)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


        
        
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
        dropout_qkv: float,
        register_entropy: bool = False, 
        layer_name: str = None
        ):
        
        super(AttentionLayer, self).__init__()
        
        self.inner_attention = attention(
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            register_entropy=register_entropy,
            layer_name=layer_name
            )
        
        self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model_keys, d_model_values * n_heads)
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
        group_ids: torch.Tensor = None,
        group_mask: bool = False,
        ):
        
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        
        # Apply projections and reshape for multi-head attention
        if H > 1:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, H, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, H, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)
            
        out, attn, ent = self.inner_attention(
            query=query,
            key=key,
            value=value,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos=pos,
            causal_mask=causal_mask,
            group_ids=group_ids,
            group_mask=group_mask,
            )
        
        # Reshape output and apply final projection if multi-head
        if H > 1:
            # out shape is (B, L, H, d_v) -> reshape to (B, L, H*d_v)
            out = out.view(B, L, -1)
            out = self.out_projection(out)
        else:
            # out shape is already (B, L, d_v)
            out = out.view(B, L, -1)
        
        return out, attn, ent
    
    
    
def main():
    """Quick test for both single-head and multi-head attention

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
    
    print("Testing single-head attention (n_heads=1):")
    attention_single = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)
    
    out_single, score_single, ent = attention_single.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
        )
    
    print(f"Single-head - Output shape: {out_single.shape}, Score shape: {score_single.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    print("\nTesting multi-head attention (n_heads=4):")
    attention_multi = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
        d_queries_keys=d_queries_keys,
        n_heads=4,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)
    
    out_multi, score_multi, ent = attention_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
        )
    
    print(f"Multi-head - Output shape: {out_multi.shape}, Score shape: {score_multi.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    # Test with causal mask and position
    print("\nTesting with causal mask:")
    pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()  # (1, 5, 1)
    
    out_causal, score_causal, ent = attention_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
        )
    
    print(f"Causal multi-head - Output shape: {out_causal.shape}, Score shape: {score_causal.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    print("All tests completed successfully!")
    
    
    
# NaN-guard configuration for sanitize_nan. The mode is read at call time so
# it can be toggled without re-importing the module:
#   PROT_NAN_GUARD=warn  (default) warn with diagnostics and zero NaN entries
#   PROT_NAN_GUARD=raise           raise RuntimeError (debugging)
_NAN_GUARD_ENV_VAR = "PROT_NAN_GUARD"
_NAN_WARN_MAX = 20  # rate limit: at most this many full warnings per process
_nan_warn_state = {"count": 0}


def sanitize_nan(
    tensor: torch.Tensor,
    name: str,
    layer_name: str = None,
    n_nan_scores: int = 0,
    all_neginf_rows: torch.Tensor = None,
) -> torch.Tensor:
    """Robust NaN guard for the attention weights (replaces check_nan).

    A NaN in the attention weights is a symptom, not the disease: the Q/K
    projections were already non-finite (upstream weight poisoning, e.g. a
    bad optimizer step) or a masking path removed every key of a row.
    Crashing the run (old check_nan) loses hours of compute over a
    recoverable event, while silently continuing would hide a real problem.

    Behavior (env var PROT_NAN_GUARD, default "warn"):
      - "warn": emit a rate-limited warning with diagnostics and replace the
        NaN entries with 0 (a poisoned query then contributes no attention
        output; the residual stream of the encoder layer carries the rest).
        Training continues.
      - "raise": raise a RuntimeError with the same diagnostics (debugging).

    Args:
        tensor: Attention weights after softmax, (B, H, L, S) or (B, L, S).
        name: Tensor label used in the message.
        layer_name: Attention layer label (e.g. "enc_self_att", "pool_att").
        n_nan_scores: Number of NaN scores masked to -inf before softmax
            (> 0 flags upstream poisoning of Q/K).
        all_neginf_rows: Bool tensor from the all-(-inf)-row guard; the row
            count is evaluated lazily, only when a warning fires.

    Returns:
        The tensor with NaN entries replaced by 0 (unchanged if clean).
    """
    n_nan = int(torch.isnan(tensor).sum())
    if n_nan == 0 and n_nan_scores == 0:
        return tensor

    n_masked_rows = int(all_neginf_rows.sum()) if all_neginf_rows is not None else 0
    msg = (
        f"NaN guard in {name} (layer={layer_name}): {n_nan} NaN attention "
        f"entries replaced by 0 ({n_nan / tensor.numel():.2%} of "
        f"{tensor.numel()}); {n_nan_scores} NaN scores masked to -inf before "
        f"softmax; {n_masked_rows} all-masked rows zeroed. Likely causes: "
        "(a) non-finite weights from a previous bad update (upstream "
        "poisoning - check earlier non-finite warnings), (b) masking removed "
        "every key of a query row, (c) score overflow. Training continues; "
        f"set {_NAN_GUARD_ENV_VAR}=raise to crash on this event instead."
    )
    if os.environ.get(_NAN_GUARD_ENV_VAR, "warn").lower() == "raise":
        raise RuntimeError(msg)
    if _nan_warn_state["count"] < _NAN_WARN_MAX:
        _nan_warn_state["count"] += 1
        warnings.warn(msg, RuntimeWarning)
        if _nan_warn_state["count"] == _NAN_WARN_MAX:
            warnings.warn(
                "NaN guard: warning rate limit reached; further events are "
                "suppressed from the log.", RuntimeWarning)
    return torch.nan_to_num(tensor, nan=0.0)
    
if __name__ == "__main__":
    main()
