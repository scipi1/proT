from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from proT.modules.attention import ScaledDotAttention, AttentionLayer
from proT.modules.extra_layers import UniformAttentionMask

import torch


BATCH_SIZE = 1
seq_len = 5
d_model = 12
d_queries_keys = 8
d_values = 8
n_heads = 1
dropout_qkv = 0.1


    
def main(*args, **kwargs):
    """Quick test

    Returns:
        _tuple(torch.Tensor): attention output and score
    """
    x = torch.rand(BATCH_SIZE, seq_len, d_model)
    
    attention = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys, 
        n_heads=n_heads, 
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=dropout_qkv)
    
    out, score = attention(
        query=x, 
        key=x, 
        value=x, 
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )
    
    print(f"Attention output shape: {out.shape}")
    print(f"Attention score shape: {score.shape}")
    
    # For single-head attention, score shape should be (B, L, L)
    expected_out_shape = (BATCH_SIZE, seq_len, d_model)
    expected_score_shape = (BATCH_SIZE, seq_len, seq_len) if n_heads == 1 else (BATCH_SIZE, n_heads, seq_len, seq_len)
    
    if expected_out_shape == out.shape and expected_score_shape == score.shape:
        print("Quick test passed!")
    else:
        print(f"Test failed! Expected out: {expected_out_shape}, got: {out.shape}")
        print(f"Expected score: {expected_score_shape}, got: {score.shape}")
    
    return out, score, attention

if __name__ == "__main__":
    main()
