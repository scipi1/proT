from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from prochain_transformer.modules.attention import ScaledDotAttention, AttentionLayer
from prochain_transformer.modules.extra_layers import UniformAttentionMask

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
    attention = ScaledDotAttention

    x = torch.rand(BATCH_SIZE,seq_len,d_model)
    
    attention = AttentionLayer(
        attention=attention, 
        d_model=d_model, 
        d_queries_keys=d_queries_keys, 
        n_heads=n_heads, 
        mask_layer=UniformAttentionMask,
        attention_dropout=0,
        dropout_qkv=dropout_qkv)
    
    out, score = attention(query=x, key=x, value=x, mask="NAIM")
    
    print(f"Attention output shape: {out.shape}")
    print(f"Attention score shape: {score.shape}")
    
    if (BATCH_SIZE,seq_len,d_model) == out.shape and (BATCH_SIZE,n_heads,seq_len,seq_len) == score.shape:
        print("Quick test passed!")
    
    return out, score, attention

if __name__ == "__main__":
    main()
    