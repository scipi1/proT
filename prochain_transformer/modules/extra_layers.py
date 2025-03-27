import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np







class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps) * self.scale
        x = x / n * self.g
        return x


class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        elif method == "scale":
            self.norm = ScaleNorm(d_model)
        
        # not needed now
        # elif method == "power":
        #     self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        
        elif method == "none":
            self.norm = lambda x: x
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method
        
    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)
    
    
class UniformAttentionMask(nn.Module):
    def __init__(self) -> None:
        super(UniformAttentionMask,self).__init__()
    
    def forward(self, attention_scores:torch.Tensor, mask:torch.Tensor,mask_val=-float("inf")):
        """
        Applies masking to the attention scores.
        
        Args:
        - attention_scores: Tensor of shape (batch_size, N_queries, N_keys).
        - mask: Boolean tensor of shape (N_keys), where False means the corresponding key should be masked (zeroed).
        
        Returns:
        - masked_attention_scores: Tensor with masked attention scores.
        """

        assert attention_scores.shape[-1] == len(mask), AssertionError(f"Got mask of length {len(mask)}, expected {attention_scores.shape[-1]}")
        
        # Ensure the mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        # Ensure the mask is on the same device as the attention scores
        if mask.device != attention_scores.device:
            mask = mask.to(attention_scores.device)
        
        # Convert boolean mask to float and expand it to match attention_scores
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N_keys)
        mask=mask.expand_as(attention_scores)
        # Apply the mask to zero out the attention scores where mask is False
        
        return attention_scores.masked_fill(mask, mask_val)
    
class NAIMAttentionMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, attention_scores:torch.Tensor, mask:torch.Tensor,mask_val=-torch.inf):
        """
        Applies masking to the attention scores.
        
        Args:
        - attention_scores: Tensor of shape (batch_size, N_queries, N_keys).
        - mask: Boolean tensor of shape (N_keys), where False means the corresponding key should be masked (zeroed).
        
        Returns:
        - masked_attention_scores: Tensor with masked attention scores.
        """

        assert attention_scores.shape[-1] == len(mask), AssertionError(f"Got mask of length {len(mask)}, expected {attention_scores.shape[-1]}")
        
        # Ensure the mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        # Ensure the mask is on the same device as the attention scores
        if mask.device != attention_scores.device:
            mask = mask.to(attention_scores.device)
        
        # Convert boolean mask to float and expand it to match attention_scores
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N_keys)
        mask=mask.expand_as(attention_scores)
        # Apply the mask to zero out the attention scores where mask is False
        
        return attention_scores.masked_fill(torch.isnan(attention_scores), mask_val)