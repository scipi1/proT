import torch
import torch.nn as nn
import os

# Try official mamba-ssm first (CUDA optimized)
# Fall back to local pure PyTorch implementation if not available
try:
    from mamba_ssm import Mamba
    MAMBA_BACKEND = "mamba-ssm (CUDA optimized)"
except ImportError:
    from proT.baseline.s6.mamba_pytorch import Mamba
    MAMBA_BACKEND = "mamba-pytorch (pure PyTorch fallback)"
    # Print warning only once per process using environment variable
    if not os.environ.get('MAMBA_PYTORCH_WARNING_SHOWN'):
        print(
            "WARNING: Using pure PyTorch implementation of Mamba (slower but functional). "
            "For faster performance on GPU clusters, install mamba-ssm with CUDA support: "
            "pip install mamba-ssm causal-conv1d"
        )
        os.environ['MAMBA_PYTORCH_WARNING_SHOWN'] = '1'

# Print backend info only once per process
if not os.environ.get('MAMBA_BACKEND_LOGGED'):
    print(f"S6 Mamba backend: {MAMBA_BACKEND}")
    os.environ['MAMBA_BACKEND_LOGGED'] = '1'


class BiMamba(nn.Module):
    """
    Bidirectional Mamba (S6) wrapper for sequence modeling.
    
    Processes sequences in both forward and backward directions using
    separate Mamba blocks, then concatenates the outputs.
    
    Args:
        d_model: Hidden dimension size
        n_layers: Number of Mamba layers to stack
        d_state: SSM state dimension (default: 16, from paper)
        d_conv: Convolution kernel size (default: 4, from paper)
        expand: Expansion factor (default: 2, from paper)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Forward direction Mamba layers with Layer Normalization
        self.forward_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization for forward direction (one per layer)
        self.forward_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Backward direction Mamba layers with Layer Normalization
        self.backward_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization for backward direction (one per layer)
        self.backward_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bidirectional Mamba.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Bidirectional hidden states of shape (B, L, d_model*2)
        """
        # DIAGNOSTIC: Check input
        if torch.isnan(x).any():
            print(f"[BiMamba DEBUG] NaN in input! Shape: {x.shape}, min={x.min()}, max={x.max()}")
        
        # Forward direction
        h_forward = x
        for i, (layer, norm) in enumerate(zip(self.forward_layers, self.forward_norms)):
            h_forward = layer(h_forward)
            h_forward = norm(h_forward)  # Apply LayerNorm after each Mamba layer
            # DIAGNOSTIC: Check after each forward layer
            if torch.isnan(h_forward).any():
                print(f"[BiMamba DEBUG] NaN after forward layer {i}! Shape: {h_forward.shape}")
                print(f"[BiMamba DEBUG] Stats: min={h_forward[~torch.isnan(h_forward)].min() if (~torch.isnan(h_forward)).any() else 'all nan'}, max={h_forward[~torch.isnan(h_forward)].max() if (~torch.isnan(h_forward)).any() else 'all nan'}")
            else:
                print(f"[BiMamba DEBUG] Forward layer {i} OK - min={h_forward.min():.4f}, max={h_forward.max():.4f}, mean={h_forward.mean():.4f}")
            h_forward = self.dropout(h_forward)
        
        # Backward direction (reverse sequence)
        h_backward = torch.flip(x, dims=[1])  # Reverse along sequence dimension
        for i, (layer, norm) in enumerate(zip(self.backward_layers, self.backward_norms)):
            h_backward = layer(h_backward)
            h_backward = norm(h_backward)  # Apply LayerNorm after each Mamba layer
            # DIAGNOSTIC: Check after each backward layer
            if torch.isnan(h_backward).any():
                print(f"[BiMamba DEBUG] NaN after backward layer {i}! Shape: {h_backward.shape}")
            else:
                print(f"[BiMamba DEBUG] Backward layer {i} OK - min={h_backward.min():.4f}, max={h_backward.max():.4f}, mean={h_backward.mean():.4f}")
            h_backward = self.dropout(h_backward)
        h_backward = torch.flip(h_backward, dims=[1])  # Reverse back to original order
        
        # Concatenate forward and backward
        h_bi = torch.cat([h_forward, h_backward], dim=-1)  # (B, L, d_model*2)
        
        # DIAGNOSTIC: Check final output
        if torch.isnan(h_bi).any():
            print(f"[BiMamba DEBUG] NaN in final bidirectional output! Shape: {h_bi.shape}")
        
        return h_bi
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get pooled context vector from sequence.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Context vector of shape (B, d_model*2)
        """
        h_bi = self.forward(x)  # (B, L, d_model*2)
        ctx = h_bi.mean(dim=1)  # Mean pool across sequence: (B, d_model*2)
        return ctx
