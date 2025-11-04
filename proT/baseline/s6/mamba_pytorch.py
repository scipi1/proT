"""
Pure PyTorch implementation of Mamba (Selective SSM)
Simplified version for local testing without CUDA compilation requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Simplified Selective State Space Model (S6) in pure PyTorch.
    Based on Mamba architecture but without CUDA kernels.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Initialize A (diagonal state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) output tensor
        """
        B, L, D = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution (causal)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Causal masking
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM step
        y = self.selective_scan(x)
        
        # Gate
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan mechanism.
        
        Args:
            x: (B, L, d_inner) input
        Returns:
            (B, L, d_inner) output
        """
        B, L, D = x.shape
        
        # Get SSM parameters from input
        x_dbl = self.x_proj(x)  # (B, L, d_state*2)
        delta, B_param = x_dbl.chunk(2, dim=-1)  # Each (B, L, d_state)
        
        # Delta projection
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretization: simplified Euler method
        # A_bar = exp(delta * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        
        # B_bar = delta * B
        deltaB = delta.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Selective scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            # State update: h = A_bar * h + B_bar * x
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            # Output: y = h (simplified, normally would have C matrix)
            y = h.sum(dim=-1)  # (B, d_inner)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class Mamba(nn.Module):
    """
    Pure PyTorch Mamba block, API-compatible with mamba-ssm.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) output tensor
        """
        return x + self.ssm(self.norm(x))
