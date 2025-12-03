import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.core.modules.embedding import ModularEmbedding
from proT.baseline.s6 import BiMamba



class RNN(nn.Module):
    def __init__(
        self, 
        model, 
        d_in, 
        d_emb, 
        d_hidden, 
        n_layers,
        dropout,
        comps_embed, 
        ds_embed_in, 
        ds_embed_trg
        ):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # define embeddings
        self.input_embedding = ModularEmbedding(
            ds_embed=ds_embed_in,
            comps=comps_embed,
            device=device)
        
        self.target_embedding = ModularEmbedding(
            ds_embed=ds_embed_trg,
            comps=comps_embed,
            device=device)
        
        # select RNN model
        if model == "GRU":
            nn_module = nn.GRU
        elif model == "LSTM":
            nn_module = nn.LSTM
            
        self.rnn = nn_module(
            input_size=d_in, 
            hidden_size=d_hidden, 
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
            )
        
        # define final projection layer
        self.head = nn.Linear(d_emb+n_layers*d_hidden, 1)
    
    
    def _mask_to_lengths(self, mask, min_valid_length=2):
        """
        Convert boolean mask to sequence lengths for pack_padded_sequence.
        
        Args:
            mask: (B, L, 1) - True means missing/padding, False means valid
            min_valid_length: Minimum number of valid positions required
        
        Returns:
            lengths: (B,) tensor with valid positions per sequence, or None if validation fails
            is_valid_batch: Boolean indicating if batch is safe for packing
        """
        if mask is None:
            return None, True
        
        is_valid = ~mask.squeeze(-1)  # (B, L) - True = valid, False = missing
        lengths = is_valid.sum(dim=1)  # Count valid positions per sequence
        
        # Check for sequences with insufficient valid data
        invalid_sequences = lengths < min_valid_length
        num_invalid = invalid_sequences.sum().item()
        
        if num_invalid > 0:
            # Log problematic sequences
            print(f"[RNN WARNING] Found {num_invalid} sequences with < {min_valid_length} valid positions")
            print(f"[RNN WARNING] Length distribution: min={lengths.min().item()}, "
                  f"max={lengths.max().item()}, mean={lengths.float().mean().item():.2f}")
            print(f"[RNN WARNING] Invalid sequence indices: {invalid_sequences.nonzero(as_tuple=True)[0].tolist()}")
            
            # Return None to trigger fallback to unpacked behavior
            return None, False
        
        # Additional safety: check for zero-length sequences (shouldn't happen with min_valid_length check)
        if (lengths == 0).any():
            print(f"[RNN ERROR] Found sequences with zero valid positions!")
            return None, False
        
        return lengths, True
        
        
    def forward(self, X, y):           
        
        # Get mask from input using existing infrastructure
        mask_in = self.input_embedding.get_mask(X)  # (B, L_in, 1) or None
        
        # Embed inputs
        X_emb = self.input_embedding(X)  # (B, L_in, d_in)
        y_emb = self.target_embedding(y)  # (B, L_out, d_emb)
        B, L_in, _ = X_emb.shape
        _, L_out, _ = y_emb.shape
        
        # Check for NaN in embeddings (diagnostic)
        if torch.isnan(X_emb).any():
            print(f"[RNN ERROR] NaN detected in input embeddings! Shape: {X_emb.shape}")
            nan_count = torch.isnan(X_emb).sum().item()
            print(f"[RNN ERROR] Total NaN values: {nan_count}")
        
        # Convert mask to lengths for packing with validation
        X_lengths, is_valid_batch = self._mask_to_lengths(mask_in, min_valid_length=2)
        
        # RNN forward pass with optional packing
        use_packing = X_lengths is not None and is_valid_batch
        
        if use_packing:
            # Log that we're using packing (only occasionally to avoid spam)
            if torch.rand(1).item() < 0.01:  # Log ~1% of batches
                print(f"[RNN INFO] Using packed sequences - Batch size: {B}, "
                      f"Length stats: min={X_lengths.min().item()}, max={X_lengths.max().item()}, "
                      f"mean={X_lengths.float().mean().item():.2f}")
            
            # Pack sequences to skip padded/missing positions
            X_packed = pack_padded_sequence(
                X_emb, X_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # RNN last hidden state (context) from packed sequence
            if isinstance(self.rnn, nn.GRU):
                _, h_n = self.rnn(X_packed)
            else:  # LSTM
                _, (h_n, _) = self.rnn(X_packed)
        else:
            # Fallback to unpacked behavior
            if not is_valid_batch:
                print(f"[RNN INFO] Falling back to unpacked sequences due to invalid batch")
            
            if isinstance(self.rnn, nn.GRU):
                _, h_n = self.rnn(X_emb)
            else:  # LSTM
                _, (h_n, _) = self.rnn(X_emb)
        
        # Check for NaN in hidden states (diagnostic)
        if torch.isnan(h_n).any():
            print(f"[RNN ERROR] NaN detected in hidden states! Shape: {h_n.shape}")
            print(f"[RNN ERROR] Using packing: {use_packing}")
            
        ctx = h_n.reshape(h_n.size(1), -1)    # (B, n_layers*d_hidden)
        
        # Check context validity (diagnostic)
        if torch.isnan(ctx).any():
            print(f"[RNN ERROR] NaN detected in context after reshape! Shape: {ctx.shape}")
        
        # expand context to match the target embeddings
        ctx_expanded = ctx.unsqueeze(1).expand(-1, L_out, -1)  # (B, L_out, n_layers*d_hidden)
        
        # concatenate target embeddings with context
        z = torch.cat((y_emb, ctx_expanded), dim=-1)
        
        # final projection
        out = self.head(z).squeeze(-1)
        
        # Final check for NaN in output
        if torch.isnan(out).any():
            print(f"[RNN ERROR] NaN detected in final output! Shape: {out.shape}")
            print(f"[RNN ERROR] NaN count: {torch.isnan(out).sum().item()}")
        
        return out



class TCN(nn.Module):
    def __init__(
        self,
        model: str,
        d_in: int,           # input embedding dim
        d_emb: int,          # target embedding dim
        channels: list[int], # conv widths e.g. [64, 64, 128]
        dropout: float,
        comps_embed: str,
        ds_embed_in: dict,
        ds_embed_trg: dict,
        kernel: int = 3,
        d_out: int = 1
        ):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embedding  = ModularEmbedding(
            ds_embed_in,  
            comps_embed,
            device=device
            )
        self.target_embedding = ModularEmbedding(
            ds_embed_trg, 
            comps_embed,
            device=device
            )

        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            pad = (kernel - 1) * dilation        # causal padding
            layers += [
                nn.Conv1d(
                    in_channels=d_in if i == 0 else channels[i - 1],
                    out_channels=ch,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding=pad
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ]
        self.tcn = nn.Sequential(*layers)

        in_feats = channels[-1] + d_emb         # ctx + tgt_emb per step
        self.head = nn.Linear(in_feats, d_out)  # shared across time steps

    
    def forward(self, X, y):

        # embed input & run causal TCN
        x_emb   = self.input_embedding(X)       # B × L_in × d_in_emb
        x_conv  = x_emb.permute(0, 2, 1)        # B × d_in_emb × L_in
        conv_out = self.tcn(x_conv)             # B × C × (L_in + pad)
        conv_out = conv_out[..., :X.size(1)]    # trim future‐leak padding
        ctx = conv_out[..., -1]                 # B × C (last time step)

        # embed target-side features
        y_emb = self.target_embedding(y)
        _, L_out, _ = y_emb.shape

        # broadcast context and concat
        ctx_exp = ctx.unsqueeze(1).expand(-1, L_out, -1)  # B × L_out × C
        z = torch.cat([y_emb, ctx_exp], dim=-1)           # B × L_out × (C+d_emb)

        # 4) time-distributed prediction
        out = self.head(z).squeeze(-1)  # B × L_out
        return out



class MLP(nn.Module):
    def __init__(
        self,
        model,
        d_in: int,
        d_emb: int,
        dropout: float,
        comps_embed: str,
        ds_embed_in: dict,
        ds_embed_trg: dict,
        hidden: list[int] = [256, 256],
        d_out: int = 1
        ):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embedding  = ModularEmbedding(
            ds_embed=ds_embed_in,  
            comps=comps_embed, 
            device=device)
        
        self.target_embedding = ModularEmbedding(
            ds_embed=ds_embed_trg, 
            comps=comps_embed, 
            device=device)

        # mean-pool along the sequence length L_in
        self.pool = lambda x: x.mean(dim=1)           # B × d_in

        #  MLP over pooled context
        layers, in_dim = [], d_in
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p=dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)             # ctx: B × h_last

        # final projection layer
        self.head = nn.Linear(hidden[-1] + d_emb, d_out)

    def forward(self, X, y):
        
        # input and target embeddings
        X = self.input_embedding(X)                         # input emb --> B × L_in × d_in_emb
        yemb = self.target_embedding(y)                     # target emb --> B × L_out × d_emb
        _, L_out, _ = yemb.shape

        # MLP forward pass
        ctx  = self.mlp(self.pool(X))                       # input emb --> B × h_last
        
        # expand context to match the target embeddings
        ctx_exp = ctx.unsqueeze(1).expand(-1, L_out, -1)    # input emb --> B × L_out × h_last
        
        # concatenate target embeddings with context
        z   = torch.cat([yemb, ctx_exp], -1)                # input emb --> B × L_out × (h_last+d_emb)
        
        # final projection
        out = self.head(z).squeeze(-1)
        
        return out



class S6(nn.Module):
    def __init__(
        self,
        model: str,
        d_in: int,           # input embedding dim
        d_emb: int,          # target embedding dim
        d_model: int,        # Mamba hidden dimension
        n_layers: int,       # number of Mamba layers
        dropout: float,
        comps_embed: str,
        ds_embed_in: dict,
        ds_embed_trg: dict,
        d_state: int = 16,   # SSM state dimension (paper default)
        d_conv: int = 4,     # convolution kernel size (paper default)
        expand: int = 2,     # expansion factor (paper default)
        d_out: int = 1
        ):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embedding  = ModularEmbedding(
            ds_embed=ds_embed_in,  
            comps=comps_embed, 
            device=device)
        
        self.target_embedding = ModularEmbedding(
            ds_embed=ds_embed_trg, 
            comps=comps_embed, 
            device=device)
        
        # bidirectional Mamba model
        self.bimamba = BiMamba(
            d_model=d_in,        # input embedding dimension
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # final projection layer
        # BiMamba outputs d_model*2 due to bidirectionality
        self.head = nn.Linear(d_in * 2 + d_emb, d_out)
    
    def forward(self, X, y):
        
        # input and target embeddings
        X_emb = self.input_embedding(X)                         # B × L_in × d_in_emb
        yemb = self.target_embedding(y)                         # B × L_out × d_emb
        
        # DIAGNOSTIC: Check embeddings
        if torch.isnan(X_emb).any():
            print(f"[S6 DEBUG] NaN detected in input embeddings! Shape: {X_emb.shape}")
            print(f"[S6 DEBUG] X_emb stats: min={X_emb.min()}, max={X_emb.max()}, mean={X_emb.mean()}")
        if torch.isnan(yemb).any():
            print(f"[S6 DEBUG] NaN detected in target embeddings! Shape: {yemb.shape}")
            print(f"[S6 DEBUG] yemb stats: min={yemb.min()}, max={yemb.max()}, mean={yemb.mean()}")
        
        _, L_out, _ = yemb.shape
        
        # BiMamba forward pass with mean pooling
        ctx = self.bimamba.get_context(X_emb)                   # B × (d_in*2)
        
        # DIAGNOSTIC: Check context from BiMamba
        if torch.isnan(ctx).any():
            print(f"[S6 DEBUG] NaN detected in BiMamba context! Shape: {ctx.shape}")
            print(f"[S6 DEBUG] ctx stats: min={ctx.min()}, max={ctx.max()}, mean={ctx.mean()}")
        else:
            print(f"[S6 DEBUG] BiMamba context OK - Shape: {ctx.shape}, min={ctx.min()}, max={ctx.max()}, mean={ctx.mean()}")
        
        # expand context to match the target embeddings
        ctx_exp = ctx.unsqueeze(1).expand(-1, L_out, -1)    # B × L_out × (d_in*2)
        
        # concatenate target embeddings with context
        z = torch.cat([yemb, ctx_exp], -1)                  # B × L_out × (d_emb+d_in*2)
        
        # DIAGNOSTIC: Check concatenation
        if torch.isnan(z).any():
            print(f"[S6 DEBUG] NaN detected after concatenation! Shape: {z.shape}")
        
        # final projection
        out = self.head(z).squeeze(-1)
        
        # DIAGNOSTIC: Check final output
        if torch.isnan(out).any():
            print(f"[S6 DEBUG] NaN detected in final output! Shape: {out.shape}")
            print(f"[S6 DEBUG] out stats: min={out.min()}, max={out.max()}, mean={out.mean()}")
        else:
            print(f"[S6 DEBUG] Final output OK - Shape: {out.shape}, min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
        
        return out
