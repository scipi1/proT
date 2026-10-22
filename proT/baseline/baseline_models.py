"""
Encoder benchmark models.

All benchmark models share the same pipeline and differ ONLY in the encoder:

    X --(sort: chain order, missing tokens last)--> X_sorted
      --> SharedEmbedding (ModularEmbedding, summation, fixed d_model)  (B, L, d)
      --> Encoder_k (causal, forward direction)                         (B, L, d_k)
      --> Readout_k (encoder specific)                                  (B, d_k)
      --> Adapter Linear(d_k, d_head)                                   (B, d_head)
      --> SharedScalarHead MLP (fixed size)                             (B, n_out)

Same conditions for all encoders:
- tokens are sorted by the "pass" order feature (chain order); tokens with a
  missing value (padding or missing measurement) are moved to the end and
  ignored (packed RNN, zeroed + last-valid readout for TCN/MLP, attention
  masks for the Transformer);
- causal direction: the representation of token i depends only on tokens
  j <= i (rank in the sorted chain). The Transformer uses the token rank as
  causal position, so ties in the order feature do not open a bidirectional
  window;
- no timestamp encoding (time2vec) is allowed in the shared embedding.

Encoders and readouts:
- "proT_bench": Transformer encoder (causal self-attention) + learnable-query
  attention pooling over the valid tokens.
- "GRU"/"LSTM": recurrent encoder, readout = last-layer hidden state at the
  last valid token.
- "TCN": causal dilated convolutions, readout = output at the last valid token.
- "MLP": token-wise MLP + causal cumulative mean, readout at the last valid
  token (= masked mean over the chain). Order-agnostic reference.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from proT.core.modules.embedding import ModularEmbedding
from proT.core.modules.encoder import Encoder, EncoderLayer
from proT.core.modules.attention import ScaledDotAttention, AttentionLayer
from proT.core.modules.extra_layers import Normalization, UniformAttentionMask


# embedding types allowed in the shared benchmark embedding (no time2vec)
ALLOWED_EMBEDS = {"nn_embedding", "linear", "sinusoidal", "mask", "pass", "group"}

BENCHMARK_ENCODERS = ["proT_bench", "GRU", "LSTM", "TCN", "MLP"]


# =============================================================================
# SHARED COMPONENTS
# =============================================================================

def check_shared_embedding(ds_embed_in: dict, comps_embed: str) -> None:
    """Validate the shared embedding config (summation, no time encoding)."""
    if comps_embed != "summation":
        raise ValueError(f"Benchmark embedding must use 'summation', got '{comps_embed}'")
    embeds = [m["embed"] for m in ds_embed_in["modules"]]
    bad = [e for e in embeds if e not in ALLOWED_EMBEDS]
    if bad:
        raise ValueError(
            f"Embedding types {bad} not allowed in the benchmark "
            f"(allowed: {sorted(ALLOWED_EMBEDS)}). Timestamp encoding is excluded.")
    if embeds.count("mask") != 1:
        raise ValueError("Benchmark embedding needs exactly one 'mask' module (value index)")


class SharedScalarHead(nn.Module):
    """
    Fixed-size MLP head shared (same class and size) by all benchmark models.

    Args:
        d_head: input dimension (output of the encoder adapter)
        hidden: hidden layer widths
        n_out: number of scalar targets (1 Ishigami, 2 Dyconex)
        dropout: dropout after each hidden layer
    """
    def __init__(self, d_head: int, hidden: List[int], n_out: int, dropout: float = 0.0):
        super().__init__()
        layers, d = [], d_head
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def last_valid(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Gather h[b, lengths[b]-1] -> (B, D). lengths must be >= 1."""
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


# =============================================================================
# ENCODERS: (x_emb (B,L,d), valid (B,L) bool, lengths (B,)) -> context (B, d_out)
# =============================================================================

class TransformerBenchEncoder(nn.Module):
    """Causal Transformer encoder + learnable-query attention pooling."""
    def __init__(self, d_model: int, e_layers: int, n_heads: int, d_ff: int,
                 d_qk: int, dropout: float, activation: str = "gelu",
                 norm: str = "layer"):
        super().__init__()

        def attn(name):
            return AttentionLayer(
                attention=ScaledDotAttention,
                d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
                d_queries_keys=d_qk, n_heads=n_heads,
                mask_layer=UniformAttentionMask(),
                attention_dropout=dropout, dropout_qkv=dropout,
                register_entropy=False, layer_name=name)

        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(global_attention=attn("enc_self_att"), d_model_enc=d_model,
                             activation=activation, norm=norm, d_ff=d_ff,
                             dropout_ff=dropout, dropout_attn_out=dropout)
                for _ in range(e_layers)],
            norm_layer=Normalization(norm, d_model=d_model),
            emb_dropout=dropout)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attention = attn("pool_att")
        self.d_out = d_model

    def forward(self, x, valid, lengths):
        B, L, _ = x.shape
        miss = (~valid).unsqueeze(-1)                                   # (B, L, 1)
        # causal position = rank in the sorted chain (strictly token-causal)
        pos = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1).expand(B, -1, -1)
        enc_out, _, _ = self.encoder(X=x, mask_miss_k=miss, mask_miss_q=miss,
                                     enc_input_pos=pos, causal_mask=True)
        enc_out = enc_out.masked_fill(miss, 0.0)
        ctx, _, _ = self.pool_attention(
            query=self.pool_query.expand(B, -1, -1), key=enc_out, value=enc_out,
            mask_miss_k=miss, mask_miss_q=None, pos=None, causal_mask=False)
        return ctx.squeeze(1)                                           # (B, d_model)


class RNNBenchEncoder(nn.Module):
    """GRU/LSTM encoder, readout = last-layer hidden state at the last valid token."""
    def __init__(self, cell: str, d_model: int, d_hidden: int, n_layers: int, dropout: float):
        super().__init__()
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.rnn = rnn_cls(input_size=d_model, hidden_size=d_hidden, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.d_out = d_hidden

    def forward(self, x, valid, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        if isinstance(h_n, tuple):                                      # LSTM: (h_n, c_n)
            h_n = h_n[0]
        return h_n[-1]                                                  # (B, d_hidden)


class TCNBenchEncoder(nn.Module):
    """Causal dilated TCN (left padding only), readout at the last valid token."""
    def __init__(self, d_model: int, channels: List[int], kernel: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pads = []
        d_in = d_model
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            self.pads.append((kernel - 1) * dilation)
            self.convs.append(nn.Conv1d(d_in, ch, kernel_size=kernel, dilation=dilation))
            d_in = ch
        self.dropout = nn.Dropout(dropout)
        self.d_out = channels[-1]

    def forward(self, x, valid, lengths):
        h = x.transpose(1, 2)                                           # (B, d, L)
        for conv, pad in zip(self.convs, self.pads):
            h = self.dropout(F.relu(conv(F.pad(h, (pad, 0)))))
        return last_valid(h.transpose(1, 2), lengths)                   # (B, C)


class MLPBenchEncoder(nn.Module):
    """Token-wise MLP + causal cumulative mean, read at the last valid token."""
    def __init__(self, d_model: int, hidden: List[int], dropout: float):
        super().__init__()
        layers, d = [], d_model
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.d_out = d

    def forward(self, x, valid, lengths):
        h = self.mlp(x) * valid.unsqueeze(-1)                           # (B, L, h)
        cum = h.cumsum(dim=1) / valid.cumsum(dim=1).clamp(min=1).unsqueeze(-1)
        return last_valid(cum, lengths)                                 # (B, h)


def build_encoder(encoder_type: str, d_model: int, kwargs: dict) -> nn.Module:
    """Encoder registry."""
    if encoder_type == "proT_bench":
        return TransformerBenchEncoder(d_model=d_model, **kwargs)
    if encoder_type in ("GRU", "LSTM"):
        return RNNBenchEncoder(cell=encoder_type, d_model=d_model, **kwargs)
    if encoder_type == "TCN":
        return TCNBenchEncoder(d_model=d_model, **kwargs)
    if encoder_type == "MLP":
        return MLPBenchEncoder(d_model=d_model, **kwargs)
    raise ValueError(f"Unknown encoder '{encoder_type}', choose from {BENCHMARK_ENCODERS}")


# =============================================================================
# BENCHMARK MODEL
# =============================================================================

class BenchmarkModel(nn.Module):
    """
    Shared embedding -> encoder -> adapter -> shared scalar head.

    Args:
        d_model: fixed embedding dimension (shared by all encoders)
        comps_embed: must be "summation"
        ds_embed_in: ModularEmbedding config of the input tokens
        encoder: {"type": one of BENCHMARK_ENCODERS, "kwargs": {...}}
        head: {"d_head": int, "hidden": list[int], "n_out": int, "dropout": float}
    """
    def __init__(self, d_model: int, comps_embed: str, ds_embed_in: dict,
                 encoder: dict, head: dict):
        super().__init__()
        check_shared_embedding(ds_embed_in, comps_embed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = ModularEmbedding(ds_embed=ds_embed_in, comps=comps_embed, device=device)
        self.encoder_type = encoder["type"]
        self.encoder = build_encoder(self.encoder_type, d_model, dict(encoder.get("kwargs") or {}))
        self.adapter = nn.Linear(self.encoder.d_out, head["d_head"])
        self.head = SharedScalarHead(d_head=head["d_head"], hidden=list(head["hidden"]),
                                     n_out=head["n_out"], dropout=head.get("dropout", 0.0))
        self.n_out = head["n_out"]

    def sort_chain(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort tokens by chain order (first "pass" feature), missing tokens last.

        Returns:
            X_sorted: (B, L, D) reordered input
            valid: (B, L) bool, True for tokens with an observed value
            lengths: (B,) number of valid tokens (clamped to >= 1)
        """
        valid = ~self.embedding.get_mask(X).squeeze(-1)                 # (B, L)
        order = self.embedding.pass_var(X)
        if order is not None:
            key = torch.nan_to_num(order[:, :, 0].float(), nan=float("inf"))
        else:
            key = torch.arange(X.size(1), device=X.device).float().expand(X.size(0), -1)
        key = key.masked_fill(~valid, float("inf"))
        idx = torch.argsort(key, dim=1, stable=True)                    # (B, L)
        X_sorted = X.gather(1, idx.unsqueeze(-1).expand(-1, -1, X.size(-1)))
        valid = valid.gather(1, idx)
        lengths = valid.sum(dim=1).clamp(min=1)
        return X_sorted, valid, lengths

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Context vector of the chain (B, d_k), before the adapter."""
        X_sorted, valid, lengths = self.sort_chain(X)
        x_emb = self.embedding(X_sorted).masked_fill(~valid.unsqueeze(-1), 0.0)
        return self.encoder(x_emb, valid, lengths)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, L, D) -> (B, n_out)."""
        return self.head(self.adapter(self.encode(X)))

    def param_counts(self) -> Dict[str, int]:
        """Trainable parameters per component."""
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"embedding": count(self.embedding), "encoder": count(self.encoder),
                "adapter": count(self.adapter), "head": count(self.head), "total": count(self)}
