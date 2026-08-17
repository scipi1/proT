# Standard library imports
import warnings
from typing import Optional, Tuple, List

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .modules.attention import ScaledDotAttention, AttentionLayer
from .modules.embedding import ModularEmbedding
from .modules.encoder import Encoder, EncoderLayer
from .modules.extra_layers import Normalization, UniformAttentionMask


class ProTEncoder(nn.Module):
    """
    Encoder-only proT variant.

    The transformer encoder processes the input sequence; a learnable query
    token then pools the encoder outputs via cross-attention into a single
    context vector per sample. Two head variants are supported:

    - head_type="scalar": the context vector is projected to n_scalars *
      out_dim values (e.g. mu, raw_sigma per scalar target). Used when the
      dataset provides a D-dimensional vector of scalar targets per sample
      (e.g. IST slope, fitting coefficients) instead of a sequential target.
      With n_scalars == d_hidden the output dimension matches the GRU hidden
      dimension, so switching between the two variants is seamless.

    - head_type="gru": the context vector initializes the hidden state of a
      GRU (h0), which is then unrolled over the target positions. Per-step
      inputs are the target covariates (position, variable, value,
      given-indicator) embedded with the existing ModularEmbedding. The value
      channel is fed with the real target where given (teacher forcing /
      "real IST" at inference) and with the predicted mean elsewhere
      (autoregressive rollout). A per-step linear head produces out_dim
      channels (mu, raw_sigma), so the probabilistic losses
      (PositionWeightedMSE / CensoredGaussianNLL) apply unchanged.

    The forward signature mirrors ProT.forward: decoder attention/entropy
    entries are returned as empty lists so that callbacks and plotting code
    keep working unchanged.
    """

    def __init__(
        self,
        model: str,

        # embeddings
        ds_embed_enc,
        comps_embed_enc,
        ds_embed_dec,
        comps_embed_dec,

        # attention
        enc_attention_type,
        enc_mask_type,
        n_heads: int,
        enc_causal_mask: bool,

        # dropout
        dropout_emb: float,
        dropout_data: float,  # kept for config compatibility (unused)
        dropout_attn_out: float,
        dropout_ff: float,
        enc_dropout_qkv: float,
        enc_attention_dropout: float,

        # options
        e_layers: int,
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        out_dim: int,
        d_ff: int,
        d_model_enc: int,
        d_model_dec: int,
        d_qk: int,

        # encoder-only specific
        enc_group_mask: bool = False,
        head_type: str = "gru",            # "scalar" | "gru"
        output_dim: Optional[int] = None,  # scalar head: number of scalar targets
        d_hidden: int = 128,               # GRU hidden dimension
        rnn_n_layers: int = 1,
        rnn_dropout: float = 0.0,
        dec_val_idx: int = 3,              # value channel in the target tensor
        dec_pos_idx: int = 1,              # position channel in the target tensor
        ):
        super().__init__()

        assert head_type in ["scalar", "gru"], \
            f"head_type must be 'scalar' or 'gru', got {head_type}"
        if head_type == "scalar":
            assert output_dim is not None and output_dim > 0, \
                "output_dim must be a positive integer for the scalar head"

        self.head_type = head_type
        self.output_dim = output_dim
        self.d_hidden = d_hidden
        self.rnn_n_layers = rnn_n_layers
        self.out_dim = out_dim
        self.dec_val_idx = dec_val_idx
        self.dec_pos_idx = dec_pos_idx

        # encoder input embedding
        self.enc_embedding = ModularEmbedding(
            ds_embed=ds_embed_enc,
            comps=comps_embed_enc,
            device=device)

        # attention options (encoder self-attention)
        attn_enc_kwargs = {
            "n_heads"           : n_heads,
            "d_queries_keys"    : d_qk,
            "d_model_queries"   : d_model_enc,
            "d_model_keys"      : d_model_enc,
            "d_model_values"    : d_model_enc,
            "attention_type"    : enc_attention_type,
            "mask_type"         : enc_mask_type,
            "dropout_qkv"       : enc_dropout_qkv,
            "attention_dropout" : enc_attention_dropout,
            "register_entropy"  : True,
            "layer_name"        : "enc_self_att"
        }

        self.enc_causal_mask = enc_causal_mask
        self.enc_group_mask = enc_group_mask

        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    global_attention=self._attn(**attn_enc_kwargs),
                    d_model_enc=d_model_enc,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm
                    ) for _ in range(e_layers)],
            norm_layer=Normalization(norm, d_model=d_model_enc) if use_final_norm else None,
            emb_dropout=dropout_emb
            )

        # attention pooling: learnable query token attends over encoder outputs
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model_enc) * 0.02)
        self.pool_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model_enc,
            d_model_keys=d_model_enc,
            d_model_values=d_model_enc,
            d_queries_keys=d_qk,
            n_heads=n_heads,
            mask_layer=UniformAttentionMask() if enc_mask_type == "Uniform" else None,
            attention_dropout=enc_attention_dropout,
            dropout_qkv=enc_dropout_qkv,
            register_entropy=False,
            layer_name="pool_att"
            )

        # prediction heads
        if head_type == "scalar":
            assert output_dim is not None, "output_dim must be set for the scalar head"
            self.scalar_head = nn.Linear(d_model_enc, output_dim * out_dim, bias=True)
        else:
            # context -> initial hidden state of the GRU
            self.h0_projection = nn.Linear(d_model_enc, rnn_n_layers * d_hidden, bias=True)
            # target covariate embedding for the GRU inputs
            self.dec_embedding = ModularEmbedding(
                ds_embed=ds_embed_dec,
                comps=comps_embed_dec,
                device=device)
            self.rnn = nn.GRU(
                input_size=d_model_dec + 1,
                hidden_size=d_hidden,
                num_layers=rnn_n_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_n_layers > 1 else 0.0
                )
            self.rnn_head = nn.Linear(d_hidden, out_dim, bias=True)

        # reconstruction head (kept for parity with ProT)
        self.reconstructor = nn.Linear(d_model_enc, 1, bias=True)

    def _attn(
        self,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        n_heads: int,
        d_queries_keys: int,
        attention_type: str,
        mask_type: str,
        dropout_qkv: float,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str
        ):

        assert attention_type in ["Kernel", "ScaledDotProduct"]

        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention

        mask_layer = None
        if mask_type is not None:
            if mask_type == "Uniform":
                mask_layer = UniformAttentionMask()

        att = AttentionLayer(
            attention=attention_module,
            d_model_queries=d_model_queries,
            d_model_keys=d_model_keys,
            d_model_values=d_model_values,
            d_queries_keys=d_queries_keys,
            n_heads=n_heads,
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv,
            register_entropy=register_entropy,
            layer_name=layer_name
            )

        return att

    def _encode(self, input_tensor):
        """Embed input and run the transformer encoder."""
        enc_input = self.enc_embedding(X=input_tensor)
        enc_input_pos = self.enc_embedding.pass_var(X=input_tensor)
        enc_mask = self.enc_embedding.get_mask(X=input_tensor)
        enc_group = self.enc_embedding.get_group_var(X=input_tensor)

        if enc_input_pos is None and self.enc_causal_mask == True:
            self.enc_causal_mask = False
            warnings.warn(
                "Encoder causal_mask is set to True but encoder received null input positions. "
                "To use causal masking, configure a position variable with 'embed': 'pass' in "
                "ds_embed_enc. Causal mask has been disabled."
            )

        if enc_group is None and self.enc_group_mask == True:
            self.enc_group_mask = False
            warnings.warn(
                "Encoder group_mask is set to True but encoder received null group ids. "
                "To use group masking, configure a grouping variable with 'embed': 'group' in "
                "ds_embed_enc. Group mask has been disabled."
            )

        enc_out, enc_self_att, enc_self_ent = self.encoder(
            X=enc_input,
            mask_miss_k=enc_mask,
            mask_miss_q=enc_mask,
            enc_input_pos=enc_input_pos,
            causal_mask=self.enc_causal_mask,
            group_ids=enc_group,
            group_mask=self.enc_group_mask
            )

        return enc_out, enc_self_att, enc_self_ent, enc_mask

    def _pool(self, enc_out, enc_mask):
        """Attention-pool the encoder outputs into a single context vector."""
        B = enc_out.size(0)
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model_enc)
        context, pool_att, _ = self.pool_attention(
            query=query,
            key=enc_out,
            value=enc_out,
            mask_miss_k=enc_mask,
            mask_miss_q=None,
            pos=None,
            causal_mask=False
            )
        return context.squeeze(1), pool_att  # (B, d_model_enc)

    def forward(
        self,
        input_tensor,
        target_tensor,
        trg_pos_mask=None
        ):
        """
        Args:
            input_tensor: encoder input (B, L_enc, D_enc)
            target_tensor: target tensor (B, L_dec, D_dec) for the GRU head;
                ignored by the scalar head (may be None).
            trg_pos_mask: optional (B, L_dec, 1) bool mask, True where the
                target value must NOT be shown to the GRU (i.e. predicted).
                If None, no teacher forcing is applied.

        Returns:
            forecast_out: (B, L_dec, out_dim) for the GRU head,
                          (B, n_scalars, out_dim) for the scalar head
            recon_out: (B, L_enc, 1)
            (enc_self_att, [], pool_att): attention lists (decoder entries empty)
            enc_mask: encoder missing-value mask
            (enc_self_ent, [], []): entropy lists (decoder entries empty)
        """
        enc_out, enc_self_att, enc_self_ent, enc_mask = self._encode(input_tensor)
        context, pool_att = self._pool(enc_out, enc_mask)

        if self.head_type == "scalar":
            forecast_out = self.scalar_head(context).view(
                context.size(0), self.output_dim, self.out_dim)
        else:
            forecast_out = self._gru_rollout(context, target_tensor, trg_pos_mask)

        recon_out = self.reconstructor(enc_out)

        return forecast_out, recon_out, (enc_self_att, [], [pool_att]), enc_mask, \
            (enc_self_ent, [], [])

    def _gru_rollout(self, context, target_tensor, trg_pos_mask):
        """
        Initialize the GRU hidden state from the pooled context and unroll over
        the target positions. Where the target is given (trg_pos_mask == False)
        the real value is fed; elsewhere the predicted mean is fed back
        (autoregressive rollout).
        """
        B, L_dec, _ = target_tensor.shape

        # initial hidden state from the attention-pooled context
        h0 = torch.tanh(self.h0_projection(context))          # (B, n_layers * d_hidden)
        h = h0.view(B, self.rnn_n_layers, self.d_hidden) \
              .permute(1, 0, 2).contiguous()                  # (n_layers, B, d_hidden)

        # embed the target covariates (value channel zeroed; it is fed separately)
        dec_cov = target_tensor.clone()
        dec_cov[:, :, self.dec_val_idx] = 0.0
        dec_emb = self.dec_embedding(X=dec_cov)               # (B, L_dec, d_model_dec)

        # value channel to feed at each step: real where given, predicted elsewhere
        if trg_pos_mask is not None:
            given = torch.logical_not(trg_pos_mask).squeeze(-1)  # (B, L_dec) bool
        else:
            given = torch.zeros(B, L_dec, dtype=torch.bool, device=target_tensor.device)

        real_val = torch.nan_to_num(target_tensor[:, :, self.dec_val_idx])

        outputs = []
        prev_val = real_val[:, 0]  # first step input value (overwritten below if not given)
        for t in range(L_dec):
            step_val = torch.where(given[:, t], real_val[:, t], prev_val)
            step_in = torch.cat(
                [dec_emb[:, t, :], step_val.unsqueeze(-1)], dim=-1).unsqueeze(1)  # (B, 1, d_model_dec+1)
                # note: d_model_dec must match the GRU input_size; the value channel is
                # appended after the modular embedding, so the GRU input size is
                # d_model_dec + 1. See __init__ (input_size=d_model_dec + 1).
            out, h = self.rnn(step_in, h)                        # (B, 1, d_hidden)
            pred = self.rnn_head(out.squeeze(1))                 # (B, out_dim)
            outputs.append(pred)
            prev_val = pred[:, 0]                                # predicted mean fed back

        return torch.stack(outputs, dim=1)                       # (B, L_dec, out_dim)
