# Standard library imports
import math
import warnings
from typing import Optional, Tuple, List

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from .modules.attention import ScaledDotAttention, AttentionLayer
from .modules.embedding import ModularEmbedding
from .modules.encoder import Encoder, EncoderLayer
from .modules.extra_layers import Normalization, UniformAttentionMask
from .modules.gru_ops import gru_cell, stack_gru_params, hidden_gates
from .innovation_update import InnovationUpdate
from .spline_head import SplineHead


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

    - head_type="gru_per_var": like "gru", but with one independent GRU (and
      one independent linear head) per target variable. The pooled context is
      projected to n_dec_vars initial hidden states, so each variable's
      rollout starts from its own encoder-derived h0 and no hidden state
      flows across variables. The target sequence is split by the variable
      channel (dec_var_idx, ids 1..n_dec_vars) and the per-variable
      predictions are scattered back to the original positions, so the
      output shape (B, L_dec, out_dim) is unchanged.

    GRU output modes (rnn_output_mode):

    - "absolute" (default): per-step (mu, raw_sigma) of the absolute value.
    - "delta": the head outputs per-step increments (delta_mu, raw_delta_var)
      in the native space of the likelihood (log-space for log-space losses,
      y-space otherwise). The rollout accumulates them in-graph into the
      cumulative distribution (mu_cum = sum of delta_mu, var_cum = sum of
      softplus(delta_var)), which is the model's output: a sum of Gaussian
      increments is exactly Gaussian, so the probabilistic losses (Gaussian /
      lognormal, censored or not) apply to the cumulative channels unchanged,
      and censoring/failure probabilities are computed on the absolute
      signal. After a given (teacher-forced) position the state is updated to
      the true value with zero variance (a measurement), so uncertainty
      starts accumulating only over the predicted tail. rnn_monotone is an
      independent add-on constraining delta_mu >= 0. The per-step increments
      of the latest forward are exposed as self.last_step_deltas (B, L, 2)
      for the increment NLL.

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
        head_type: str = "gru",            # "scalar" | "gru" | "gru_per_var"
        output_dim: Optional[int] = None,  # scalar head: number of scalar targets
        d_hidden: int = 128,               # GRU hidden dimension
        rnn_n_layers: int = 1,
        rnn_dropout: float = 0.0,
        dec_val_idx: int = 3,              # value channel in the target tensor
        dec_pos_idx: int = 1,              # position channel in the target tensor
        n_dec_vars: Optional[int] = None,  # gru_per_var: number of target variables
        dec_var_idx: Optional[int] = None, # gru_per_var: variable channel in the target tensor

        # GRU head options
        rnn_output_mode: str = "absolute",  # "absolute": (mu, raw_sigma) per step;
        # "delta": (delta_mu, raw_delta_var) increments in the likelihood's
        # native space, accumulated in-graph to the cumulative distribution
        rnn_monotone: bool = False,        # delta mode: constrain delta_mu >= 0
        rnn_absorbing_fail: bool = False,  # eval-time absorbing "failed" state
        rnn_delta_max: Optional[float] = None,  # increment bound: delta =
        # delta_max * sigmoid(raw); None = softplus(raw) (unbounded)
        rnn_delta_var_min: float = 0.0,  # delta mode: additive floor on the
        # increment variance (prevents overconfident near-zero sigma and the
        # resulting unbounded increment NLL on out-of-distribution tails)

        # context pooling / propagation (GRU heads)
        n_ctx_queries: int = 1,          # pool queries per GRU variable: richer,
        # multi-head-style context (gru_per_var: variable v owns queries
        # [v*k : (v+1)*k], so the total is n_dec_vars * n_ctx_queries)
        ctx_mlp_hidden: Optional[int] = None,  # if set, per-query context MLP
        # Linear(d->h)->GELU->Linear(h->d): nonlinear readout of the pooled
        # context (the raw attention output is a linear combination of values)
        rnn_ctx_feed: bool = False,      # propagate the context: concat its
        # projection to the GRU input at EVERY step (shared map, trained
        # jointly by all steps; the context is computed once per forward)
        rnn_ctx_feed_dim: int = 64,      # dimension of the per-step context feed
        rnn_fused: bool = True,          # gru_per_var, 1 layer: run all
        # variables in ONE batched step loop (stacked weights, same math and
        # parameters as the per-variable nn.GRU loop; False = reference loop)
        rnn_update: Optional[str] = None,  # measurement update at given
        # positions: None = level reset only (legacy), "innovation" = also
        # correct the hidden state from the standardized innovation with a
        # per-variable gated map (see innovation_update.py)
        rnn_update_vmeas: Optional[float] = None,  # variance after an update
        # (native space); None = learned (softplus + rnn_delta_var_min)
        rnn_update_clip: float = 10.0,   # clamp of the standardized innovation
        rnn_update_map: str = "mlp",     # candidate map: "mlp" ([h, x] -> h)
        # or "film" (x = (e, z_true, log v) modulates h affinely)
        rnn_update_d_map: Optional[int] = None,  # inner width (None = d_hidden)

        # multi-head output pooling (scalar head only)
        n_output_heads: int = 1,           # number of pool queries / dedicated MLP heads
        n_head_outputs: Optional[int] = None,  # per-head output width (default: out_dim)
        head_hidden_dim: Optional[int] = None, # per-head MLP hidden dim (None: single Linear)
        head_var_embed: bool = False,      # add a per-head embedding to the pool queries

        # spline_per_var head (see spline_head.py)
        spline_n_basis: int = 16,          # number of I-spline coefficients K
        spline_rank: int = 2,              # rank of the coefficient covariance factor
        spline_hidden: int = 128,          # per-variable MLP hidden width
        spline_knot_power: float = 1.0,    # >1: denser knots near the end
        spline_t_max: Optional[float] = None,  # horizon; None = L_dec // n_dec_vars
        spline_var_min: float = 1e-4,      # floor on the coefficient variance
        spline_noise_min: float = 1e-4,    # floor on the observation noise
        ):
        super().__init__()

        assert head_type in ["scalar", "gru", "gru_per_var", "spline_per_var"], \
            f"head_type must be 'scalar', 'gru', 'gru_per_var' or " \
            f"'spline_per_var', got {head_type}"
        if head_type == "scalar":
            assert output_dim is not None and output_dim > 0, \
                "output_dim must be a positive integer for the scalar head"
        if head_type == "spline_per_var":
            assert n_dec_vars is not None and n_dec_vars > 0 \
                and dec_var_idx is not None, \
                "spline_per_var requires n_dec_vars and dec_var_idx"
            assert out_dim == 2, "spline_per_var requires out_dim=2 (mu, raw_sigma)"
        if head_type == "gru_per_var":
            assert n_dec_vars is not None and n_dec_vars > 0, \
                "n_dec_vars must be a positive integer for the gru_per_var head"
            assert dec_var_idx is not None, \
                "dec_var_idx must be set for the gru_per_var head"
        assert n_output_heads >= 1, \
            f"n_output_heads must be >= 1, got {n_output_heads}"
        assert rnn_output_mode in ("absolute", "delta"), \
            f"rnn_output_mode must be 'absolute' or 'delta', got {rnn_output_mode}"
        if rnn_output_mode == "delta":
            assert head_type in ("gru", "gru_per_var"), \
                "rnn_output_mode='delta' requires a GRU head"
            assert out_dim == 2, \
                "rnn_output_mode='delta' requires out_dim=2 (delta_mu, raw_delta_var)"
        if rnn_monotone:
            assert rnn_output_mode == "delta", \
                "rnn_monotone constrains the increment and requires " \
                "rnn_output_mode='delta'"
        assert rnn_update in (None, "innovation"), \
            f"rnn_update must be None or 'innovation', got {rnn_update}"
        if rnn_update is not None:
            assert head_type == "gru_per_var" and rnn_output_mode == "delta", \
                "rnn_update='innovation' requires head_type='gru_per_var' " \
                "and rnn_output_mode='delta'"
            assert rnn_n_layers == 1, "rnn_update requires rnn_n_layers=1"
        if n_output_heads > 1:
            assert head_type == "scalar", \
                "n_output_heads > 1 is only supported for the scalar head"
            assert n_output_heads == output_dim, \
                f"n_output_heads ({n_output_heads}) must equal output_dim " \
                f"({output_dim}): one pool query + MLP head per scalar target"

        self.head_type = head_type
        self.output_dim = output_dim
        self.d_hidden = d_hidden
        self.rnn_n_layers = rnn_n_layers
        self.out_dim = out_dim
        self.dec_val_idx = dec_val_idx
        self.dec_pos_idx = dec_pos_idx
        self.n_dec_vars = n_dec_vars
        self.dec_var_idx = dec_var_idx
        self.n_output_heads = n_output_heads
        self.n_head_outputs = n_head_outputs if n_head_outputs is not None else out_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_var_embed = head_var_embed
        self.rnn_monotone = rnn_monotone
        self.rnn_absorbing_fail = rnn_absorbing_fail
        self.rnn_delta_max = rnn_delta_max
        self.rnn_delta_var_min = rnn_delta_var_min
        self.rnn_output_mode = rnn_output_mode
        self.n_ctx_queries = n_ctx_queries
        self.rnn_ctx_feed = rnn_ctx_feed
        self.rnn_fused = rnn_fused
        self.rnn_update = rnn_update

        # Delta mode extras: sigma_eps mirrors the loss eps (sigma = softplus(raw)
        # + eps) so the cumulative raw_sigma channel inverts the accumulated
        # std exactly; set by the forecaster (it owns the loss configuration).
        # last_step_deltas stores the per-step (delta_mu, raw_delta_sigma)
        # increments of the latest forward pass for the increment NLL.
        self.sigma_eps = 1e-3
        self.last_step_deltas = None

        # Failure threshold for the absorbing "failed" state (eval only): once
        # the predicted median of any variable crosses fail_threshold, the
        # device is dead and all later positions (both variables) emit the
        # degenerate zero distribution. fail_shift mirrors feedback_shift: set
        # means mu parameterizes log(y + fail_shift) (zero median = log(shift)),
        # None means mu is already in y-space (zero median = 0). Both are set
        # by the forecaster (it owns the loss configuration).
        self.fail_threshold = None
        self.fail_shift = None

        # Autoregressive feedback scale for the GRU heads: None = feed back the
        # raw mu channel (regression losses); a float shift means mu
        # parameterizes log(y + shift) (lognormal/student/laplace losses) and
        # the y-space point forecast exp(mu) - shift is fed back instead, so
        # the recurrent input matches the raw target scale. Set by the
        # forecaster (it owns the loss configuration).
        self.feedback_shift = None

        # Upper clamp for the autoregressive feedback value (y-space),
        # ALWAYS active (train + eval) as a runaway guard; set by the
        # forecaster from training.loss_censor_threshold. Independent of
        # fail_threshold (eval-only absorbing state), which the forecaster
        # disables during training - coupling the clamp to fail_threshold
        # left the guard off in training and caused the NaN cascade of run
        # 16140288. None = no clamp (losses without a censor threshold).
        self.feedback_clamp_max = None

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

        # attention pooling: learnable query token(s) attend over encoder outputs.
        # Scalar head: n_output_heads queries (one per scalar target). GRU heads:
        # n_ctx_queries per unrolled variable (gru_per_var: n_dec_vars * k).
        self.n_pool_queries = n_output_heads if head_type == "scalar" else (
            n_ctx_queries if head_type == "gru" else n_dec_vars * n_ctx_queries)
        self.pool_query = nn.Parameter(
            torch.randn(1, self.n_pool_queries, d_model_enc) * 0.02)
        if head_var_embed:
            # per-head embedding added to the pool queries: shared base query
            # plus a head-specific offset (faster specialization than free queries)
            self.pool_query_embed = nn.Embedding(self.n_pool_queries, d_model_enc)
            nn.init.normal_(self.pool_query_embed.weight, std=0.02)
        # optional nonlinear readout of the pooled context (per query vector)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d_model_enc, ctx_mlp_hidden, bias=True),
            nn.GELU(),
            nn.Linear(ctx_mlp_hidden, d_model_enc, bias=True),
            ) if ctx_mlp_hidden is not None else None
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
            if n_output_heads == 1:
                self.scalar_head = nn.Linear(d_model_enc, output_dim * out_dim, bias=True)
            else:
                # one dedicated MLP per pool query / scalar target
                self.scalar_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model_enc, head_hidden_dim, bias=True),
                        nn.ReLU() if activation == "relu" else nn.GELU(),
                        nn.Linear(head_hidden_dim, self.n_head_outputs, bias=True),
                    ) if head_hidden_dim is not None else
                    nn.Linear(d_model_enc, self.n_head_outputs, bias=True)
                    for _ in range(n_output_heads)])
        elif head_type == "spline_per_var":
            # functional head: per-variable MLP -> Gaussian over I-spline
            # coefficients, conditioned in closed form on given positions
            self.spline_head = SplineHead(
                n_vars=n_dec_vars, ctx_dim=n_ctx_queries * d_model_enc,
                n_basis=spline_n_basis, rank=spline_rank, hidden=spline_hidden,
                var_min=spline_var_min, noise_min=spline_noise_min,
                knot_power=spline_knot_power, t_max=spline_t_max)
        else:
            # target covariate embedding for the GRU inputs
            self.dec_embedding = ModularEmbedding(
                ds_embed=ds_embed_dec,
                comps=comps_embed_dec,
                device=device)
            # per-step context feed (P1): the flattened per-variable context
            # (n_ctx_queries * d_model_enc) is projected once per forward and
            # concatenated to the GRU input at every step
            ctx_in_dim = n_ctx_queries * d_model_enc
            ctx_extra = rnn_ctx_feed_dim if rnn_ctx_feed else 0
            if rnn_ctx_feed:
                self.ctx_feed_proj = nn.Linear(ctx_in_dim, rnn_ctx_feed_dim, bias=True)
            if head_type == "gru":
                # context -> initial hidden state of the GRU
                self.h0_projection = nn.Linear(ctx_in_dim, rnn_n_layers * d_hidden, bias=True)
                self.rnn = nn.GRU(
                    input_size=d_model_dec + 1 + ctx_extra,
                    hidden_size=d_hidden,
                    num_layers=rnn_n_layers,
                    batch_first=True,
                    dropout=rnn_dropout if rnn_n_layers > 1 else 0.0
                    )
                self.rnn_head = nn.Linear(d_hidden, out_dim, bias=True)
            else:
                # gru_per_var: context -> one initial hidden state per variable;
                # one independent GRU and one independent head per variable.
                # The h0 / ctx-feed projections are SHARED across variables
                # (their input is already variable-specific via the queries)
                self.h0_projection = nn.Linear(
                    ctx_in_dim, rnn_n_layers * d_hidden, bias=True)
                self.rnn = nn.ModuleList([
                    nn.GRU(
                        input_size=d_model_dec + 1 + ctx_extra,
                        hidden_size=d_hidden,
                        num_layers=rnn_n_layers,
                        batch_first=True,
                        dropout=rnn_dropout if rnn_n_layers > 1 else 0.0
                        ) for _ in range(n_dec_vars)])
                self.rnn_head = nn.ModuleList([
                    nn.Linear(d_hidden, out_dim, bias=True) for _ in range(n_dec_vars)])
                # innovation update cell (only built when enabled, so legacy
                # state dicts are unchanged)
                if rnn_update == "innovation":
                    self.update_cell = InnovationUpdate(
                        n_dec_vars, d_hidden, var_floor=rnn_delta_var_min,
                        v_meas=rnn_update_vmeas, clip=rnn_update_clip,
                        update_map=rnn_update_map, d_map=rnn_update_d_map)

            # monotone increment head (delta mode): init the increment channel
            # (channel 0) to softplus(-8) ~ 3e-4 so untrained rollouts stay
            # near zero instead of drifting ~0.7/step (softplus of a
            # random-init output)
            if rnn_monotone:
                heads = self.rnn_head if isinstance(self.rnn_head, nn.ModuleList) \
                    else [self.rnn_head]
                for head in heads:
                    head.bias.data[0] = -8.0

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
        """Attention-pool the encoder outputs into context vector(s).

        Returns (B, d_model_enc) with a single query, (B, n_output_heads,
        d_model_enc) with one query per scalar target.
        """
        B = enc_out.size(0)
        query = self.pool_query.expand(B, -1, -1)  # (B, H, d_model_enc)
        if self.head_var_embed:
            query = query + self.pool_query_embed.weight.unsqueeze(0)
        context, pool_att, _ = self.pool_attention(
            query=query,
            key=enc_out,
            value=enc_out,
            mask_miss_k=enc_mask,
            mask_miss_q=None,
            pos=None,
            causal_mask=False
            )
        if self.ctx_mlp is not None:
            context = self.ctx_mlp(context)        # nonlinear readout (last dim)
        if self.n_pool_queries == 1:
            context = context.squeeze(1)  # (B, d_model_enc)
        return context, pool_att

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
            if self.n_output_heads == 1:
                forecast_out = self.scalar_head(context).view(
                    context.size(0), self.output_dim, self.out_dim)
            else:
                # context: (B, H, d_model_enc); one dedicated MLP per head
                forecast_out = torch.stack(
                    [head(context[:, h]) for h, head in enumerate(self.scalar_head)],
                    dim=1)  # (B, n_output_heads, n_head_outputs)
        elif self.head_type == "gru":
            forecast_out = self._gru_rollout(context, target_tensor, trg_pos_mask)
        elif self.head_type == "spline_per_var":
            forecast_out = self._spline_forward(context, target_tensor, trg_pos_mask)
        else:
            forecast_out = self._gru_rollout_per_var(context, target_tensor, trg_pos_mask)

        recon_out = self.reconstructor(enc_out)

        return forecast_out, recon_out, (enc_self_att, [], [pool_att]), enc_mask, \
            (enc_self_ent, [], [])

    def _spline_forward(self, context, target_tensor, trg_pos_mask):
        """Spline head: predictive (mu, raw_sigma) at every target token.

        Given AND observed positions (trg_pos_mask == False, value not NaN)
        condition the coefficients exactly (Kalman update, see SplineHead);
        all positions, given ones included, get the posterior predictive.
        """
        B, L_dec, _ = target_tensor.shape
        head = self.spline_head
        if float(head.t_max) <= 0:
            # shared physical axis: horizon = per-variable padded length
            head.t_max.fill_(float(L_dec // self.n_dec_vars))
        if context.dim() == 2:
            context = context.unsqueeze(1)                    # (B, 1, d)
        context = context.reshape(B, self.n_dec_vars, -1)     # (B, V, k*d)

        pos = target_tensor[:, :, self.dec_pos_idx]
        var_ids = torch.nan_to_num(target_tensor[:, :, self.dec_var_idx], nan=0.0).long()
        real_val = target_tensor[:, :, self.dec_val_idx]
        obs = ~torch.isnan(real_val)
        real_val = torch.nan_to_num(real_val)
        z_real = torch.log((real_val + self.feedback_shift).clamp_min(1e-8)) \
            if self.feedback_shift is not None else real_val
        if trg_pos_mask is not None:
            given = torch.logical_not(trg_pos_mask).squeeze(-1) & obs
        else:
            given = torch.zeros_like(obs)

        mu, var = head(context, pos, var_ids, z_real, given, self._delta_z0())
        outputs = torch.stack([mu, self._raw_sigma_from_var(var)], dim=-1)
        return self._finish_rollout(outputs, None, target_tensor)

    def _feedback_value(self, mu):
        """Value fed back into the GRU at the next autoregressive step.

        For log-space losses mu parameterizes log(y + shift), so the y-space
        point forecast exp(mu) - shift is fed back to match the raw target
        scale; identity otherwise (regression losses predict y directly).

        Runaway guard (train AND eval): when feedback_clamp_max is set (the
        forecaster derives it from the censor threshold, INDEPENDENT of the
        eval-only fail_threshold), mu is sanitized and clamped in LOG space
        BEFORE the exp. exp(mu) overflows fp32 at mu >~ 88.7, and clamping
        the RESULT is not enough: exp's backward multiplies the incoming
        gradient by exp(mu), so a gradient zeroed at a clamped/nan_to_num'd
        inf node still becomes 0 x inf = NaN into mu (that route
        NaN-poisoned every weight in run 16140288). Values beyond the
        failure threshold carry no information anyway (the device is dead).
        """
        if self.feedback_shift is not None:
            if self.feedback_clamp_max is not None:
                # sanitize mu first (a NaN/inf mu would survive the clamp),
                # then clamp in log space: exp stays finite and the clamp
                # backward yields a clean zero gradient - no 0 x inf route
                mu_max = math.log(self.feedback_clamp_max + self.feedback_shift)
                mu = torch.nan_to_num(mu, nan=0.0, posinf=mu_max, neginf=-20.0)
                val = torch.exp(mu.clamp(max=mu_max)) - self.feedback_shift
            else:
                val = torch.exp(mu) - self.feedback_shift
        else:
            val = mu
        # belt-and-braces clamp on the y-space value (no-op for the log-space
        # path above, where mu was already clamped; active for regression
        # losses and as a last resort)
        if self.feedback_clamp_max is not None:
            val = torch.nan_to_num(val, nan=0.0,
                                   posinf=self.feedback_clamp_max, neginf=0.0)
            val = val.clamp(max=self.feedback_clamp_max)
        return val

    def _delta_z0(self):
        """Native-space value corresponding to y = 0 (delta-mode state init).

        log-space losses (feedback_shift set): z = log(y + shift), so y = 0
        maps to log(shift). y-space losses: 0.0.
        """
        if self.feedback_shift is not None:
            return math.log(self.feedback_shift)
        return 0.0

    def _delta_step(self, raw, z_base, v_base):
        """Increment step (rnn_output_mode="delta") in the native loss space.

        The head outputs (delta_mu_raw, delta_var_raw) per step. Conditioned on
        the history (hidden state), the increment is Gaussian:
        delta ~ N(delta_mu, delta_var), with delta_var = softplus(raw)
        (variance increments are non-negative, so the accumulated variance is
        structurally non-decreasing). rnn_monotone optionally constrains
        delta_mu >= 0 (softplus, or delta_max * sigmoid when bounded), giving
        non-decreasing cumulative paths (in native space, hence in y-space).

        Args:
            raw: head output (B, 2)
            z_base: cumulative native-space mean at the previous step (B,)
            v_base: accumulated native-space variance at the previous step (B,)

        Returns:
            z: cumulative mean z_base + delta_mu (B,)
            v: accumulated variance v_base + delta_var (B,)
            d_mu: the increment mean (B,)
            d_var: the increment variance (B,)
        """
        if self.rnn_monotone:
            if self.rnn_delta_max is not None:
                d_mu = self.rnn_delta_max * torch.sigmoid(raw[:, 0])
            else:
                d_mu = F.softplus(raw[:, 0])
        else:
            d_mu = raw[:, 0]
        d_var = F.softplus(raw[:, 1]) + self.rnn_delta_var_min
        return z_base + d_mu, v_base + d_var, d_mu, d_var

    def _raw_sigma_from_var(self, var):
        """Inverse of the loss convention sigma = softplus(raw) + sigma_eps."""
        sigma = var.clamp_min(0).sqrt()
        return torch.log(torch.expm1((sigma - self.sigma_eps).clamp_min(1e-6)))

    @staticmethod
    def increments_to_cumulative(d_mu, d_var, z0=0.0):
        """Accumulate per-step increments into the cumulative distribution.

        Args:
            d_mu: increment means (B, L), native likelihood space
            d_var: increment variances (B, L), native likelihood space
            z0: native-space value of y = 0 (log(shift) for log-space
                likelihoods, 0.0 otherwise); the cumulative mean starts here

        Returns:
            mu_cum: cumulative mean (B, L) = z0 + cumsum(d_mu)
            sigma_cum: cumulative std (B, L) = sqrt(cumsum(d_var))

        Both live in the native space of the likelihood (log-space for the
        lognormal losses, y-space for the Gaussian loss); a sum of Gaussian
        increments is exactly Gaussian, so no approximation is involved. For
        y-space summaries under a log-space likelihood use the lognormal
        formulas, e.g. median = exp(mu_cum) - shift.
        """
        return z0 + d_mu.cumsum(dim=1), d_var.cumsum(dim=1).clamp_min(0).sqrt()

    def _apply_fail_stop(self, outputs, target_tensor):
        """Absorbing "failed" state at inference (rnn_absorbing_fail).

        Per variable and fully independent: once the predicted median of a
        variable crosses fail_threshold, all later positions of THAT variable
        emit the degenerate zero distribution (zero median, minimal sigma).
        The sibling variable is unaffected and stops only at its own crossing
        (the data shows the two variables fail independently).

        No learnable failed hidden state is used on purpose: post-stop
        positions are masked in the (fail-mode) loss and their outputs are
        overridden here, so such a state would receive no gradient.
        """
        mu = outputs[:, :, 0]
        median = torch.exp(mu) - self.fail_shift if self.fail_shift is not None else mu
        pos = target_tensor[:, :, self.dec_pos_idx]
        var_ids = torch.nan_to_num(target_tensor[:, :, self.dec_var_idx], nan=0.0).long()
        valid = ~torch.isnan(pos) & (var_ids > 0)

        crossed = valid & (median >= self.fail_threshold)
        # independent stop per variable: dead strictly after its own crossing
        dead = torch.zeros_like(valid)
        for v in range(1, self.n_dec_vars + 1):
            is_v = var_ids == v
            pos_v = torch.where(crossed & is_v, pos,
                                torch.full_like(pos, float("inf")))
            stop_v = pos_v.min(dim=1).values                 # (B,)
            dead |= valid & is_v & (pos > stop_v.unsqueeze(1))
        if not dead.any():
            return outputs
        mu_zero = math.log(self.fail_shift) if self.fail_shift is not None else 0.0
        out = outputs.clone()
        out[:, :, 0] = torch.where(dead, torch.full_like(mu, mu_zero), mu)
        out[:, :, 1] = torch.where(dead, torch.full_like(mu, -20.0),
                                   outputs[:, :, 1])     # softplus(-20) ~ 0
        return out

    def _gru_rollout(self, context, target_tensor, trg_pos_mask):
        """
        Initialize the GRU hidden state from the pooled context and unroll over
        the target positions. Where the target is given (trg_pos_mask == False)
        the real value is fed; elsewhere the predicted mean is fed back
        (autoregressive rollout).
        """
        B, L_dec, _ = target_tensor.shape

        # initial hidden state from the attention-pooled context
        # linear projection only (dimension match k*d_model_enc -> n_layers *
        # d_hidden); no tanh: the GRU gates provide the nonlinearities
        if self.n_ctx_queries > 1:
            context = context.flatten(1)                     # (B, k * d_model_enc)
        h0 = self.h0_projection(context)                      # (B, n_layers * d_hidden)
        h = h0.view(B, self.rnn_n_layers, self.d_hidden) \
              .permute(1, 0, 2).contiguous()                  # (n_layers, B, d_hidden)
        # per-step context feed (shared map, constant across steps)
        ctx_feed = self.ctx_feed_proj(context) if self.rnn_ctx_feed else None  # (B, feed_dim)

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

        if self.rnn_output_mode == "delta":
            # native-space real values (log-space for log-space losses); the
            # clamp is a numerical last resort: the data can dip slightly
            # below -shift
            obs = ~torch.isnan(target_tensor[:, :, self.dec_val_idx])
            z_real = torch.log((real_val + self.feedback_shift).clamp_min(1e-8)) \
                if self.feedback_shift is not None else real_val
            # cumulative state: (z_state, v_state); after a given (observed)
            # position the state is the true value with zero variance.
            # Init at the "y = 0" point of the native space: log(shift) for
            # log-space losses (z = log(y + shift) == log(shift) at y = 0),
            # 0.0 for y-space losses. (Was 0.0 unconditionally: in log-space
            # z=0 corresponds to y ~ 1, the worst possible init for curves
            # starting at 0, and it made the first increment target of every
            # curve ~ log(shift) ~ -6.9, unmatchable for a monotone head.)
            z_state = torch.full((B,), self._delta_z0(),
                                 device=target_tensor.device,
                                 dtype=target_tensor.dtype)
            v_state = torch.zeros_like(z_state)
            outputs, deltas = [], []
            for t in range(L_dec):
                # history input: y-space value of the current state (0 at the
                # first step: sequences start at y ~ 0, no leak)
                val_in = torch.zeros_like(z_state) if t == 0 \
                    else self._feedback_value(z_state)
                parts = [dec_emb[:, t, :], val_in.unsqueeze(-1)]
                if ctx_feed is not None:
                    parts.append(ctx_feed)
                step_in = torch.cat(parts, dim=-1).unsqueeze(1)
                out, h = self.rnn(step_in, h)                    # (B, 1, d_hidden)
                raw = self.rnn_head(out.squeeze(1))              # (B, out_dim)
                z, v, d_mu, d_var = self._delta_step(raw, z_state, v_state)
                outputs.append(torch.stack([z, self._raw_sigma_from_var(v)], -1))
                deltas.append(torch.stack([d_mu, self._raw_sigma_from_var(d_var)], -1))
                # measurement update (true value, variance zero)
                given_obs = given[:, t] & obs[:, t]
                z_state = torch.where(given_obs, z_real[:, t], z)
                v_state = torch.where(given_obs, torch.zeros_like(v), v)
            self.last_step_deltas = torch.stack(deltas, dim=1)   # (B, L_dec, 2)
            return torch.stack(outputs, dim=1)                   # (B, L_dec, out_dim)

        outputs = []
        prev_val = real_val[:, 0]  # first step input value (overwritten below if not given)
        for t in range(L_dec):
            step_val = torch.where(given[:, t], real_val[:, t], prev_val)
            parts = [dec_emb[:, t, :], step_val.unsqueeze(-1)]
            if ctx_feed is not None:
                parts.append(ctx_feed)
            step_in = torch.cat(parts, dim=-1).unsqueeze(1)  # (B, 1, d_model_dec+1+ctx)
                # note: d_model_dec must match the GRU input_size; the value channel is
                # appended after the modular embedding, so the GRU input size is
                # d_model_dec + 1. See __init__ (input_size=d_model_dec + 1).
            out, h = self.rnn(step_in, h)                        # (B, 1, d_hidden)
            pred = self.rnn_head(out.squeeze(1))                 # (B, out_dim)
            outputs.append(pred)
            prev_val = self._feedback_value(pred[:, 0])          # y-space forecast fed back

        return torch.stack(outputs, dim=1)                       # (B, L_dec, out_dim)

    def _gru_rollout_per_var(self, context, target_tensor, trg_pos_mask):
        """
        Like _gru_rollout, but with one independent GRU per target variable.
        The pooled context is projected to n_dec_vars initial hidden states;
        each variable's subsequence (selected via the variable channel) is
        unrolled by its own GRU from its own h0, so no hidden state flows
        across variables. Predictions are scattered back to the original
        positions, preserving the (B, L_dec, out_dim) output layout.

        Ragged layouts are supported: variable blocks may have different
        lengths across samples (packed sequences), and padded tokens
        (variable id 0 or NaN) are excluded from the rollouts.
        """
        B, L_dec, _ = target_tensor.shape

        # per-variable initial hidden states from the attention-pooled context:
        # variable v owns queries [v*k : (v+1)*k]; the projection is shared
        # across variables (no tanh: the GRU gates provide the nonlinearities)
        if self.n_ctx_queries > 1:
            context = context.view(B, self.n_dec_vars, -1)     # (B, n_vars, k*d)
        else:
            context = context.unsqueeze(1)                     # (B, n_vars, d)
        h0 = self.h0_projection(context)                       # (B, n_vars, n_layers*d_hidden)
        h0 = h0.view(B, self.n_dec_vars, self.rnn_n_layers, self.d_hidden)
        # per-step context feed (shared map, constant across steps)
        ctx_feed = self.ctx_feed_proj(context) if self.rnn_ctx_feed else None  # (B, n_vars, feed_dim)

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
        # delta mode: native-space real values and observed mask; the clamp is
        # a numerical last resort: the data can dip slightly below -shift
        if self.rnn_output_mode == "delta":
            obs = ~torch.isnan(target_tensor[:, :, self.dec_val_idx])
            z_real = torch.log((real_val + self.feedback_shift).clamp_min(1e-8)) \
                if self.feedback_shift is not None else real_val
        # NaN variable ids (padded tail) become 0 = padding: excluded from all streams
        var_ids = torch.nan_to_num(target_tensor[:, :, self.dec_var_idx], nan=0.0).long()

        outputs = target_tensor.new_zeros(B, L_dec, self.out_dim)
        delta_out = target_tensor.new_zeros(B, L_dec, 2) \
            if self.rnn_output_mode == "delta" else None

        if self.rnn_fused and self.rnn_n_layers == 1:
            return self._fused_rollout_per_var(
                h0, ctx_feed, dec_emb, given, real_val, var_ids, outputs,
                delta_out, target_tensor)

        for v in range(1, self.n_dec_vars + 1):
            mask_v = var_ids == v                                # (B, L_dec)
            L_v = int(mask_v.sum(dim=1).max())
            if L_v == 0:
                continue

            # gather each sample's variable-v positions (original order) to the
            # front of the gathered sequence; padded entries (beyond a sample's
            # valid length) sit at the end and their predictions are discarded
            idx_v = torch.argsort(
                torch.logical_not(mask_v).long(), dim=1, stable=True)[:, :L_v]  # (B, L_v)
            valid_v = mask_v.gather(1, idx_v)                    # (B, L_v)

            h = h0[:, v - 1].permute(1, 0, 2).contiguous()       # (n_layers, B, d_hidden)
            emb_v = dec_emb.gather(1, idx_v.unsqueeze(-1).expand(-1, -1, dec_emb.size(-1)))
            given_v = given.gather(1, idx_v)                     # (B, L_v)
            real_v = real_val.gather(1, idx_v)                   # (B, L_v)

            if self.rnn_output_mode == "delta":
                obs_v = obs.gather(1, idx_v)
                z_real_v = z_real.gather(1, idx_v)
                # cumulative state; after a given (observed) position the
                # state is the true value with zero variance. Init at the
                # "y = 0" point of the native space (see _delta_z0).
                z_state = torch.full_like(real_v[:, 0], self._delta_z0())
                v_state = torch.zeros_like(z_state)
                preds_v, deltas_v = [], []
                for t in range(L_v):
                    # history input: y-space value of the current state (0 at
                    # the first step: sequences start at y ~ 0, no leak)
                    val_in = torch.zeros_like(z_state) if t == 0 \
                        else self._feedback_value(z_state)
                    parts = [emb_v[:, t, :], val_in.unsqueeze(-1)]
                    if ctx_feed is not None:
                        parts.append(ctx_feed[:, v - 1])
                    step_in = torch.cat(parts, dim=-1).unsqueeze(1)
                    out, h = self.rnn[v - 1](step_in, h)         # (B, 1, d_hidden)
                    raw = self.rnn_head[v - 1](out.squeeze(1))   # (B, out_dim)
                    z, v_acc, d_mu, d_var = self._delta_step(raw, z_state, v_state)
                    preds_v.append(torch.stack([z, self._raw_sigma_from_var(v_acc)], -1))
                    deltas_v.append(torch.stack([d_mu, self._raw_sigma_from_var(d_var)], -1))
                    # measurement update: level reset to the true value (legacy)
                    # and, with rnn_update, innovation correction of h
                    given_obs = given_v[:, t] & obs_v[:, t]
                    if self._use_update():
                        h = self.update_cell.step(
                            v - 1, h[0], z, v_acc, z_real_v[:, t], given_obs
                        ).unsqueeze(0)
                    z_state = torch.where(given_obs, z_real_v[:, t], z)
                    v_state = torch.where(given_obs, self._v_after_update(v_acc), v_acc)
                preds_v = torch.stack(preds_v, dim=1)            # (B, L_v, out_dim)
                deltas_v = torch.stack(deltas_v, dim=1)          # (B, L_v, 2)
                outputs[mask_v] = preds_v[valid_v]
                delta_out[mask_v] = deltas_v[valid_v]
                continue

            preds_v = []
            prev_val = torch.zeros_like(real_v[:, 0])            # no leak of the first real value
            for t in range(L_v):
                step_val = torch.where(given_v[:, t], real_v[:, t], prev_val)
                parts = [emb_v[:, t, :], step_val.unsqueeze(-1)]
                if ctx_feed is not None:
                    parts.append(ctx_feed[:, v - 1])
                step_in = torch.cat(parts, dim=-1).unsqueeze(1)  # (B, 1, d_model_dec+1+ctx)
                out, h = self.rnn[v - 1](step_in, h)             # (B, 1, d_hidden)
                pred = self.rnn_head[v - 1](out.squeeze(1))      # (B, out_dim)
                preds_v.append(pred)
                prev_val = self._feedback_value(pred[:, 0])      # y-space forecast fed back
            preds_v = torch.stack(preds_v, dim=1)                # (B, L_v, out_dim)

            # scatter back to the original positions (valid entries only; both
            # boolean masks select in the same batch-major, position order)
            outputs[mask_v] = preds_v[valid_v]

        return self._finish_rollout(outputs, delta_out, target_tensor)

    def _finish_rollout(self, outputs, delta_out, target_tensor):
        """Shared rollout epilogue: store increments, apply the fail-stop."""
        if delta_out is not None:
            self.last_step_deltas = delta_out                    # (B, L_dec, 2)

        # absorbing "failed" state (inference only): override post-stop outputs
        if self.rnn_absorbing_fail and not self.training \
                and self.fail_threshold is not None:
            outputs = self._apply_fail_stop(outputs, target_tensor)

        return outputs                                           # (B, L_dec, out_dim)

    def _fused_rollout_per_var(self, h0, ctx_feed, dec_emb, given, real_val,
                               var_ids, outputs, delta_out, target_tensor):
        """All variables in ONE step loop (gru_per_var, rnn_n_layers == 1).

        Mathematically identical to the per-variable nn.GRU loop and using
        the SAME parameters (self.rnn[v], self.rnn_head[v]), stacked on the
        fly so gradients flow to the originals; checkpoints are unaffected.
        Speed-ups: (1) the n_vars rollouts share one loop of max(L_v) steps
        instead of sum(L_v); (2) the input projection of the step-invariant
        inputs (decoder embedding, context feed) is precomputed for all steps
        in one batched matmul; only the fed-back value column is per-step.
        GRU equations as in torch.nn.GRU (gate order r, z, n):
            r = sig(Wx_r + b_ir + Wh_r h + b_hr)
            z = sig(Wx_z + b_iz + Wh_z h + b_hz)
            n = tanh(Wx_n + b_in + r * (Wh_n h + b_hn))
            h = (1 - z) * n + z * h
        Rows are ordered variable-major: row v*B + b = (variable v+1, sample b).
        """
        V, H = self.n_dec_vars, self.d_hidden
        B, L_dec, _ = target_tensor.shape
        delta = self.rnn_output_mode == "delta"

        # per-variable gather (same ordering as the reference loop)
        masks = [var_ids == v for v in range(1, V + 1)]
        L = max(int(m.sum(dim=1).max()) for m in masks)
        if L == 0:
            return self._finish_rollout(outputs, delta_out, target_tensor)
        idx = torch.cat([torch.argsort(torch.logical_not(m).long(), dim=1,
                                       stable=True)[:, :L] for m in masks], 0)
        valid = torch.cat([m for m in masks], 0).gather(1, idx)   # (V*B, L)

        def g(x):
            """Gather a (B, L_dec[, F]) tensor for all variables -> (V*B, L[, F])."""
            xs = x.repeat(V, *([1] * (x.dim() - 1)))
            if x.dim() == 3:
                return xs.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            return xs.gather(1, idx)

        h, step, x_pre, w_val_r = self._fused_setup(h0, ctx_feed, dec_emb, g, B, L)
        given_g, real_g = g(given), g(real_val)
        preds, deltas = self._fused_loop(h, step, x_pre, w_val_r, given_g,
                                         real_g, g, real_val, target_tensor, L)

        # scatter back per variable (same boolean-mask ordering as reference)
        for v in range(V):
            sl = slice(v * B, (v + 1) * B)
            outputs[masks[v]] = preds[sl][valid[sl]]
            if delta:
                delta_out[masks[v]] = deltas[sl][valid[sl]]

        return self._finish_rollout(outputs, delta_out, target_tensor)

    def _fused_setup(self, h0, ctx_feed, dec_emb, g, B, L):
        """Stack the per-variable GRU/head parameters and precompute the
        step-invariant input projection. Returns (h, step_fn, x_pre, w_val_r)."""
        V, H = self.n_dec_vars, self.d_hidden
        W_ih, W_hh, b_ih, b_hh = stack_gru_params(self.rnn)       # (V, 3H, D_in), ...
        W_hhT = W_hh.transpose(1, 2)
        b_hh = b_hh.repeat_interleave(B, 0)
        W_oT = torch.stack([hd.weight for hd in self.rnn_head]).transpose(1, 2)
        b_o = torch.stack([hd.bias for hd in self.rnn_head]).repeat_interleave(B, 0)

        # input layout [dec_emb | value | ctx_feed] (as in the reference loop)
        D = dec_emb.size(-1)
        emb = g(dec_emb).view(V, B, L, D)
        x_pre = torch.einsum("vbld,vgd->vblg", emb, W_ih[:, :, :D]) \
            + b_ih[:, None, None, :]                              # (V, B, L, 3H)
        if ctx_feed is not None:
            c = ctx_feed.permute(1, 0, 2)                         # (V, B, feed)
            x_pre = x_pre + torch.einsum(
                "vbf,vgf->vbg", c, W_ih[:, :, D + 1:])[:, :, None, :]
        x_pre = x_pre.reshape(V * B, L, 3 * H)
        w_val_r = W_ih[:, :, D].repeat_interleave(B, 0)           # (V*B, 3H)

        def step(xg, h):
            """One GRU cell step (torch.nn.GRU equations) + output head."""
            h = gru_cell(xg, hidden_gates(h, W_hhT, b_hh, V), h)
            raw = torch.bmm(h.view(V, B, H), W_oT).view(V * B, -1) + b_o
            return h, raw

        h = h0[:, :, 0].permute(1, 0, 2).reshape(V * B, H)        # (V*B, H)
        return h, step, x_pre, w_val_r

    def _fused_loop(self, h, step, x_pre, w_val_r, given_g, real_g, g,
                    real_val, target_tensor, L):
        """Step loop of the fused rollout (same per-step logic as the
        reference loop). Returns (preds (V*B, L, out), deltas or None)."""
        if self.rnn_output_mode == "delta":
            obs = ~torch.isnan(target_tensor[:, :, self.dec_val_idx])
            z_real = torch.log((real_val + self.feedback_shift).clamp_min(1e-8)) \
                if self.feedback_shift is not None else real_val
            obs_g, z_real_g = g(obs), g(z_real)
            z_state = real_g.new_full((real_g.size(0),), self._delta_z0())
            v_state = torch.zeros_like(z_state)
            preds, deltas = [], []
            for t in range(L):
                val_in = torch.zeros_like(z_state) if t == 0 \
                    else self._feedback_value(z_state)
                h, raw = step(x_pre[:, t] + val_in.unsqueeze(-1) * w_val_r, h)
                z, v_acc, d_mu, d_var = self._delta_step(raw, z_state, v_state)
                preds.append(torch.stack([z, self._raw_sigma_from_var(v_acc)], -1))
                deltas.append(torch.stack([d_mu, self._raw_sigma_from_var(d_var)], -1))
                # measurement update AFTER the prior was emitted (no leak)
                given_obs = given_g[:, t] & obs_g[:, t]
                if self._use_update():
                    h = self.update_cell.step_fused(h, z, v_acc, z_real_g[:, t],
                                                    given_obs)
                z_state = torch.where(given_obs, z_real_g[:, t], z)
                v_state = torch.where(given_obs, self._v_after_update(v_acc), v_acc)
            return torch.stack(preds, 1), torch.stack(deltas, 1)

    def _use_update(self):
        """True when the innovation update of the hidden state is enabled."""
        return getattr(self, "rnn_update", None) == "innovation"

    def _v_after_update(self, v_acc):
        """Native-space variance after a measurement: 0 (legacy) or the
        update cell's measurement variance (rnn_update)."""
        if self._use_update():
            return self.update_cell.v_meas().to(v_acc).expand_as(v_acc)
        return torch.zeros_like(v_acc)

        preds = []
        prev_val = torch.zeros_like(real_g[:, 0])  # no leak of the first real value
        for t in range(L):
            step_val = torch.where(given_g[:, t], real_g[:, t], prev_val)
            h, pred = step(x_pre[:, t] + step_val.unsqueeze(-1) * w_val_r, h)
            preds.append(pred)
            prev_val = self._feedback_value(pred[:, 0])
        return torch.stack(preds, 1), None
