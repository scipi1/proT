"""Innovation-driven measurement update of the GRU hidden state.

Kalman-style correction step for the delta-mode gru_per_var head
(ProTEncoder, rnn_update="innovation"). At every given (observed) target
position, after the prior (z_t, v_t) has been emitted for the loss, the
hidden state is corrected from the standardized innovation

    e_t = (z_true_t - z_t) / sqrt(v_t)

The update is a single (non-recurrent) step, so the candidate state is a
plain per-variable nonlinear map of x = [e_t, z_true_t, log v_t] and h_minus:

    "mlp" : h_cand = tanh(W2 gelu(W1 [h_minus, x] + b1) + b2)
    "film": gamma, beta = A2 gelu(A1 x + a1) + a2   (gamma init 1, beta 0)
            h_cand = tanh(gamma * (W_h h_minus + b_h) + beta)

    gate = sigmoid(W_g [h_minus, x] + b_g)          (b_g init negative)
    h_t  = h_minus + gate * (h_cand - h_minus)

The gate is the only (learned, element-wise) Kalman gain. tanh keeps h_cand
in the GRU range (-1, 1), and the convex blend keeps h_t there too. The gated
residual starts close to the identity. At non-given positions h_t = h_minus.
In "film" the innovation features modulate the hidden state multiplicatively
instead of being concatenated to it (3 scalars vs H entries).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 3  # update input features: (e, z_true, log v_prior)
UPDATE_MAPS = ("mlp", "film")


def _stacked_linear(V, d_in, d_out):
    """Per-variable Linear weights (V, d_in, d_out) and bias (V, d_out) with
    the default nn.Linear initialization."""
    W = torch.empty(V, d_in, d_out)
    bound = 1.0 / math.sqrt(d_in)
    nn.init.uniform_(W, -bound, bound)
    b = torch.empty(V, d_out).uniform_(-bound, bound)
    return nn.Parameter(W), nn.Parameter(b)


def _apply(x, W, b):
    """Per-variable affine map: x (V, B, d_in) -> (V, B, d_out)."""
    return torch.baddbmm(b.unsqueeze(1), x, W)


class InnovationUpdate(nn.Module):
    """Per-variable innovation update maps acting on the shared hidden state.

    Args:
        n_vars: number of target variables (one map per variable)
        d_hidden: GRU hidden size (same as the predict cell)
        update_map: candidate map, "mlp" or "film"
        d_map: inner width of the map (None = d_hidden)
        var_floor: floor of the learned measurement variance (log-space);
            typically rnn_delta_var_min
        v_meas: fixed measurement variance; None = learned
            (softplus(raw) + var_floor)
        clip: clamp of the standardized innovation for stability
        gate_bias: initial bias of the residual gate (sigmoid(-3) ~ 0.05)
    """

    def __init__(self, n_vars: int, d_hidden: int, var_floor: float = 0.0,
                 v_meas: Optional[float] = None, clip: float = 10.0,
                 gate_bias: float = -3.0, update_map: str = "mlp",
                 d_map: Optional[int] = None):
        super().__init__()
        assert update_map in UPDATE_MAPS, \
            f"update_map must be one of {UPDATE_MAPS}, got {update_map}"
        self.n_vars, self.d_hidden = n_vars, d_hidden
        self.update_map = update_map
        self.clip = clip
        self.var_floor = float(var_floor)
        V, H = n_vars, d_hidden
        D = d_map or d_hidden
        if update_map == "mlp":
            self.W1, self.b1 = _stacked_linear(V, H + N_FEAT, D)
            self.W2, self.b2 = _stacked_linear(V, D, H)
        else:
            self.Wh, self.bh = _stacked_linear(V, H, H)
            self.A1, self.a1 = _stacked_linear(V, N_FEAT, D)
            self.A2, self.a2 = _stacked_linear(V, D, 2 * H)
            # identity modulation at init: gamma = 1, beta = 0
            nn.init.zeros_(self.A2)
            with torch.no_grad():
                self.a2.zero_()
                self.a2[:, :H] = 1.0
        self.gates = nn.ModuleList([nn.Linear(d_hidden + N_FEAT, d_hidden)
                                    for _ in range(n_vars)])
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, gate_bias)
        if v_meas is None:
            # learned; init so that v_meas ~ var_floor + 1e-3
            self.raw_v_meas = nn.Parameter(torch.tensor(math.log(math.expm1(1e-3))))
            self.register_buffer("fixed_v_meas", None)
        else:
            assert v_meas >= 0, f"v_meas must be >= 0, got {v_meas}"
            self.raw_v_meas = None
            self.register_buffer("fixed_v_meas", torch.tensor(float(v_meas)))

    def v_meas(self):
        """Measurement variance (log-space) assigned after an update."""
        if self.raw_v_meas is None:
            return self.fixed_v_meas
        return F.softplus(self.raw_v_meas) + self.var_floor

    def features(self, z_prior, v_prior, z_true):
        """Update input (N, 3): clipped standardized innovation, the true
        native-space value and the log prior variance."""
        v = v_prior.clamp_min(1e-8)
        e = ((z_true - z_prior) / v.sqrt()).clamp(-self.clip, self.clip)
        return torch.stack([e, z_true, v.log()], dim=-1)

    def step(self, v_idx, h_minus, z_prior, v_prior, z_true, active):
        """Reference update for one variable.

        Args:
            v_idx: 0-based variable index
            h_minus: predicted hidden state (B, H)
            z_prior, v_prior: prior mean / variance emitted at this step (B,)
            z_true: true native-space value (B,), finite where active
            active: (B,) bool, given & observed

        Returns:
            corrected hidden state (B, H)
        """
        x = self.features(z_prior, v_prior, z_true)
        sl = slice(v_idx, v_idx + 1)
        h_cand = self.candidate(h_minus.unsqueeze(0), x.unsqueeze(0), sl)[0]
        gate = torch.sigmoid(self.gates[v_idx](torch.cat([h_minus, x], -1)))
        h_upd = h_minus + gate * (h_cand - h_minus)
        return torch.where(active.unsqueeze(-1), h_upd, h_minus)

    def candidate(self, h, x, sl=slice(None)):
        """Candidate state for variables sl: h (V', B, H), x (V', B, 3)."""
        if self.update_map == "mlp":
            a = F.gelu(_apply(torch.cat([h, x], -1), self.W1[sl], self.b1[sl]))
            return torch.tanh(_apply(a, self.W2[sl], self.b2[sl]))
        gb = _apply(F.gelu(_apply(x, self.A1[sl], self.a1[sl])),
                    self.A2[sl], self.a2[sl])
        gamma, beta = gb.chunk(2, dim=-1)
        return torch.tanh(gamma * _apply(h, self.Wh[sl], self.bh[sl]) + beta)

    def step_fused(self, h_minus, z_prior, v_prior, z_true, active):
        """Same update for all variables at once (variable-major rows,
        row v*B + b)."""
        V, H = self.n_vars, self.d_hidden
        B = h_minus.size(0) // V
        x = self.features(z_prior, v_prior, z_true)                 # (V*B, 3)
        h_cell = self.candidate(h_minus.view(V, B, H),
                                x.view(V, B, -1)).reshape(V * B, H)
        W_g = torch.stack([g.weight for g in self.gates])            # (V, H, H+3)
        b_g = torch.stack([g.bias for g in self.gates]).repeat_interleave(B, 0)
        gin = torch.cat([h_minus, x], -1).view(V, B, -1)
        gate = torch.sigmoid(torch.bmm(gin, W_g.transpose(1, 2)).view(V * B, H) + b_g)
        h_upd = h_minus + gate * (h_cell - h_minus)
        return torch.where(active.unsqueeze(-1), h_upd, h_minus)
