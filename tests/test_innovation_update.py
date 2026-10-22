"""Unit tests for the innovation measurement update (rnn_update="innovation")
of the delta-mode gru_per_var head, and for the shared GRU-cell helper.

Layout as in test_failstop.py: target tensor (B, L, F) with channels
position=1, variable=2, value=3; two variable blocks. CPU, seconds runtime.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from proT.core.model_encoder import ProTEncoder
from proT.core.innovation_update import InnovationUpdate
from proT.core.modules.gru_ops import gru_cell, stack_gru_params, hidden_gates

POS, VAR, VAL = 1, 2, 3
E = 12                 # positions per variable
SHIFT = 1e-3
H, D_EMB, D_DEC = 8, 4, 11  # hidden, stub embedding, 10 channels + given flag


class StubDecEmbedding(nn.Module):
    """Stand-in for ModularEmbedding: linear map of the target covariates."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D_DEC, D_EMB)

    def forward(self, X):
        return self.lin(X)


def make_model(update=None, fused=True, seed=0, update_map="mlp"):
    """Bare delta-mode, monotone gru_per_var ProTEncoder (rollout parts only)."""
    torch.manual_seed(seed)
    m = ProTEncoder.__new__(ProTEncoder)
    nn.Module.__init__(m)
    m.head_type, m.out_dim = "gru_per_var", 2
    m.dec_val_idx, m.dec_pos_idx, m.dec_var_idx = VAL, POS, VAR
    m.n_dec_vars, m.rnn_n_layers, m.d_hidden = 2, 1, H
    # n_ctx_queries > 1: context (B, 6) is split into one (B, 3) per variable
    m.n_ctx_queries, m.rnn_ctx_feed, m.rnn_fused = 2, False, fused
    m.rnn_output_mode, m.rnn_monotone, m.rnn_delta_max = "delta", True, None
    m.rnn_delta_var_min, m.sigma_eps = 1e-3, 1e-3
    m.rnn_absorbing_fail, m.fail_threshold, m.fail_shift = False, None, None
    m.feedback_shift, m.feedback_clamp_max = SHIFT, 1.5
    m.last_step_deltas = None
    m.h0_projection = nn.Linear(3, H)
    m.rnn = nn.ModuleList([nn.GRU(D_EMB + 1, H, batch_first=True) for _ in range(2)])
    m.rnn_head = nn.ModuleList([nn.Linear(H, 2) for _ in range(2)])
    for hd in m.rnn_head:
        hd.bias.data[0] = -3.0
    m.dec_embedding = StubDecEmbedding()
    m.rnn_update = update
    if update == "innovation":
        m.update_cell = InnovationUpdate(2, H, var_floor=1e-3,
                                        update_map=update_map)
    return m


def make_batch(n=3, seed=1):
    """n samples, two variables of E increasing observed positions."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        blocks = []
        for v in (1, 2):
            blk = np.zeros((E, 10), dtype=np.float32)
            blk[:, POS] = np.arange(1, E + 1)
            blk[:, VAR] = v
            blk[:, VAL] = np.sort(rng.uniform(0.0, 0.9, E))
            blocks.append(blk)
        out.append(np.concatenate(blocks))
    return torch.from_numpy(np.stack(out))


def rollout(model, Y, n_show=0):
    """Per-variable rollout; the first n_show positions of each variable are
    given (teacher forced) to the model."""
    given = (Y[:, :, POS] <= n_show) & (Y[:, :, VAR] > 0)
    dec = torch.cat([Y.clone(), given.float().unsqueeze(-1)], dim=-1)
    mask = (~given).unsqueeze(-1) if n_show > 0 else None
    ctx = torch.randn(Y.shape[0], 6, generator=torch.Generator().manual_seed(7))
    return model._gru_rollout_per_var(ctx, dec, mask)


def _randomize_update(m, seed=3):
    """Open the near-identity gates so the update has a visible effect."""
    torch.manual_seed(seed)
    for g in m.update_cell.gates:
        nn.init.normal_(g.weight, std=0.3)
        nn.init.constant_(g.bias, 1.0)
    if m.update_cell.update_map == "film":
        nn.init.normal_(m.update_cell.A2, std=0.3)   # non-identity FiLM

MAPS = pytest.mark.parametrize("update_map", ["mlp", "film"])

# ---------------------------------------------------------- shared GRU helper

def test_gru_cell_matches_torch():
    torch.manual_seed(0)
    cells = [nn.GRUCell(5, H) for _ in range(2)]
    B = 4
    x, h = torch.randn(2 * B, 5), torch.randn(2 * B, H)
    W_ih, W_hh, b_ih, b_hh = stack_gru_params(cells)
    xg = torch.bmm(x.view(2, B, 5), W_ih.transpose(1, 2)).view(2 * B, -1) \
        + b_ih.repeat_interleave(B, 0)
    hg = hidden_gates(h, W_hh.transpose(1, 2), b_hh.repeat_interleave(B, 0), 2)
    ref = torch.cat([cells[v](x[v * B:(v + 1) * B], h[v * B:(v + 1) * B])
                     for v in range(2)])
    assert torch.allclose(gru_cell(xg, hg, h), ref, atol=1e-6)


# --------------------------------------------------------- flag off = legacy

def test_flag_off_has_no_update_params():
    assert not any("update_cell" in k for k in make_model(None).state_dict())


def test_flag_off_fused_equals_reference():
    Y = make_batch()
    for n_show in (0, 5):
        a = rollout(make_model(None, fused=True), Y, n_show)
        b = rollout(make_model(None, fused=False), Y, n_show)
        assert torch.allclose(a, b, atol=1e-5)


@MAPS
def test_update_inactive_without_given_positions(update_map):
    # same predict weights, nothing given: the update never fires -> legacy
    Y = make_batch()
    legacy, upd = make_model(None), make_model("innovation", update_map=update_map)
    upd.load_state_dict({**upd.state_dict(), **legacy.state_dict()})
    assert torch.allclose(rollout(legacy, Y, 0), rollout(upd, Y, 0), atol=1e-6)

# ---------------------------------------------------------- update path

@MAPS
def test_update_fused_equals_reference(update_map):
    Y = make_batch()
    a_m = make_model("innovation", update_map=update_map, fused=True)
    _randomize_update(a_m)
    b_m = copy.deepcopy(a_m)
    b_m.rnn_fused = False
    assert torch.allclose(rollout(a_m, Y, 5), rollout(b_m, Y, 5), atol=1e-5)


@MAPS
def test_no_leakage_of_current_value(update_map):
    # output at position t must not depend on y_t (only on earlier givens)
    Y = make_batch(n=1)
    m = make_model("innovation", update_map=update_map)
    _randomize_update(m)
    with torch.no_grad():
        base = rollout(m, Y, n_show=E)
        t = 4                                 # variable 1, position 5
        Y2 = Y.clone()
        Y2[0, t, VAL] += 0.05
        out = rollout(m, Y2, n_show=E)
    assert torch.allclose(out[0, :t + 1], base[0, :t + 1], atol=1e-6)
    assert not torch.allclose(out[0, t + 1:E], base[0, t + 1:E])


@MAPS
def test_measurement_changes_later_increments(update_map):
    # a given measurement must move h, hence LATER increments (slope)
    Y = make_batch(n=1)
    m = make_model("innovation", update_map=update_map)
    _randomize_update(m)
    with torch.no_grad():
        rollout(m, Y, n_show=4)
        d0 = m.last_step_deltas[0, :E, 0].clone()
        Y2 = Y.clone()
        Y2[0, 3, VAL] += 0.2                  # last given position of var 1
        rollout(m, Y2, n_show=4)
        d1 = m.last_step_deltas[0, :E, 0]
    assert torch.allclose(d0[:4], d1[:4], atol=1e-6)
    assert (d1[4:] - d0[4:]).abs().max() > 1e-4


@MAPS
def test_update_gradients_finite_and_nonzero(update_map):
    m = make_model("innovation", update_map=update_map)
    out = rollout(m, make_batch(), n_show=6)
    out[:, :, 0].sum().backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert m.update_cell.gates[0].weight.grad.abs().sum() > 0
    assert m.update_cell.raw_v_meas.grad is not None


@MAPS
def test_variance_after_update_is_vmeas(update_map):
    # first step after the last given position: prior var = v_meas + d_var
    m = make_model("innovation", update_map=update_map)
    with torch.no_grad():
        out = rollout(m, make_batch(n=1), n_show=3)
    sp = nn.functional.softplus
    d_sigma = sp(m.last_step_deltas[0, 3, 1]) + m.sigma_eps
    sigma = sp(out[0, 3, 1]) + m.sigma_eps
    v_exp = m.update_cell.v_meas() + d_sigma ** 2
    assert torch.allclose(sigma ** 2, v_exp, rtol=1e-3)



@MAPS
def test_init_near_identity_and_map_grads(update_map):
    # gate ~ sigmoid(-3): the corrected state stays close to h_minus at init
    torch.manual_seed(0)
    u = InnovationUpdate(2, H, update_map=update_map)
    h = torch.rand(2 * 4, H) * 1.6 - 0.8
    z, v, zt = torch.rand(8), torch.rand(8) + 0.1, torch.rand(8)
    act = torch.ones(8, dtype=torch.bool)
    out = u.step_fused(h, z, v, zt, act)
    assert (out - h).abs().max() < 0.1
    out.sum().backward()
    # film: A2 is zero-init, so A1 gets gradient only after the first step
    first = u.W1 if update_map == 'mlp' else u.A2
    assert first.grad is not None and first.grad.abs().sum() > 0

