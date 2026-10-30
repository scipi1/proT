"""Unit tests for the probabilistic I-spline head (spline_head.py).

CPU, tiny tensors, seconds runtime.
"""
import numpy as np
import torch

from proT.core.spline_head import SplineHead, ispline_basis

B, V, D, L = 3, 2, 8, 20          # batch, variables, ctx dim, length per var


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, V, D, generator=g)
    pos = torch.arange(1, L + 1, dtype=torch.float32).repeat(V).expand(B, -1).clone()
    var_ids = torch.arange(1, V + 1).repeat_interleave(L).expand(B, -1).clone()
    return ctx, pos, var_ids


def make_head(**kw):
    torch.manual_seed(0)
    return SplineHead(n_vars=V, ctx_dim=D, n_basis=10, t_max=L, **kw)


def test_ispline_basis_properties():
    I = ispline_basis(12, n_grid=256, knot_power=2.0)
    assert I.shape == (256, 12)
    assert np.allclose(I[0], 0.0, atol=1e-8) and np.allclose(I[-1], 1.0)
    assert (np.diff(I, axis=0) >= -1e-10).all()


def test_prior_mean_and_variance_monotone():
    head = make_head()
    ctx, pos, var_ids = make_inputs()
    given = torch.zeros(B, V * L, dtype=torch.bool)
    mu, var = head(ctx, pos, var_ids, torch.zeros(B, V * L), given, z0=-6.9)
    for v in range(V):
        m = var_ids[0] == v + 1
        assert (mu[:, m].diff(dim=1) >= -1e-6).all()
        assert (var[:, m].diff(dim=1) >= -1e-6).all()
    assert torch.isfinite(mu).all() and (var > 0).all()


def test_conditioning_matches_dense_gaussian():
    """Information-form update equals the textbook joint-Gaussian posterior."""
    head = make_head(rank=3)
    ctx, pos, var_ids = make_inputs()
    m, S, r = head.params(ctx)
    phi = head.basis(pos)
    dz = torch.linspace(0, 2, V * L).expand(B, -1)
    w = torch.zeros(B, V * L)
    w[:, :6] = 1.0                                      # first 6 obs of var 1
    m_post, S_post = SplineHead.condition(m[:, 0], S[:, 0], r[:, 0], phi, dz, w)
    for b in range(B):
        Po = phi[b, :6].double()
        Sb, mb = S[b, 0].double(), m[b, 0].double()
        Sy = Po @ Sb @ Po.T + r[b, 0].double() * torch.eye(6, dtype=torch.float64)
        G = Sb @ Po.T @ torch.linalg.inv(Sy)
        m_ref = mb + G @ (dz[b, :6].double() - Po @ mb)
        S_ref = Sb - G @ Po @ Sb
        assert torch.allclose(m_post[b].double(), m_ref, atol=1e-4)
        assert torch.allclose(S_post[b].double(), S_ref, atol=1e-5)


def test_conditioning_tracks_observations_and_shrinks_variance():
    head = make_head(rank=2)
    ctx, pos, var_ids = make_inputs()
    z0 = 0.0
    z_true = 0.05 * pos                                 # target curve
    given = torch.zeros(B, V * L, dtype=torch.bool)
    given[:, :10] = True                                # half of var 1
    mu0, var0 = head(ctx, pos, var_ids, z_true, torch.zeros_like(given), z0)
    mu1, var1 = head(ctx, pos, var_ids, z_true, given, z0)
    err0 = (mu0[:, :10] - z_true[:, :10]).abs().mean()
    err1 = (mu1[:, :10] - z_true[:, :10]).abs().mean()
    assert err1 < err0
    assert (var1[:, :L] <= var0[:, :L] + 1e-6).all()
    # variable 2 is untouched by variable-1 measurements
    assert torch.allclose(mu1[:, L:], mu0[:, L:]) and torch.allclose(var1[:, L:], var0[:, L:])


def test_gradients_finite_with_conditioning():
    head = make_head()
    ctx, pos, var_ids = make_inputs()
    ctx.requires_grad_(True)
    z = 0.05 * pos
    given = torch.zeros(B, V * L, dtype=torch.bool)
    given[:, :5] = True
    given[:, L:L + 5] = True
    mu, var = head(ctx, pos, var_ids, z, given, 0.0)
    nll = 0.5 * (var.log() + (z - mu) ** 2 / var).mean()
    nll.backward()
    for p in head.parameters():
        assert torch.isfinite(p.grad).all()
    assert torch.isfinite(ctx.grad).all()


def test_protencoder_spline_forward_wiring():
    """ProTEncoder._spline_forward: t_max inference, shapes, conditioning."""
    import torch.nn as nn
    from proT.core.model_encoder import ProTEncoder
    POS, VAR, VAL = 1, 2, 3
    m = ProTEncoder.__new__(ProTEncoder)
    nn.Module.__init__(m)
    m.head_type, m.n_dec_vars = "spline_per_var", V
    m.dec_pos_idx, m.dec_var_idx, m.dec_val_idx = POS, VAR, VAL
    m.feedback_shift, m.sigma_eps = 1e-3, 1e-3
    m.rnn_absorbing_fail, m.fail_threshold = False, None
    m.last_step_deltas = None
    m.spline_head = SplineHead(n_vars=V, ctx_dim=D, n_basis=10)
    _, pos, var_ids = make_inputs()
    trg = torch.zeros(B, V * L, 6)
    trg[:, :, POS], trg[:, :, VAR] = pos, var_ids.float()
    trg[:, :, VAL] = 0.01 * pos
    trg[:, -2:, VAL] = float("nan")                     # censored tail
    ctx = torch.randn(B, V * D)                         # flattened queries
    out0 = m._spline_forward(ctx, trg, None)
    assert out0.shape == (B, V * L, 2) and torch.isfinite(out0).all()
    assert float(m.spline_head.t_max) == L
    mask = torch.ones(B, V * L, 1, dtype=torch.bool)
    mask[:, :8] = False                                 # first 8 given
    out1 = m._spline_forward(ctx, trg, mask)
    assert torch.isfinite(out1).all()
    assert not torch.allclose(out0[:, :L], out1[:, :L])


def test_padding_positions():
    head = make_head()
    ctx, pos, var_ids = make_inputs()
    pos[:, -3:] = float("nan")
    var_ids[:, -3:] = 0
    mu, var = head(ctx, pos, var_ids, torch.zeros_like(pos.nan_to_num()),
                   torch.zeros(B, V * L, dtype=torch.bool), 0.0)
    assert torch.isfinite(mu).all() and torch.isfinite(var).all()
