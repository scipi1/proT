"""Unit tests for the termination-aware censored loss ("fail mode"), the
monotone GRU head (softplus-cumsum) and the absorbing fail-stop override.

Layout convention (dyconex seq2seq): target tensor (B, L, F) with channels
position=1, variable=2, value=3; two variable blocks (ids 1, 2) of
expected_len positions each. CPU, tiny tensors, seconds runtime.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from proT.training.losses import CensoredLogNormalNLL
from proT.core.model_encoder import ProTEncoder

POS, VAR, VAL = 1, 2, 3      # channel indices in the target tensor
E = 10                        # expected sequence length per variable
THRESH, SHIFT, MARGIN = 1.0, 1e-3, 0.8


def make_sample(var_specs):
    """One sample (L, F) from {var_id: (values, n_obs)}: values (Lv,) array,
    entries after n_obs are NaN (censored). Block length = len(values)."""
    blocks = []
    for v, (vals, n_obs) in var_specs.items():
        vals = np.asarray(vals, dtype=np.float32).copy()
        Lv = len(vals)
        vals[n_obs:] = np.nan
        blk = np.zeros((Lv, 10), dtype=np.float32)
        blk[:, POS] = np.arange(1, Lv + 1)
        blk[:, VAR] = v
        blk[:, VAL] = vals
        blocks.append(blk)
    return np.concatenate(blocks, axis=0)


def make_batch(cases):
    return torch.from_numpy(np.stack([make_sample(c) for c in cases]))


def classify(Y):
    return CensoredLogNormalNLL.classify_termination(
        Y[:, :, VAL], Y[:, :, VAR], Y[:, :, POS], E, THRESH, MARGIN)


def loss_fn(**kw):
    return CensoredLogNormalNLL(threshold=THRESH, shift=SHIFT, **kw)

# ---------------------------------------------------------------- loss: masks

def test_fully_observed_sample_untouched():
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    m = classify(Y)
    assert m["active"].sum() == 2 * E
    assert not m["anchor"].any() and not m["survival"].any()
    assert m["honest"].sum() == 0


def honest_batch():
    """One honestly-failing sample: var1 reaches 0.9 by obs 7 (trigger), var2
    lower; both censored afterwards."""
    v1 = np.concatenate([np.linspace(0.1, 0.9, 7), np.zeros(E - 7)])
    v2 = np.concatenate([np.linspace(0.1, 0.5, 7), np.zeros(E - 7)])
    return make_batch([{1: (v1, 7), 2: (v2, 7)}])


def test_honest_failure_masks_and_boundary():
    # var1 climbs to 0.9 by obs 7 and is censored afterwards; var2 (lower) too
    Y = honest_batch()
    m = classify(Y)
    assert m["honest"].item() == 1.0
    # trigger = var1 (higher last value); anchor at its last observed (pos 7)
    anch = torch.nonzero(m["anchor"][0], as_tuple=True)[0]
    assert len(anch) == 1 and Y[0, anch[0], VAR] == 1 and Y[0, anch[0], POS] == 7
    # survival at the trigger's first censored position (pos 8)
    surv = torch.nonzero(m["survival"][0], as_tuple=True)[0]
    assert len(surv) == 1 and Y[0, surv[0], VAR] == 1 and Y[0, surv[0], POS] == 8
    # active = observed + the single survival position
    assert m["active"][0].sum() == 7 + 7 + 1


def test_early_interruption_fully_masked():
    # both variables stop at obs 7 but the trigger stays below margin*threshold
    Y = make_batch([{1: (np.linspace(0.1, 0.5, E), 7), 2: (np.linspace(0.1, 0.4, E), 7)}])
    m = classify(Y)
    assert m["honest"].item() == 0.0
    assert not m["anchor"].any() and not m["survival"].any()
    assert m["active"][0].sum() == 7 + 7  # tails masked


def test_contradictory_termination_masked():
    # var1 terminated early, var2 full-length: same device -> treat as artifact
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), 7), 2: (np.linspace(0.1, 0.8, E), E)}])
    m = classify(Y)
    assert m["honest"].item() == 0.0
    assert not m["anchor"].any() and not m["survival"].any()
    assert m["active"][0].sum() == 7 + E


def test_interior_nan_gap_is_not_termination():
    # var1 has a NaN gap at position 5 but runs to the expected length
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    Y[0, 4, VAL] = float("nan")  # interior gap in var1 block
    m = classify(Y)
    assert not m["anchor"].any() and not m["survival"].any()
    assert m["active"][0].sum() == 2 * E - 1  # only the gap is excluded


def test_expected_len_is_configurable():
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), 7), 2: (np.linspace(0.1, 0.9, E), 7)}])
    m7 = CensoredLogNormalNLL.classify_termination(
        Y[:, :, VAL], Y[:, :, VAR], Y[:, :, POS], 7, THRESH, MARGIN)
    # with expected_len=7 the sequences are "full-length" -> survived
    assert m7["honest"].item() == 0.0 and not m7["anchor"].any()


# ---------------------------------------------------------------- loss: values

def _rand_out(B, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, generator=g), torch.randn(B, L, generator=g)


def test_backward_compat_flags_off():
    # legacy mode (no expected_len) must match the old behavior bit-wise
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), 7), 2: (np.linspace(0.1, 0.5, E), 7)}])
    mu, rs = _rand_out(1, 2 * E)
    tgt = Y[:, :, VAL]
    assert loss_fn().forward(mu, rs, tgt).item() == \
        loss_fn(expected_len=E).forward(mu, rs, tgt).item()  # masks=None legacy path


def test_masked_positions_have_no_gradient():
    fn = loss_fn(expected_len=E, fail_anchor=1.0)
    Y = honest_batch()
    tgt = Y[:, :, VAL]
    masks = classify(Y)
    mu1, rs = _rand_out(1, 2 * E)
    mu2 = mu1.clone()
    mu2[0, ~masks["active"][0]] = 100.0  # perturb inactive positions only
    l1 = fn.forward(mu1, rs, tgt, masks=masks)
    l2 = fn.forward(mu2, rs, tgt, masks=masks)
    assert torch.allclose(l1, l2)


def test_anchor_term_zero_at_threshold():
    fn = loss_fn(expected_len=E, fail_anchor=2.0)
    Y = honest_batch()
    tgt = Y[:, :, VAL]
    masks = classify(Y)
    mu, rs = _rand_out(1, 2 * E)
    mu[masks["anchor"]] = math.log(THRESH + SHIFT)  # perfect boundary
    l_anchored = fn.forward(mu, rs, tgt, masks=masks)
    l_noanchor = loss_fn(expected_len=E, fail_anchor=0.0).forward(
        mu, rs, tgt, masks=masks)
    assert abs(l_anchored.item() - l_noanchor.item()) < 1e-6  # anchor contributes 0
    # moving mu away from the threshold must increase the loss
    mu_off = mu.clone()
    mu_off[masks["anchor"]] -= 1.0
    assert fn.forward(mu_off, rs, tgt, masks=masks).item() > l_anchored.item()


# ------------------------------------------------------------------ model head

D_DEC = 11  # 10 target channels + given indicator (appended by the forecaster)


def make_model(rnn_monotone=False, rnn_absorbing_fail=False, fail_threshold=None,
               rnn_delta_max=None):
    """Bare ProTEncoder with only the parts _gru_rollout_per_var needs."""
    m = ProTEncoder.__new__(ProTEncoder)
    nn.Module.__init__(m)
    m.head_type = "gru_per_var"
    m.out_dim = 2
    m.dec_val_idx, m.dec_pos_idx, m.dec_var_idx = VAL, POS, VAR
    m.n_dec_vars = 2
    m.rnn_n_layers = 1
    m.d_hidden = 8
    m.h0_projection = nn.Linear(6, 2 * 8)
    m.rnn = nn.ModuleList([nn.GRU(4 + 1, 8) for _ in range(2)])
    m.rnn_head = nn.ModuleList([nn.Linear(8, 2) for _ in range(2)])

    class StubDecEmbedding(nn.Module):
        """Stand-in for ModularEmbedding: linear map of the target covariates."""
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D_DEC, 4)

        def forward(self, X):
            return self.lin(X)

    m.dec_embedding = StubDecEmbedding()
    m.feedback_shift = SHIFT
    m.rnn_monotone = rnn_monotone
    m.rnn_absorbing_fail = rnn_absorbing_fail
    m.rnn_delta_max = rnn_delta_max
    m.fail_threshold = fail_threshold
    m.fail_shift = SHIFT if fail_threshold is not None else None
    return m


def rollout(model, Y, training=False, force_all_given=False):
    dec = Y.clone()
    dec[:, :, VAL] = torch.nan_to_num(dec[:, :, VAL])  # forecaster zeroes values
    dec = torch.cat([dec, torch.zeros(Y.shape[0], Y.shape[1], 1)], dim=-1)
    model.training = training
    ctx = torch.randn(Y.shape[0], 6, generator=torch.Generator().manual_seed(123))
    mask = (torch.zeros(Y.shape[0], Y.shape[1], 1, dtype=torch.bool)
            if force_all_given else None)
    if training:
        return model._gru_rollout_per_var(ctx, dec, mask)
    with torch.no_grad():
        return model._gru_rollout_per_var(ctx, dec, mask)


def test_monotone_rollout_nondecreasing():
    torch.manual_seed(0)
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    out = rollout(make_model(rnn_monotone=True), Y)
    for v in (1, 2):
        m = Y[0, :, VAR] == v
        mu_v = out[0, m, 0]
        assert torch.all(mu_v[1:] >= mu_v[:-1]), f"var {v}: mu decreased"
    # non-monotone baseline must be allowed to decrease (sanity: seed produces
    # at least one decrease somewhere with random weights)
    out_free = rollout(make_model(rnn_monotone=False), Y)
    assert torch.allclose(out_free[..., 0], out_free[..., 0])  # runs


def test_monotone_first_step_free():
    # first step: base = 0 (no previous value), so y1 = softplus(delta); with
    # delta bias 0.1 the loss-facing channel is mu1 = log(softplus(0.1) + shift)
    m = make_model(rnn_monotone=True)
    for head in m.rnn_head:
        nn.init.zeros_(head.weight)
        head.bias.data[:] = torch.tensor([0.1, 0.0])
    Y = make_batch([{1: (np.zeros(E), E), 2: (np.zeros(E), E)}])
    out = rollout(m, Y)
    m1 = Y[0, :, VAR] == 1
    import torch.nn.functional as Fn
    assert abs(out[0, m1, 0][0].item()
               - math.log(Fn.softplus(torch.tensor(0.1)).item() + SHIFT)) < 1e-4
    # later steps: base = own previous prediction, increment >= 0 -> increasing
    mu_v = out[0, m1, 0]
    assert torch.all(mu_v[1:] > mu_v[:-1])


def test_increment_base_is_previous_value():
    # under full forcing the increment is added to the REAL previous value:
    # y_t = real[t-1] + softplus(delta), delta = 0.1 fixed via head bias
    m = make_model(rnn_monotone=True)
    for head in m.rnn_head:
        nn.init.zeros_(head.weight)
        head.bias.data[:] = torch.tensor([0.1, 0.0])
    vals = np.linspace(0.05, 0.9, E)
    Y = make_batch([{1: (vals, E), 2: (vals * 0.5, E)}])
    out = rollout(m, Y, force_all_given=True)
    import torch.nn.functional as Fn
    inc = Fn.softplus(torch.tensor(0.1)).item()
    m1 = Y[0, :, VAR] == 1
    mu1 = out[0, m1, 0]
    for t in range(1, E):
        expect = math.log(vals[t - 1] + inc + SHIFT)
        assert abs(mu1[t].item() - expect) < 1e-4, f"t={t}: {mu1[t].item()} vs {expect}"


def test_increment_base_is_detached():
    # the increment base must never carry gradients (no accumulation chain)
    m = make_model(rnn_monotone=True)
    seen = []
    orig = m._monotone_step

    def spy(pred, base):
        seen.append(base.requires_grad)
        return orig(pred, base)

    m._monotone_step = spy
    m.train()
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    out = rollout(m, Y, training=True)
    out.sum().backward()
    assert seen and not any(seen), "increment base carries gradients"


def test_monotone_long_rollout_finite():
    """Regression test for the log-space integrator explosion: a full-length
    200-position rollout with random init must stay finite (the feedback is
    y itself, never exp(mu))."""
    torch.manual_seed(1)
    Y = make_batch([{1: (np.linspace(0.1, 0.9, 200), 200),
                     2: (np.linspace(0.1, 0.8, 200), 200)}])
    out = rollout(make_model(rnn_monotone=True), Y)
    assert torch.isfinite(out).all(), "non-finite outputs in long rollout"


def test_fail_stop_override():
    m = make_model(rnn_absorbing_fail=True, fail_threshold=THRESH)
    # var1 crosses immediately (median ~ exp(2) > 1), var2 never (median ~ 0)
    nn.init.zeros_(m.rnn_head[0].weight)
    m.rnn_head[0].bias.data[:] = torch.tensor([2.0, 0.0])
    nn.init.zeros_(m.rnn_head[1].weight)
    m.rnn_head[1].bias.data[:] = torch.tensor([-5.0, 0.0])
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    out = rollout(m, Y, training=False)
    dead = Y[0, :, POS] > 1  # crossing at pos 1 of var1 -> shared stop at 1
    assert torch.allclose(out[0, dead, 0],
                          torch.full_like(out[0, dead, 0], math.log(SHIFT)))
    assert torch.allclose(out[0, dead, 1],
                          torch.full_like(out[0, dead, 1], -20.0))
    # the crossing step itself keeps its prediction (median must reach 1.0)
    assert out[0, Y[0, :, POS] == 1, 0].max() > 0


def test_fail_stop_eval_only():
    m = make_model(rnn_absorbing_fail=True, fail_threshold=THRESH)
    nn.init.zeros_(m.rnn_head[0].weight)
    m.rnn_head[0].bias.data[:] = torch.tensor([2.0, 0.0])
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    out = rollout(m, Y, training=True)  # train mode: no override
    assert not (out[..., 0] == math.log(SHIFT)).any()


def test_fail_stop_no_crossing_unchanged():
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    torch.manual_seed(0)
    m1 = make_model(rnn_absorbing_fail=True, fail_threshold=1e6)  # never crosses
    torch.manual_seed(0)
    m2 = make_model(rnn_absorbing_fail=False)
    o1, o2 = rollout(m1, Y), rollout(m2, Y)
    # random init rarely crosses 1.0 -> identical outputs
    assert torch.allclose(o1, o2)


def test_gradient_flow_finite():
    m = make_model(rnn_monotone=True, rnn_absorbing_fail=True)
    Y = honest_batch()
    m.train()
    dec = Y.clone()
    dec[:, :, VAL] = torch.nan_to_num(dec[:, :, VAL])
    dec = torch.cat([dec, torch.zeros(1, 2 * E, 1)], dim=-1)
    out = m._gru_rollout_per_var(torch.randn(1, 6), dec, None)
    fn = loss_fn(expected_len=E, fail_anchor=1.0)
    masks = classify(Y)
    loss = fn(out[:, :, 0], out[:, :, 1], Y[0, :, VAL].unsqueeze(0), masks=masks)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert m.rnn_head[0].weight.grad.abs().sum() > 0


# ------------------------------------------------- bounded increments + clamp

def test_increment_bounded():
    # delta = delta_max * sigmoid(raw): increments never exceed delta_max,
    # even for extreme head outputs; monotonicity preserved
    DMAX = 0.05
    m = make_model(rnn_monotone=True, rnn_delta_max=DMAX)
    for head in m.rnn_head:
        nn.init.zeros_(head.weight)
        head.bias.data[:] = torch.tensor([50.0, 0.0])  # adversarial: saturated
    Y = make_batch([{1: (np.linspace(0.1, 0.9, 200), 200),
                     2: (np.linspace(0.1, 0.8, 200), 200)}])
    out = rollout(m, Y)
    assert torch.isfinite(out).all()
    for v in (1, 2):
        y_v = torch.exp(out[0, Y[0, :, VAR] == v, 0]) - SHIFT
        d = y_v[1:] - y_v[:-1]
        assert (d >= -1e-4).all(), "monotonicity violated"
        # exp/log round-trip in float32 -> loose tolerance on the bound
        assert (d <= DMAX + 1e-3).all(), "increment exceeds delta_max"
        # total drift over 200 saturated steps is bounded by 200 * delta_max
        assert y_v[-1] <= 200 * DMAX + DMAX + 1e-6


def test_feedback_value_clamped():
    # feedback is capped at 1.5x the failure threshold (runaway guard)
    m = make_model(fail_threshold=1.0)
    capped = m._feedback_value(torch.tensor([5.0]))     # exp(5)-s >> 1.5
    assert capped.item() == 1.5
    low = m._feedback_value(torch.tensor([math.log(0.5 + SHIFT)]))
    assert abs(low.item() - 0.5) < 1e-6                 # below cap: untouched
    m2 = make_model(fail_threshold=None)                # no threshold: no clamp
    assert abs(m2._feedback_value(torch.tensor([5.0])).item()
               - (math.exp(5.0) - SHIFT)) < 1e-3


def test_increment_softplus_when_unbounded():
    # legacy: rnn_delta_max=None keeps softplus increments
    m = make_model(rnn_monotone=True, rnn_delta_max=None)
    for head in m.rnn_head:
        nn.init.zeros_(head.weight)
        head.bias.data[:] = torch.tensor([0.1, 0.0])
    Y = make_batch([{1: (np.zeros(E), E), 2: (np.zeros(E), E)}])
    out = rollout(m, Y)
    import torch.nn.functional as Fn
    m1 = Y[0, :, VAR] == 1
    assert abs(out[0, m1, 0][0].item()
               - math.log(Fn.softplus(torch.tensor(0.1)).item() + SHIFT)) < 1e-4


def test_freemean_arm_no_increment_logic():
    """freemean ablation arm: rnn_monotone=False -> _monotone_step is a
    pass-through and mu is the raw head output (no base added)."""
    m = make_model(rnn_monotone=False)
    pred = torch.randn(4, 2)
    base = torch.ones(4)
    assert torch.equal(m._monotone_step(pred, base), pred)
    # and the rollout is NOT monotone-constrained (free mean)
    torch.manual_seed(3)
    Y = make_batch([{1: (np.linspace(0.1, 0.9, E), E), 2: (np.linspace(0.1, 0.8, E), E)}])
    out = rollout(m, Y)
    assert torch.isfinite(out).all()

