"""Leak-free fraction reveal mask (fraction_reveal_mask)."""
import math

import torch

from proT.training.forecasters.target_mask import fraction_reveal_mask

POS, VAR, VAL = 0, 1, 2
NAN = float("nan")


def _target(n_obs=(40, 100), L_var=100, pad=5):
    """B=2, two variables of L_var cycles each, observed for n_obs[b]
    cycles (censored NaN values after), plus a padded tail."""
    B, L = len(n_obs), 2 * L_var + pad
    y = torch.full((B, L, 3), NAN)
    for b, n in enumerate(n_obs):
        for v in (1, 2):
            s = (v - 1) * L_var
            y[b, s:s + L_var, POS] = torch.arange(L_var).float()
            y[b, s:s + L_var, VAR] = v
            y[b, s:s + n, VAL] = 0.1
    return y


def _shown(y, mask, b, v):
    m = y[b, :, VAR] == v
    return int((~mask[b, :, 0] & m).sum())


def test_window_independent_of_failure_time():
    y = _target()
    mask = fraction_reveal_mask(y, 0.2, POS, VAL, VAR)
    for b in range(2):          # failed (40 obs) and healthy (100 obs)
        for v in (1, 2):        # each variable counted on its own
            assert _shown(y, mask, b, v) == 20


def test_censored_positions_hidden_and_padding_untouched():
    y = _target()
    mask = fraction_reveal_mask(y, 0.5, POS, VAL, VAR)[:, :, 0]
    cens = torch.isnan(y[:, :, VAL]) & ~torch.isnan(y[:, :, POS])
    shown_cens = ~mask & cens
    assert not shown_cens.any()
    pad = torch.isnan(y[:, :, POS])
    assert not mask[pad].any()


def test_extremes_and_tensor_bound():
    y = _target()
    valid = ~torch.isnan(y[:, :, POS])
    assert fraction_reveal_mask(y, 0.0, POS, VAL, VAR)[:, :, 0][valid].all()
    cens = torch.isnan(y[:, :, VAL]) & valid
    full = fraction_reveal_mask(y, 1.0, POS, VAL, VAR)[:, :, 0]
    assert torch.equal(full, cens)  # all observed shown, censored hidden
    mask = fraction_reveal_mask(y, torch.tensor([0.1, 0.3]), POS, VAL, VAR)
    assert _shown(y, mask, 0, 1) == 10 and _shown(y, mask, 1, 2) == 30


def test_no_var_channel():
    y = _target()[:, :100]
    mask = fraction_reveal_mask(y, 0.25, POS, VAL, None)
    assert int((~mask[0, :, 0]).sum()) == 25 == int((~mask[1, :, 0]).sum())
    assert not math.isnan(float(mask.float().sum()))
