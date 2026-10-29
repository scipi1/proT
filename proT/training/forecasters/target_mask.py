"""Leak-free target reveal mask shared by the forecasters."""
import torch


def fraction_reveal_mask(dec_input: torch.Tensor, bound, pos_idx: int,
                         val_idx: int, var_idx=None) -> torch.Tensor:
    """(B, L, 1) bool mask of target positions to hide (True = hidden).

    The fraction is taken on the TIMELINE, not on the measurements: per
    sample and per target variable, the window is the first
    round(bound * L_v) valid positions, where L_v counts ALL valid positions
    (observed AND censored). Observed positions inside the window are shown;
    everything else (outside the window, or censored) is hidden. Counting only observed values would make the end of the fed
    window equal to bound * t_fail and leave the censored tail flagged as
    given, leaking the failure time into the given indicator.

    Args:
        dec_input: target tensor (B, L, D)
        bound: fraction in [0, 1], float or per-sample tensor (B,)
        pos_idx: position channel (NaN = padding)
        val_idx: value channel (NaN = censored / not measured)
        var_idx: variable channel (ids >= 1, 0/NaN = padding) or None for a
            single-variable target
    """
    pos = dec_input[:, :, pos_idx]
    valid = ~torch.isnan(pos)
    if var_idx is None:
        var = valid.long()
    else:
        var = torch.nan_to_num(dec_input[:, :, var_idx], nan=0.0).long()
        valid = valid & (var > 0)
        var = torch.where(valid, var, torch.zeros_like(var))
    if torch.is_tensor(bound):
        bound = bound.to(device=valid.device, dtype=torch.float32).view(-1, 1)
    hide = torch.zeros_like(valid)
    for v in torch.unique(var[valid]).tolist():
        m = valid & (var == v)
        rank = m.cumsum(dim=1) - 1
        n_show = (bound * m.sum(dim=1, keepdim=True).float()).round().long()
        hide |= m & (rank >= n_show)
    # censored positions are never given (they carry no measurement)
    hide |= valid & torch.isnan(dec_input[:, :, val_idx])
    return hide.unsqueeze(-1)
