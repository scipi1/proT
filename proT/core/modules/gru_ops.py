"""Shared GRU-cell math for the fused (batched over variables) rollouts.

Used by the predict cell (ProTEncoder._fused_setup) and by the innovation
update cell (proT/core/innovation_update.py), so both run the exact
torch.nn.GRU equations with their own stacked per-variable parameters.
"""
import torch


def gru_cell(xg, hg, h):
    """One GRU cell update from precomputed gate pre-activations.

    torch.nn.GRU equations (gate order r, z, n):
        r = sig(x_r + h_r), z = sig(x_z + h_z), n = tanh(x_n + r * h_n)
        h' = (1 - z) * n + z * h

    Args:
        xg: input projection W_ih x + b_ih (N, 3H)
        hg: hidden projection W_hh h + b_hh (N, 3H)
        h: previous hidden state (N, H)

    Returns:
        new hidden state (N, H)
    """
    xr, xz, xn = xg.chunk(3, -1)
    hr, hz, hn = hg.chunk(3, -1)
    r = torch.sigmoid(xr + hr)
    z = torch.sigmoid(xz + hz)
    n = torch.tanh(xn + r * hn)
    return (1 - z) * n + z * h


def stack_gru_params(cells):
    """Stack per-variable single-layer GRU parameters.

    Accepts nn.GRU (layer 0) or nn.GRUCell modules. Stacking is done on the
    fly, so gradients flow to the original parameters.

    Returns:
        W_ih (V, 3H, D_in), W_hh (V, 3H, H), b_ih (V, 3H), b_hh (V, 3H)
    """
    sfx = "_l0" if hasattr(cells[0], "weight_ih_l0") else ""
    return tuple(torch.stack([getattr(c, f"{name}{sfx}") for c in cells])
                 for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))


def hidden_gates(h, W_hhT, b_hh, V):
    """Hidden projection for variable-major rows (row v*B + b).

    Args:
        h: (V*B, H); W_hhT: (V, H, 3H); b_hh: (V*B, 3H) (already expanded)
    """
    B = h.size(0) // V
    return torch.bmm(h.view(V, B, -1), W_hhT).view(V * B, -1) + b_hh
