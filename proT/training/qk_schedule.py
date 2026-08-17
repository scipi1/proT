"""
Delayed learning of attention query/key projections.

Two mutually exclusive strategies (config key `qk_lr_ramp` selects the ramp,
otherwise `qk_freeze_epochs` applies the hard freeze):

1. Hard freeze: Q/K projections stay frozen (requires_grad=False) for the
   first `qk_freeze_epochs` epochs (see QKFreezeScheduler callback).
2. Linear LR ramp: Q/K projections get a dedicated parameter group whose LR
   ramps linearly from a very small factor of the base LR up to the full
   base LR, reached at `qk_lr_ramp_end_frac` (default 95%) of max_epochs.

Rationale: while routing (Q/K) is frozen or heavily slowed at the start,
the model is forced to organize its embeddings and value/output projections
first; attention routing is learned on top of stable representations.
"""

import torch

from proT.core.modules.attention import AttentionLayer


def _classify_attention(name: str) -> str:
    """
    Classify an AttentionLayer by its qualified module name.

    Returns:
        "cross" (decoder cross-attention), "dec_self" (decoder
        self-attention) or "enc_self" (encoder self-attention).
    """
    if "global_cross_attention" in name:
        return "cross"
    if "global_self_attention" in name:
        return "dec_self"
    return "enc_self"  # encoder 'global_attention'


def collect_qk_params(model: torch.nn.Module, scope: str = "all") -> list:
    """
    Collect query/key projection parameters of all AttentionLayer modules.

    Args:
        model: ProT model (or any module containing AttentionLayers).
        scope: "all" | "self" (encoder + decoder self-attention) | "cross".

    Returns:
        list: List of torch.nn.Parameter (Q/K projection weights and biases).
    """
    assert scope in ("all", "self", "cross"), f"Invalid Q/K scope: {scope}"
    params = []
    for name, module in model.named_modules():
        if not isinstance(module, AttentionLayer):
            continue
        kind = _classify_attention(name)
        if (
            scope == "all"
            or (scope == "self" and kind in ("enc_self", "dec_self"))
            or (scope == "cross" and kind == "cross")
        ):
            params.extend(module.query_projection.parameters())
            params.extend(module.key_projection.parameters())
    return params


def set_requires_grad(params: list, flag: bool) -> None:
    """Set requires_grad on a list of parameters."""
    for p in params:
        p.requires_grad = flag


def build_qk_ramp_lambda(max_epochs: int, end_frac: float = 0.95, start_factor: float = 1e-4):
    """
    Linear LR factor ramp for LambdaLR: start_factor at epoch 0, reaching
    1.0 at end_frac * max_epochs, constant 1.0 afterwards.

    Args:
        max_epochs: Total number of training epochs.
        end_frac: Fraction of training at which the full LR is reached.
        start_factor: Initial LR factor (relative to the group base LR).

    Returns:
        callable: lr_lambda(epoch) -> float, for torch LambdaLR.
    """
    ramp_end = end_frac * max_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch >= ramp_end:
            return 1.0
        return start_factor + (1.0 - start_factor) * (epoch / ramp_end)

    return lr_lambda


def configure_adamw_with_qk_ramp(
    forecaster,
    base_lr: float,
    weight_decay: float,
    scope: str = "all",
    max_epochs: int = 100,
    end_frac: float = 0.95,
    start_factor: float = 1e-4,
):
    """
    AdamW with a dedicated parameter group for Q/K projections whose LR
    ramps linearly from start_factor * base_lr to base_lr, reached at
    end_frac * max_epochs.

    Args:
        forecaster: LightningModule holding the ProT model as `.model`.
        base_lr: Base learning rate (used for all other parameters).
        weight_decay: AdamW weight decay.
        scope: Q/K scope ("all" | "self" | "cross").
        max_epochs: Total number of training epochs.
        end_frac: Fraction of training at which the full LR is reached.
        start_factor: Initial LR factor for the Q/K group.

    Returns:
        ([optimizer], [scheduler_dict]) as expected by configure_optimizers.
    """
    qk_params = collect_qk_params(forecaster.model, scope=scope)
    qk_ids = {id(p) for p in qk_params}
    other_params = [p for p in forecaster.parameters() if id(p) not in qk_ids and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": other_params}, {"params": qk_params}],
        lr=base_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda epoch: 1.0, build_qk_ramp_lambda(max_epochs, end_frac, start_factor)],
    )
    print(
        f"Q/K LR ramp active (scope={scope}): {len(qk_params)} tensors, "
        f"start_factor={start_factor}, full LR at epoch {int(end_frac * max_epochs)}/{max_epochs}"
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
