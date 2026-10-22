"""
Unit tests for group masking in encoder self-attention.

Group masking restricts attention so that tokens can only influence tokens
sharing the same group id (e.g., the same PCB layer / "occurrence").
It composes with the causal mask and the NAIM-style missing-value mask.
"""
from os.path import dirname, abspath
import sys

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

import torch

from proT.core.modules.attention import ScaledDotAttention, AttentionLayer, build_group_mask
from proT.core.modules.extra_layers import UniformAttentionMask
from proT.core.modules.embedding import ModularEmbedding


# ---------------------------------------------------------------------------
# build_group_mask
# ---------------------------------------------------------------------------

def test_build_group_mask_single_head():
    """Mask is -inf across groups, 0 within groups."""
    g = torch.tensor([[0., 0., 1., 1.]])  # (B=1, L=4)
    M = build_group_mask(g, n_heads=1)

    assert M.shape == (1, 4, 4)
    # within group 0
    assert M[0, 0, 1] == 0.0 and M[0, 1, 0] == 0.0
    # within group 1
    assert M[0, 2, 3] == 0.0 and M[0, 3, 2] == 0.0
    # across groups
    assert torch.isinf(M[0, 0, 2]) and M[0, 0, 2] < 0
    assert torch.isinf(M[0, 3, 0]) and M[0, 3, 0] < 0
    # diagonal is allowed
    assert torch.all(torch.diagonal(M[0]) == 0.0)


def test_build_group_mask_multi_head():
    """Multi-head mask is expanded across the head dimension."""
    g = torch.tensor([[0., 1., 0.], [2., 2., 0.]])  # (B=2, L=3)
    M = build_group_mask(g, n_heads=4)

    assert M.shape == (2, 4, 3, 3)
    # same mask for every head
    for h in range(4):
        assert torch.equal(M[:, h], build_group_mask(g, n_heads=1))


def test_build_group_mask_nan_sentinel():
    """NaN group ids form their own group (sentinel -1)."""
    g = torch.tensor([[0., float("nan"), float("nan"), 1.]])
    M = build_group_mask(g, n_heads=1)

    # the two NaN tokens attend to each other
    assert M[0, 1, 2] == 0.0 and M[0, 2, 1] == 0.0
    # but not to regular groups
    assert torch.isinf(M[0, 0, 1]) and torch.isinf(M[0, 1, 3])


def test_build_group_mask_accepts_3d_input():
    """Accepts (B, L, 1) input like the position tensor."""
    g = torch.tensor([[[0.], [0.], [1.]]])  # (B=1, L=3, 1)
    M = build_group_mask(g, n_heads=1)

    assert M.shape == (1, 3, 3)
    assert M[0, 0, 1] == 0.0
    assert torch.isinf(M[0, 0, 2])


# ---------------------------------------------------------------------------
# Attention with group mask
# ---------------------------------------------------------------------------

def _make_attention(n_heads=1, d_model=12, d_qk=8):
    return AttentionLayer(
        attention=ScaledDotAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_qk,
        n_heads=n_heads,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0.0,
        dropout_qkv=0.0,
    )


def test_group_mask_blocks_cross_group_attention():
    """Cross-group attention weights are exactly zero; rows sum to 1."""
    torch.manual_seed(0)
    B, L, d_model = 2, 6, 12
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 0., 1., 1., 2., 2.],
                      [0., 1., 0., 1., 0., 1.]])

    att_layer = _make_attention()
    _, score, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=None, mask_miss_q=None,
        pos=None, causal_mask=False,
        group_ids=g, group_mask=True,
    )

    # score shape (B, L, L) for single head
    same_group = g.unsqueeze(-1) == g.unsqueeze(-2)  # (B, L, L)
    assert torch.all(score[~same_group] == 0.0), "cross-group attention must be zero"
    assert torch.allclose(score.sum(dim=-1), torch.ones(B, L), atol=1e-6), \
        "within-group attention rows must sum to 1"
    assert not torch.isnan(score).any()


def test_group_mask_multihead():
    """Group mask applies identically to every head."""
    torch.manual_seed(0)
    B, L, d_model, H = 1, 5, 12, 3
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 0., 1., 1., 1.]])

    att_layer = _make_attention(n_heads=H)
    _, score, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=None, mask_miss_q=None,
        pos=None, causal_mask=False,
        group_ids=g, group_mask=True,
    )

    assert score.shape == (B, H, L, L)
    same_group = (g.unsqueeze(-1) == g.unsqueeze(-2)).unsqueeze(1)  # (B, 1, L, L)
    assert torch.all(score[~same_group.expand(-1, H, -1, -1)] == 0.0)


def test_group_mask_combined_with_naim_mask():
    """Group mask composes with the NAIM missing-value key mask."""
    torch.manual_seed(0)
    B, L, d_model = 1, 4, 12
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 0., 0., 1.]])

    # token 1 (same group as 0 and 2) has a missing value
    mask_miss = torch.zeros(B, L, 1, dtype=torch.bool)
    mask_miss[0, 1, 0] = True

    att_layer = _make_attention()
    _, score, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=mask_miss, mask_miss_q=mask_miss,
        pos=None, causal_mask=False,
        group_ids=g, group_mask=True,
    )

    # key 1 is masked for every query (NAIM), key 3 is masked for queries 0-2 (group)
    assert torch.all(score[0, :, 1] == 0.0), "missing-value key must be masked"
    assert torch.all(score[0, :3, 3] == 0.0), "cross-group key must be masked"
    # query 3 can only attend to itself
    assert score[0, 3, 3] > 0.0
    assert torch.all(score[0, 3, :3] == 0.0)
    assert not torch.isnan(score).any()


def test_group_mask_all_masked_row_is_safe():
    """A query whose whole group has missing values yields zero attention, no NaN."""
    torch.manual_seed(0)
    B, L, d_model = 1, 3, 12
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 1., 1.]])

    # tokens 1 and 2 (group 1) are both missing -> query 1 has no valid key
    mask_miss = torch.zeros(B, L, 1, dtype=torch.bool)
    mask_miss[0, 1:, 0] = True

    att_layer = _make_attention()
    out, score, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=mask_miss, mask_miss_q=None,
        pos=None, causal_mask=False,
        group_ids=g, group_mask=True,
    )

    assert not torch.isnan(score).any(), "no NaN allowed"
    assert torch.all(score[0, 1] == 0.0), "fully masked row must be zero attention"
    assert torch.all(score[0, 2] == 0.0)
    # query 0 is unaffected
    assert score[0, 0, 0] > 0.0


def test_group_mask_combined_with_causal_mask():
    """Group mask composes with the causal mask."""
    torch.manual_seed(0)
    B, L, d_model = 1, 4, 12
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 0., 1., 1.]])
    pos = torch.arange(L, dtype=torch.float).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

    att_layer = _make_attention()
    _, score, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=None, mask_miss_q=None,
        pos=pos, causal_mask=True,
        group_ids=g, group_mask=True,
    )

    # causal: no attention to the future
    future = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert torch.all(score[0][future] == 0.0)
    # group: no cross-group attention
    assert score[0, 0, 2] == 0.0 and score[0, 3, 1] == 0.0
    # allowed: same group and past/present
    assert score[0, 1, 0] > 0.0
    assert score[0, 3, 2] > 0.0


def test_group_mask_disabled_is_noop():
    """group_mask=False must reproduce the unmasked result."""
    torch.manual_seed(0)
    B, L, d_model = 1, 4, 12
    x = torch.rand(B, L, d_model)
    g = torch.tensor([[0., 0., 1., 1.]])

    att_layer = _make_attention()
    _, score_masked, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=None, mask_miss_q=None,
        pos=None, causal_mask=False,
        group_ids=g, group_mask=False,
    )
    _, score_plain, _ = att_layer(
        query=x, key=x, value=x,
        mask_miss_k=None, mask_miss_q=None,
        pos=None, causal_mask=False,
    )
    assert torch.allclose(score_masked, score_plain)


# ---------------------------------------------------------------------------
# ModularEmbedding "group" type
# ---------------------------------------------------------------------------

def test_modular_embedding_group_extraction():
    """The 'group' embed type stores the index and adds no embedding module."""
    d_model = 4
    ds_embed = {
        "setting": {},
        "modules": [
            {"idx": 0, "embed": "nn_embedding", "label": "variable",
             "kwargs": {"num_embeddings": 10, "embedding_dim": d_model}},
            {"idx": 1, "embed": "group", "label": "group_occurrence", "kwargs": {}},
        ],
    }

    embed = ModularEmbedding(ds_embed=ds_embed, comps="concat", device="cpu")

    assert embed.group_idx == 1
    assert len(embed.embed_list) == 1, "group feature must not add an embedding module"

    X = torch.rand(2, 5, 3)
    X[:, :, 1] = torch.tensor([0., 0., 1., 1., 2.])
    g = embed.get_group_var(X)

    assert g.shape == (2, 5)
    assert torch.equal(g[0], torch.tensor([0., 0., 1., 1., 2.]))


def test_modular_embedding_no_group_returns_none():
    """Without a 'group' module, get_group_var returns None."""
    d_model = 4
    ds_embed = {
        "setting": {},
        "modules": [
            {"idx": 0, "embed": "nn_embedding", "label": "variable",
             "kwargs": {"num_embeddings": 10, "embedding_dim": d_model}},
        ],
    }
    embed = ModularEmbedding(ds_embed=ds_embed, comps="concat", device="cpu")
    assert embed.get_group_var(torch.rand(2, 5, 3)) is None


if __name__ == "__main__":
    test_build_group_mask_single_head()
    test_build_group_mask_multi_head()
    test_build_group_mask_nan_sentinel()
    test_build_group_mask_accepts_3d_input()
    test_group_mask_blocks_cross_group_attention()
    test_group_mask_multihead()
    test_group_mask_combined_with_naim_mask()
    test_group_mask_all_masked_row_is_safe()
    test_group_mask_combined_with_causal_mask()
    test_group_mask_disabled_is_noop()
    test_modular_embedding_group_extraction()
    test_modular_embedding_no_group_returns_none()
    print("All group mask tests passed!")
