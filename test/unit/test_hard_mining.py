"""
Unit tests for proT/training/hard_mining.py.

Covers: EMA loss memory updates, hard-pool selection, pool statistics,
batch-sampler warmup fallback, batch composition (hard slots from the pool,
no within-batch duplicates, full coverage via the random stream), sampler
length consistency, determinism per (seed, epoch), and the indexed dataset
wrapper.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from proT.training.hard_mining import (
    HardMiningState,
    HardMiningBatchSampler,
    IndexedTensorDataset,
)

N = 100       # samples
B = 10        # batch size
FRAC = 0.3    # 3 hard slots per batch
POOL = 0.1    # top 10% = 10 samples
START = 5     # mining starts at epoch 5


def _state(start_epoch=START, frac=FRAC, pool_frac=POOL, momentum=0.9):
    return HardMiningState(n_samples=N, frac=frac, pool_frac=pool_frac,
                           start_epoch=start_epoch, momentum=momentum)


def _filled_state():
    """State with a known loss landscape: loss = index (sample 99 worst)."""
    st = _state()
    st.update(np.arange(N), np.arange(N, dtype=float))
    return st


# --- HardMiningState ---------------------------------------------------------

def test_update_first_write_then_ema():
    st = _state(momentum=0.5)
    st.update(np.array([0, 1]), np.array([2.0, 4.0]))
    assert st.loss[0] == 2.0 and st.loss[1] == 4.0  # first write: raw value
    st.update(np.array([0]), np.array([4.0]))
    assert st.loss[0] == 0.5 * 2.0 + 0.5 * 4.0      # then EMA blend
    assert np.isnan(st.loss[2])                     # untouched stays cold


def test_hard_pool_cold_then_top_k():
    st = _state()
    assert len(st.hard_pool()) == 0                 # cold memory -> empty pool
    st = _filled_state()
    pool = st.hard_pool()
    assert len(pool) == int(np.ceil(POOL * N))      # top 10%
    assert set(pool) == set(range(N - 10, N))       # highest losses win


def test_pool_stats():
    st = _state()
    assert st.pool_stats() is None
    st = _filled_state()
    pool_loss, rest_loss = st.pool_stats()
    assert pool_loss == np.mean(np.arange(N - 10, N, dtype=float))
    assert rest_loss == np.mean(np.arange(N - 10, dtype=float))


# --- HardMiningBatchSampler --------------------------------------------------

def test_warmup_is_plain_shuffle():
    st = _filled_state()
    st.epoch = START - 1  # before start_epoch
    sampler = HardMiningBatchSampler(N, B, st, seed=0)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert len(batches) == N // B == len(sampler)
    assert sorted(flat) == list(range(N))           # exact partition, no oversampling


def test_cold_memory_is_plain_shuffle():
    st = _state()  # no updates -> cold
    st.epoch = START + 10
    sampler = HardMiningBatchSampler(N, B, st, seed=0)
    flat = [i for b in sampler for i in b]
    assert sorted(flat) == list(range(N))


def test_active_mining_composition():
    st = _filled_state()
    st.epoch = START
    sampler = HardMiningBatchSampler(N, B, st, seed=0)
    batches = list(sampler)
    pool = set(st.hard_pool())
    n_hard = round(FRAC * B)

    # length consistency (used by Lightning for the epoch loop)
    assert len(batches) == len(sampler)

    seen = set()
    for b in batches[:-1]:
        assert len(b) == B
    for b in batches:
        # last batch may be partial (random stream exhausted, drop_last=False)
        assert 0 < len(b) <= B
        assert len(set(b)) == len(b)                # no within-batch duplicates
        # the first n_hard slots are the hard draw and must come from the pool
        assert set(b[:n_hard]) <= pool
        seen.update(b)
    assert seen == set(range(N))                    # random stream covers all

    # hard samples are oversampled: they appear more often than non-pool ones
    flat = [i for b in batches for i in b]
    n_pool_seen = sum(1 for i in flat if i in pool)
    assert n_pool_seen > len(batches) * n_hard  # pool also in random stream


def test_determinism_per_epoch():
    st1, st2 = _filled_state(), _filled_state()
    st1.epoch = st2.epoch = START
    b1 = list(HardMiningBatchSampler(N, B, st1, seed=42))
    b2 = list(HardMiningBatchSampler(N, B, st2, seed=42))
    assert b1 == b2                                 # same seed+epoch -> same batches
    st2.epoch = START + 1
    b3 = list(HardMiningBatchSampler(N, B, st2, seed=42))
    assert b1 != b3                                 # different epoch -> different batches


def test_frac_clamped_below_batch_size():
    st = HardMiningState(n_samples=N, frac=0.99, pool_frac=1.0, start_epoch=0)
    st.update(np.arange(N), np.arange(N, dtype=float))
    sampler = HardMiningBatchSampler(N, B, st, seed=0)
    n_hard = sampler._n_hard(st.hard_pool())
    assert n_hard == B - 1                          # at least one random slot


def test_pool_smaller_than_hard_slots():
    st = HardMiningState(n_samples=N, frac=0.5, pool_frac=0.01, start_epoch=0)
    st.update(np.arange(N), np.arange(N, dtype=float))
    sampler = HardMiningBatchSampler(N, B, st, seed=0)
    n_hard = sampler._n_hard(st.hard_pool())
    assert n_hard == 1                              # pool has a single sample
    for b in list(sampler)[:-1]:
        assert len(b) == B and len(set(b)) == B


# --- DataModule integration --------------------------------------------------

def test_train_dataloader_with_hard_mining():
    """train_dataloader yields (X, Y, idx) batches when a state is injected."""
    from proT.training.dataloader import ProcessDataModule
    dm = ProcessDataModule(data_dir="", input_file="", target_file="",
                           batch_size=B, num_workers=0, data_format="float32")
    dm.train_ds = TensorDataset(torch.randn(N, 4), torch.randn(N, 2))

    st = _filled_state()
    st.epoch = START
    dm.hard_mining_state = st
    xb, yb, idxb = next(iter(dm.train_dataloader()))
    assert xb.shape == (B, 4) and yb.shape == (B, 2) and idxb.shape == (B,)

    dm.hard_mining_state = None  # back to the standard 2-tuple loader
    xb2, yb2 = next(iter(dm.train_dataloader()))
    assert xb2.shape == (B, 4) and yb2.shape == (B, 2)


# --- IndexedTensorDataset ----------------------------------------------------

def test_indexed_dataset():
    x = torch.randn(N, 4)
    y = torch.randn(N, 2)
    ds = IndexedTensorDataset(TensorDataset(x, y))
    assert len(ds) == N
    xi, yi, idx = ds[7]
    assert torch.equal(xi, x[7]) and torch.equal(yi, y[7]) and idx == 7
    assert len(ds.tensors) == 2                     # forwarded for forecaster hooks
