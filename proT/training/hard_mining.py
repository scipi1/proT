"""
Hard-example batch mining for imbalanced regression.

Rationale:
- With small batches and extremely rare informative samples (e.g. defective
  parts with steep IST slopes), most batches contain no minority sample at
  all and loss reweighting (see sample_weights.py) cannot help there.
- HardMiningBatchSampler reserves a fixed fraction of every batch for the
  currently worst-fitted samples, drawn uniformly (without replacement within
  a batch) from a "hard pool": the top-pool_frac samples by EMA per-sample
  loss. The rest of the batch is filled with a random permutation stream, so
  every sample is still seen once per epoch and hard samples are oversampled
  on top.
- This is the batch-composition analog of OHEM (Shrivastava et al., 2016),
  prioritized experience replay (Schaul et al., 2015) and active bias
  (Chang et al., 2017). No importance-sampling debiasing is applied: the bias
  toward hard samples is intentional; validation and test stay untouched, so
  val/test metrics remain comparable across runs.

Wiring (all keyed off config["training"]["loss_hard_mining"]):
- trainer.py creates one HardMiningState per fold (local train indices) and
  injects it into both the DataModule (sampler side) and the forecaster
  (loss-memory side); the Trainer must reload the train dataloader every
  epoch (reload_dataloaders_every_n_epochs=1) so the sampler sees the
  updated memory.
- The forecaster updates the memory in training_step from the per-sample
  losses it already computes for the adaptive sample weighter, and sets
  state.epoch at the end of each epoch.
- The memory is deliberately not checkpointed (same policy as
  AdaptiveLossWeighter): on resume the pool is cold and the sampler falls
  back to plain shuffling until the memory refills.

Note on workers: the sampler is iterated in the main process (workers only
receive indices), so the shared state is never pickled to workers.
"""

import numpy as np
from torch.utils.data import Dataset, Sampler


class HardMiningState:
    """
    Shared state for hard-example mining: per-sample EMA loss memory.

    One scalar EMA loss per (fold-local) training sample. Written by the
    forecaster after each training step, read by the HardMiningBatchSampler
    at the start of each epoch.

    Args:
        n_samples: Number of training samples in the current fold.
        frac: Fraction of each batch reserved for hard samples (0 < frac < 1).
        pool_frac: Fraction of the training set forming the hard pool
            (top-loss samples); 0 < pool_frac <= 1.
        start_epoch: First epoch with hard mining (plain shuffling before);
            use a warmup to let the representation form.
        momentum: EMA momentum for the per-sample loss memory.
    """

    def __init__(self, n_samples: int, frac: float = 0.25, pool_frac: float = 0.10,
                 start_epoch: int = 0, momentum: float = 0.9):
        assert 0.0 < frac < 1.0, "frac must be in (0, 1)"
        assert 0.0 < pool_frac <= 1.0, "pool_frac must be in (0, 1]"
        self.loss = np.full(n_samples, np.nan, dtype=np.float64)
        self.frac = frac
        self.pool_frac = pool_frac
        self.start_epoch = start_epoch
        self.momentum = momentum
        self.epoch = 0  # set by the forecaster at the end of each epoch

    def update(self, idx: np.ndarray, losses: np.ndarray) -> None:
        """EMA update of the loss memory for the given sample indices."""
        idx = np.asarray(idx, dtype=np.int64)
        losses = np.asarray(losses, dtype=np.float64)
        old = self.loss[idx]
        self.loss[idx] = np.where(np.isnan(old), losses,
                                  self.momentum * old + (1.0 - self.momentum) * losses)

    def hard_pool(self) -> np.ndarray:
        """Indices of the top-pool_frac samples by EMA loss (empty if cold)."""
        valid = ~np.isnan(self.loss)
        if not valid.any():
            return np.empty(0, dtype=np.int64)
        k = max(1, int(np.ceil(self.pool_frac * len(self.loss))))
        order = np.argsort(np.where(valid, self.loss, -np.inf))[::-1]
        return order[:k]

    def pool_stats(self):
        """(mean pool loss, mean non-pool loss) for logging; None if cold."""
        pool = self.hard_pool()
        if len(pool) == 0:
            return None
        mask = np.ones(len(self.loss), dtype=bool)
        mask[pool] = False
        rest = self.loss[mask & ~np.isnan(self.loss)]
        pool_loss = float(np.mean(self.loss[pool]))
        rest_loss = float(np.mean(rest)) if len(rest) else float("nan")
        return pool_loss, rest_loss


class IndexedTensorDataset(Dataset):
    """
    TensorDataset wrapper that also yields the (fold-local) sample index.

    Batches become (X, Y, idx) instead of (X, Y); forecasters that support
    hard mining unpack the optional third element. The underlying
    TensorDataset's `tensors` attribute is forwarded so existing hooks
    (e.g. tail-threshold computation) keep working.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, i

    @property
    def tensors(self):
        return self.base.tensors


class HardMiningBatchSampler(Sampler):
    """
    Batch sampler reserving a fraction of each batch for hard samples.

    Each batch contains n_hard = round(state.frac * batch_size) samples drawn
    uniformly without replacement from the hard pool, filled up to batch_size
    from a random permutation of all indices (skipping within-batch
    duplicates). Before state.start_epoch, or while the loss memory is cold,
    the batches are a plain shuffled partition (equivalent to shuffle=True).

    Iteration is deterministic given (seed, state.epoch, memory contents), so
    multi-worker dataloading (which replicates the sampler iterator) yields
    identical batches in every worker.

    Args:
        n_samples: Number of samples in the (fold-local) training set.
        batch_size: Batch size.
        state: Shared HardMiningState.
        seed: Base seed; the per-epoch generator uses seed + 1000 * epoch.
    """

    def __init__(self, n_samples: int, batch_size: int, state: HardMiningState,
                 seed: int = 42):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.state = state
        self.seed = seed

    def _n_hard(self, pool: np.ndarray) -> int:
        """Hard slots per batch; clamped so at least one random slot remains."""
        if self.state.epoch < self.state.start_epoch or len(pool) == 0:
            return 0
        return min(int(round(self.state.frac * self.batch_size)),
                   len(pool), self.batch_size - 1)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.state.epoch)
        n, B = self.n_samples, self.batch_size
        pool = self.state.hard_pool()
        n_hard = self._n_hard(pool)

        stream = rng.permutation(n)
        pos = 0
        while pos < n:
            batch = []
            if n_hard > 0:
                hard = rng.choice(pool, size=n_hard, replace=False)
                batch.extend(int(h) for h in hard)
            in_batch = set(batch)
            while pos < n and len(batch) < B:
                idx = int(stream[pos])
                pos += 1
                if idx not in in_batch:
                    batch.append(idx)
            yield batch

    def __len__(self) -> int:
        n_hard = self._n_hard(self.state.hard_pool())
        per_batch = self.batch_size - n_hard  # random stream consumed per batch
        return (self.n_samples + per_batch - 1) // per_batch
