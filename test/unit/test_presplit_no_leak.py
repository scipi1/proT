"""
Regression tests for the pre-split train/test leak.

Bug (fixed): in pre-split mode (train_file + test_file), prepare_data() reset
train_idx/val_idx/test_idx to None. Lightning calls setup() -> prepare_data()
after trainer.py had installed the per-fold split via update_idx(), so
split_ds() fell through to auto_split_ds(), which re-partitioned
all_ds = train (+) test at random. Consequences:
  - the k-fold split was silently discarded (every fold trained on the same
    random_split, differing only in model init),
  - test samples leaked into the training set,
  - the reported test metrics were computed on a train/test mixture.

Run with: pytest test/unit/test_presplit_no_leak.py -v
"""

import shutil
import sys
import tempfile
from os.path import abspath, dirname, join

import numpy as np
import pytest

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from proT.training.dataloader import ProcessDataModule


N_TRAIN = 120
N_TEST = 30
SEQ_LEN = 7
N_FEAT = 4


@pytest.fixture
def presplit_dir():
    """Temp dataset with disjoint, identifiable train/test sample ids."""
    tmp = tempfile.mkdtemp()
    # channel 0 of Y carries a unique sample id: train 0..N_TRAIN-1,
    # test 1000..1000+N_TEST-1, so any leak is directly detectable.
    x_tr = np.random.randn(N_TRAIN, SEQ_LEN, N_FEAT).astype("float32")
    y_tr = np.zeros((N_TRAIN, 2, 4), dtype="float32")
    y_tr[:, :, 0] = np.arange(N_TRAIN)[:, None]

    x_te = np.random.randn(N_TEST, SEQ_LEN, N_FEAT).astype("float32")
    y_te = np.zeros((N_TEST, 2, 4), dtype="float32")
    y_te[:, :, 0] = (1000 + np.arange(N_TEST))[:, None]

    np.savez(join(tmp, "train_data.npz"), x=x_tr, y=y_tr)
    np.savez(join(tmp, "test_data.npz"), x=x_te, y=y_te)
    yield tmp
    shutil.rmtree(tmp)


def make_dm(data_dir, batch_size=8):
    return ProcessDataModule(
        data_dir=data_dir,
        input_file=None,
        target_file=None,
        batch_size=batch_size,
        num_workers=0,
        data_format="float32",
        seed=42,
        train_file="train_data.npz",
        test_file="test_data.npz",
        use_val_split=True,
    )


def ids_of(ds):
    """Sample ids (Y channel 0) contained in a TensorDataset."""
    return set(ds.tensors[1][:, 0, 0].numpy().astype(int).tolist())


def test_setup_preserves_kfold_split(presplit_dir):
    """setup() must not discard a split installed via update_idx()."""
    dm = make_dm(presplit_dir)
    dm.get_ds_len()

    train_idx = list(range(0, 90))
    val_idx = list(range(90, N_TRAIN))
    dm.update_idx(train_idx=train_idx, val_idx=val_idx, test_idx=None)

    assert (len(dm.train_ds), len(dm.val_ds), len(dm.test_ds)) == (90, 30, N_TEST)

    # the Lightning hook: this used to trigger the random re-partition
    dm.setup(stage=None)
    assert (len(dm.train_ds), len(dm.val_ds), len(dm.test_ds)) == (90, 30, N_TEST)

    # idempotent under repeated calls
    dm.setup(stage=None)
    dm.prepare_data()
    assert (len(dm.train_ds), len(dm.val_ds), len(dm.test_ds)) == (90, 30, N_TEST)


def test_no_test_leak_into_train(presplit_dir):
    """Train/val/test must stay disjoint, and test ids must never reach train."""
    dm = make_dm(presplit_dir)
    dm.get_ds_len()
    dm.update_idx(train_idx=list(range(0, 90)),
                  val_idx=list(range(90, N_TRAIN)),
                  test_idx=None)
    dm.setup(stage=None)

    tr, va, te = ids_of(dm.train_ds), ids_of(dm.val_ds), ids_of(dm.test_ds)
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()
    # every test id comes from the test file, none from the train file
    assert all(i >= 1000 for i in te)
    assert all(i < 1000 for i in tr | va)


def test_presplit_without_indices_keeps_files(presplit_dir):
    """Without indices, the pre-split datasets from the files are preserved."""
    dm = make_dm(presplit_dir)
    dm.prepare_data()
    dm.setup(stage=None)

    assert len(dm.train_ds) == N_TRAIN
    assert len(dm.test_ds) == N_TEST
    assert ids_of(dm.train_ds) & ids_of(dm.test_ds) == set()


def test_auto_split_forbidden_in_presplit_mode(presplit_dir):
    """auto_split_ds() must fail loudly rather than silently leak."""
    dm = make_dm(presplit_dir)
    dm.prepare_data()
    with pytest.raises(AssertionError):
        dm.auto_split_ds()


def test_train_steps_per_epoch_match_fold_size(presplit_dir):
    """The train loader must iterate the fold's train split, not a re-split."""
    dm = make_dm(presplit_dir, batch_size=8)
    dm.get_ds_len()
    dm.update_idx(train_idx=list(range(0, 90)),
                  val_idx=list(range(90, N_TRAIN)),
                  test_idx=None)
    dm.setup(stage=None)
    assert len(dm.train_dataloader()) == int(np.ceil(90 / 8))


def test_eval_train_dataloader_is_deterministic(presplit_dir):
    """Prediction on the train split must be reproducible and ordered."""
    dm = make_dm(presplit_dir)
    dm.get_ds_len()
    dm.update_idx(train_idx=list(range(0, 90)),
                  val_idx=list(range(90, N_TRAIN)),
                  test_idx=None)
    dm.setup(stage=None)

    def order():
        return np.concatenate(
            [b[1][:, 0, 0].numpy() for b in dm.eval_train_dataloader()]
        )

    first, second = order(), order()
    assert np.array_equal(first, second)
    # and it follows the dataset order exactly
    assert np.array_equal(first, dm.train_ds.tensors[1][:, 0, 0].numpy())
