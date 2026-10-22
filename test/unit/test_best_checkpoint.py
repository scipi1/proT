"""
Unit tests for BestCheckpointCallback.

Covers the guard that prevents post-fit trainer.validate()/trainer.test()
calls from corrupting the tracked best epoch/metrics/checkpoint.
"""

from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning.trainer.states import TrainerFn

from proT.training.callbacks.training_callbacks import BestCheckpointCallback


def _make_trainer_mock(fn, val_mae, epoch=5, sanity_checking=False):
    """Minimal trainer mock for the callback hooks."""
    trainer = MagicMock()
    trainer.state.fn = fn
    trainer.sanity_checking = sanity_checking
    trainer.logged_metrics = {"val_mae": torch.tensor(val_mae)}
    trainer.current_epoch = epoch
    return trainer


class TestBestCheckpointCallbackGuard:
    """on_validation_epoch_end must only track during fit."""

    def test_update_during_fit(self, tmp_path):
        cb = BestCheckpointCallback(str(tmp_path), monitor="val_mae", mode="min")
        trainer = _make_trainer_mock(TrainerFn.FITTING, 0.1, epoch=5)

        cb.on_validation_epoch_end(trainer, MagicMock())

        assert cb.best_metric_value == pytest.approx(0.1)
        assert cb.best_epoch == 5
        assert cb.best_checkpoint_path is not None
        trainer.save_checkpoint.assert_called_once()

    @pytest.mark.parametrize("fn", [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING])
    def test_no_update_outside_fit(self, tmp_path, fn):
        cb = BestCheckpointCallback(str(tmp_path), monitor="val_mae", mode="min")
        trainer = _make_trainer_mock(fn, 0.1, epoch=5)

        cb.on_validation_epoch_end(trainer, MagicMock())

        assert cb.best_metric_value == float("inf")
        assert cb.best_epoch == 0
        assert cb.best_checkpoint_path is None
        trainer.save_checkpoint.assert_not_called()

    def test_no_update_during_sanity_check(self, tmp_path):
        """The sanity check runs with state.fn == FITTING on the untrained model;
        its metrics (no train_loss) must never be cached as best."""
        cb = BestCheckpointCallback(str(tmp_path), monitor="val_mae", mode="min")
        trainer = _make_trainer_mock(TrainerFn.FITTING, 0.01, epoch=0, sanity_checking=True)

        cb.on_validation_epoch_end(trainer, MagicMock())

        assert cb.best_metric_value == float("inf")
        assert cb.best_checkpoint_path is None
        trainer.save_checkpoint.assert_not_called()

    def test_post_fit_validate_does_not_corrupt_best(self, tmp_path):
        """Simulate fit (improving then worsening) followed by a post-fit validate."""
        cb = BestCheckpointCallback(str(tmp_path), monitor="val_mae", mode="min")

        # fit: epoch 2 is the best
        for epoch, val_mae in [(0, 0.5), (1, 0.3), (2, 0.2), (3, 0.4)]:
            trainer = _make_trainer_mock(TrainerFn.FITTING, val_mae, epoch=epoch)
            cb.on_validation_epoch_end(trainer, MagicMock())

        assert cb.best_epoch == 2
        assert cb.best_metric_value == pytest.approx(0.2)

        # post-fit validate on the best checkpoint: numerical noise could make
        # val_mae marginally lower, but the tracked state must not change
        trainer = _make_trainer_mock(TrainerFn.VALIDATING, 0.199, epoch=4)
        cb.on_validation_epoch_end(trainer, MagicMock())

        assert cb.best_epoch == 2
        assert cb.best_metric_value == pytest.approx(0.2)
