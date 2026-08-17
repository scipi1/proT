"""
Integration tests for best-checkpoint evaluation in the k-fold trainer.

With best=True (default), the metrics reported in kfold_summary.json must come
from the best (val_mae) checkpoint for BOTH validation and test, and must be
consistent with the fold's best_metrics.json. With best=False, the summary
reports final-epoch metrics and records no best checkpoint path.
"""

import json
import os
import shutil
import tempfile

import pandas as pd
import pytest
from omegaconf import OmegaConf

from proT.training.trainer import trainer
from proT.training.experiment_control import update_config

from .utils import modify_config_for_fast_testing, DATA_DIR, ROOT_DIR

CONFIG_PATH = (
    ROOT_DIR / "experiments" / "training" / "tests" / "ok"
    / "test_baseline_proT_ishigami_sum" / "config_proT_ishigami_v5_2.yaml"
)

N_FOLDS = 2
MAX_EPOCHS = 3


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs, cleanup after test."""
    temp_dir = tempfile.mkdtemp(prefix="best_ckpt_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def _load_fast_config():
    config = OmegaConf.load(CONFIG_PATH)
    config_dev = modify_config_for_fast_testing(
        config, max_epochs=MAX_EPOCHS, k_fold=N_FOLDS, max_data_size=40
    )
    return update_config(config_dev)


def _read_summary(save_dir):
    with open(os.path.join(save_dir, "kfold_summary.json")) as f:
        return json.load(f)


def _read_best_metrics(save_dir, fold):
    with open(os.path.join(save_dir, f"k_{fold}", "best_metrics.json")) as f:
        return json.load(f)


def _val_mae_curve_min(save_dir, fold):
    """Minimum val_mae over the training curve logged in metrics.csv."""
    csv_path = os.path.join(save_dir, f"k_{fold}", "logs", "csv", "version_0", "metrics.csv")
    csv = pd.read_csv(csv_path)
    return pd.to_numeric(csv["val_mae"], errors="coerce").min()


def test_best_checkpoint_consistency(temp_output_dir):
    """best=True: summary metrics come from the best checkpoint (val and test)."""
    config = _load_fast_config()

    result_df = trainer(
        config=config,
        data_dir=str(DATA_DIR),
        save_dir=temp_output_dir,
        cluster=False,
        experiment_tag="test_best_ckpt",
        plot_pred_check=False,
        debug=False,
        best=True,
    )

    assert len(result_df) == N_FOLDS
    summary = _read_summary(temp_output_dir)

    for fold in range(N_FOLDS):
        fold_res = summary["fold_results"][str(fold)]

        # best checkpoint path recorded and best_epoch kept as diagnostics
        assert fold_res["best_checkpoint_path"] is not None
        assert "best_epoch" in fold_res["metrics"]

        # summary metrics identical to best_metrics.json (val AND test)
        best_metrics = _read_best_metrics(temp_output_dir, fold)
        for key in ("val_mae", "val_loss", "test_mae", "test_loss"):
            assert fold_res["metrics"][key] == pytest.approx(best_metrics[key]), key

        # reported val_mae is the minimum of the training curve
        curve_min = _val_mae_curve_min(temp_output_dir, fold)
        assert fold_res["metrics"]["val_mae"] == pytest.approx(curve_min, rel=1e-4)


def test_final_epoch_metrics_when_best_false(temp_output_dir):
    """best=False: summary reports final-epoch metrics, no best path recorded."""
    config = _load_fast_config()

    result_df = trainer(
        config=config,
        data_dir=str(DATA_DIR),
        save_dir=temp_output_dir,
        cluster=False,
        experiment_tag="test_best_ckpt",
        plot_pred_check=False,
        debug=False,
        best=False,
    )

    assert len(result_df) == N_FOLDS
    summary = _read_summary(temp_output_dir)

    for fold in range(N_FOLDS):
        fold_res = summary["fold_results"][str(fold)]
        assert fold_res["best_checkpoint_path"] is None
        assert "best_epoch" not in fold_res["metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
